#!/usr/bin/env python3
import csv
import glob
import os
import sys
import time
from collections import deque

from ultralytics import YOLO


CONFIG = {
    "classify": ("yolov8n-cls.engine", "classify", "expanded_imagenet_images"),
    "detect": ("yolov8n.engine", "detect", "expanded_coco_images"),
    "pose": ("yolov8n-pose.engine", "pose", "expanded_coco_images"),
    "segment": ("yolov8n-seg.engine", "segment", "expanded_coco_images"),
    "obb": ("yolov8n-obb.engine", "obb", "expanded_dota_images"),
}


def collect_images(image_dir):
    extensions = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted(
        path
        for path in glob.glob(os.path.join(image_dir, "*"))
        if os.path.isfile(path) and path.lower().endswith(extensions)
    )


def resolve_unique_images(workload, available_count):
    value = os.environ.get(
        f"UNIQUE_IMAGES_{workload.upper()}",
        os.environ.get("UNIQUE_IMAGES", "100"),
    )
    if str(value).lower() in {"auto", "all"}:
        if available_count <= 0:
            raise RuntimeError(f"no images found for workload={workload}")
        return available_count
    return int(value)


def snapshot_result(workload, result):
    snapshot = {
        "path": result.path,
        "speed": dict(result.speed),
    }
    if workload == "classify" and result.probs is not None:
        snapshot["top5"] = list(result.probs.top5)
        snapshot["top5conf"] = result.probs.top5conf.cpu().numpy().copy()
        return snapshot

    if result.boxes is not None and getattr(result.boxes, "data", None) is not None:
        snapshot["boxes"] = result.boxes.data.cpu().numpy().copy()
    if getattr(result, "obb", None) is not None and getattr(result.obb, "data", None) is not None:
        snapshot["obb"] = result.obb.data.cpu().numpy().copy()
    if getattr(result, "keypoints", None) is not None and getattr(result.keypoints, "data", None) is not None:
        snapshot["keypoints"] = result.keypoints.data.cpu().numpy().copy()
    if getattr(result, "masks", None) is not None and getattr(result.masks, "data", None) is not None:
        snapshot["masks"] = result.masks.data.cpu().numpy().copy()
    return snapshot


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in CONFIG:
        raise SystemExit(f"usage: {sys.argv[0]} WORKLOAD OUTPUT_CSV")

    workload, output_csv = sys.argv[1:]
    engine, task, image_dir = CONFIG[workload]
    total_requests = int(os.environ.get("TOTAL_REQUESTS", "5000"))
    pressure_enabled = os.environ.get("PRESSURE_MODE", "0") == "1"
    pressure_retain_items = int(os.environ.get("PRESSURE_RETAIN_ITEMS", "0"))
    pressure_buffer_mb = int(os.environ.get("PRESSURE_BUFFER_MB", "0"))
    available_images = collect_images(image_dir)
    unique_images = resolve_unique_images(workload, len(available_images))
    images = available_images[:unique_images]
    if len(images) != unique_images:
        raise RuntimeError(
            f"expected {unique_images} images, found {len(images)} in {image_dir}"
        )

    model = YOLO(engine, task=task)
    retained_results = deque(maxlen=pressure_retain_items or None)
    pressure_buffer = (
        bytearray(pressure_buffer_mb * 1024 * 1024)
        if pressure_enabled and pressure_buffer_mb > 0
        else None
    )

    # Warm-up is excluded from the 5,000 measured requests.
    for _ in range(5):
        model.predict(images[0], save=False, verbose=False)

    # Optional barrier used by the concurrent experiment. Each container loads
    # and warms up independently, then starts measured requests together.
    start_at_ns = int(os.environ.get("START_AT_NS", "0"))
    while start_at_ns and time.time_ns() < start_at_ns:
        time.sleep(0.001)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    started = time.perf_counter()
    count = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "workload",
                "request_index",
                "source_image_index",
                "source_image",
                "request_started_ns",
                "request_finished_ns",
                "inference_ms",
            ]
        )
        for count in range(1, total_requests + 1):
            image = images[(count - 1) % unique_images]
            request_started_ns = time.time_ns()
            result = model.predict(image, save=False, verbose=False)[0]
            if pressure_enabled and pressure_retain_items > 0:
                retained_results.append(snapshot_result(workload, result))
            request_finished_ns = time.time_ns()
            writer.writerow(
                [
                    workload,
                    count,
                    ((count - 1) % unique_images) + 1,
                    result.path,
                    request_started_ns,
                    request_finished_ns,
                    result.speed["inference"],
                ]
            )

    elapsed = time.perf_counter() - started
    print(f"workload={workload}")
    print(f"unique_images={len(images)}")
    print(f"total_requests={total_requests}")
    print(f"completed_requests={count}")
    print(f"measurement_started_ns={start_at_ns or 'immediate'}")
    print(f"pressure_enabled={pressure_enabled}")
    print(f"pressure_retain_items={pressure_retain_items}")
    print(f"pressure_buffer_mb={pressure_buffer_mb}")
    print(f"retained_results={len(retained_results)}")
    print(f"pressure_buffer_bytes={len(pressure_buffer) if pressure_buffer is not None else 0}")
    print(f"elapsed_sec={elapsed:.6f}")
    print(f"output_csv={output_csv}")
    if count != total_requests:
        raise RuntimeError(f"expected {total_requests} results, got {count}")


if __name__ == "__main__":
    main()
