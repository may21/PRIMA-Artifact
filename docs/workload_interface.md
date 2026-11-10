# YOLO workload script interface

The YOLO runners (`run_isolated_baseline.sh`, `run_isolated_memory_sweep.sh`,
`run_concurrent_4mode.sh`, `run_mps50_2mode.sh`) execute five external scripts
inside the Ultralytics container. They are kept with the engines in
`$PRIMA_ARTIFACT_ROOT/yolo/workload/` and are not part of this repository.
Any implementation that follows this contract works with the organizers.

| Script | Engine | Task | Images |
|---|---|---|---|
| `classify.py` | `yolov8n-cls.engine` | classify | `expanded_imagenet_images` |
| `detect.py` | `yolov8n.engine` | detect | `expanded_coco_images` |
| `pose.py` | `yolov8n-pose.engine` | pose | `expanded_coco_images` |
| `segment.py` | `yolov8n-seg.engine` | segment | `expanded_coco_images` |
| `obb.py` | `yolov8n-obb.engine` | obb | `expanded_dota_images` |

Engines: YOLOv8n, 640x640, batch 1, FP32 (`half=False`), static shape,
exported on the target Jetson with `scripts/setup/export_yolo_engines.sh`.

## Inputs

- Working directory: the mounted workload directory
  (`/usr/src/ultralytics/yolo_new` in the container).
- `IMAGE_LIMIT` (environment): number of images to process (100 in the paper).
- `image_lists/{imagenet,coco,dota}_${IMAGE_LIMIT}.txt`: file names to use,
  in order, so that every configuration sees the same images.

## Output on stdout

One line per image, in either of these forms:

```text
image 1/100 /usr/src/ultralytics/yolo_new/expanded_coco_images/000000000139.jpg: 480x640 1 person, 26.6ms
[1/100] 000000000139.jpg preprocess=3.1ms inference=26.6ms postprocess=2.0ms
```

The organizers record only the model inference time, i.e. Ultralytics
`result.speed["inference"]`, which is the latency reported in the paper. The
first form is the Ultralytics verbose line, whose last value is that time.

The last line of stdout is the total elapsed time in seconds as a bare
number (for example `12.53`).
