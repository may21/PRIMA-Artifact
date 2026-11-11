# YOLO workload scripts

The YOLO runners (`run_isolated_baseline.sh`, `run_isolated_memory_sweep.sh`,
`run_concurrent_4mode.sh`, `run_mps50_2mode.sh`) execute five scripts inside
the Ultralytics container, from the mounted workload directory
(`/usr/src/ultralytics/yolo_new`). The scripts used for the revised paper are
in [`workloads/yolo/`](../workloads/yolo), copied unchanged from orin1;
`scripts/setup/install_workloads.sh` installs them into
`$PRIMA_ARTIFACT_ROOT/yolo/workload/`.

| Script | Engine | Image list | Saved under `runs3/` |
|---|---|---|---|
| `classify.py` | `yolov8n-cls.engine` | `image_lists/imagenet_${IMAGE_LIMIT}.txt` | `cls` |
| `detect.py` | `yolov8n.engine` | `image_lists/coco_${IMAGE_LIMIT}.txt` | `det` |
| `pose.py` | `yolov8n-pose.engine` | `image_lists/coco_${IMAGE_LIMIT}.txt` | `est` |
| `segment.py` | `yolov8n-seg.engine` | `image_lists/coco_${IMAGE_LIMIT}.txt` | `seg` |
| `obb.py` | `yolov8n-obb.engine` | `image_lists/dota_${IMAGE_LIMIT}.txt` | `obb` |

Engines: YOLOv8n, 640x640, batch 1, FP32 (`half=False`), static shape,
exported on the target Jetson with `scripts/setup/export_yolo_engines.sh`.

## Behaviour that determines the measured memory

Each script makes **one** non-streaming call,
`model.predict(<image list>, save=True, project=.../runs3)`. Ultralytics then
keeps the `Results` object of every image, including the full-resolution
input image, until the call returns, and writes an annotated copy of every
image. Per-workload memory therefore grows with each processed image
(paper Figure 3) and is dominated by the input resolution: with the paper's
full-size DOTA images OBB peaks at about 2 GB and cannot finish below 2 GB
(Table 2).

Scripts that predict one image at a time, discard the result, or skip saving
use far less memory. On such a pipeline the five workloads stay below the
device capacity, the per-container limits never bind, and the concurrent
experiments do not show the contention the paper evaluates.

## Image set

The image lists in `workloads/yolo/image_lists/` name the paper's images by
their container path, for example
`/usr/src/ultralytics/yolo_new/expanded_dota_images/001_P0009.png`. They are
the orin1 `expanded_{imagenet,coco,dota}_images` folders: numbered copies of a
fixed image set, with the DOTA images at original resolution (about 4.4 MPix
on average, up to several thousand pixels per side). Resized or tiled DOTA
images change the OBB footprint by about a factor of two.
`scripts/setup/prepare_yolo_inputs.sh` refuses to continue if a listed image
is missing.

## Output on stdout

Ultralytics prints one line per image, whose last value is the inference
time (`result.speed["inference"]`) that the organizers record:

```text
image 1/100 /usr/src/ultralytics/yolo_new/expanded_coco_images/000_000000000676.jpg: 480x640 1 person, 26.6ms
```

The last line is the elapsed time of the `predict()` call in seconds. Each
script names its main thread after the workload (`pthread_setname_np`), so
pidstat lists the process under `classify`, `detect`, `pose`, `segment` or
`obb`. The organizers also accept `[i/N] ... inference=X ms` lines from other
implementations, but such implementations do not reproduce the paper's
memory behaviour.
