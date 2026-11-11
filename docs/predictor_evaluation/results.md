# Predictor evaluation

Data: `datasets/predictor_memory_features.csv`, target `Max_cpu_mem_MB`. 1547 rows, 368 distinct configurations (1051 YOLOv8 rows in 352 configurations, 496 non-YOLO rows in 16 configurations).

## Dataset

| family | model | task | runtime | parameters_M | rows | configurations | repeats_per_config | input_px | batch | precision | target_min_MB | target_max_MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non-YOLO | EfficientNet-B0 | cls | trtexec | 5.3 | 62 | 2 | 31-31 | 320 | 1 | fp16/fp32 | 2262 | 2564 |
| non-YOLO | ResNet-50 | cls | trtexec | 25.5 | 62 | 2 | 31-31 | 320 | 1 | fp16/fp32 | 2169 | 2503 |
| non-YOLO | ViT-B/16 | cls | trtexec | 86.6 | 62 | 2 | 31-31 | 320 | 1 | fp16/fp32 | 1951 | 2401 |
| non-YOLO | RetinaNet-R50-FPN | det | PyTorch | 34.1 | 62 | 2 | 31-31 | 320 | 1 | fp16/fp32 | 2950 | 2985 |
| non-YOLO | Faster R-CNN-R50-FPN | det | PyTorch | 41.8 | 62 | 2 | 31-31 | 320 | 1 | fp16/fp32 | 2735 | 2854 |
| non-YOLO | Keypoint R-CNN-R50-FPN | pose | PyTorch | 59.2 | 62 | 2 | 31-31 | 320 | 1 | fp16/fp32 | 3387 | 3465 |
| non-YOLO | DeepLabV3-ResNet50 | seg | trtexec | 42.0 | 62 | 2 | 31-31 | 320 | 1 | fp16/fp32 | 2301 | 2637 |
| non-YOLO | Mask R-CNN-R50-FPN | seg | PyTorch | 44.5 | 62 | 2 | 31-31 | 320 | 1 | fp16/fp32 | 3615 | 3626 |
| YOLOv8 | yolov8n-cls | cls | TensorRT (Ultralytics) | 2.7 | 107 | 36 | 2-3 | 320/480/640 | 1/2/4/8 | fp16/fp32/int8 | 512 | 984 |
| YOLOv8 | yolov8s-cls | cls | TensorRT (Ultralytics) | 6.4 | 108 | 36 | 3-3 | 320/480/640 | 1/2/4/8 | fp16/fp32/int8 | 519 | 980 |
| YOLOv8 | yolov8n | det | TensorRT (Ultralytics) | 3.2 | 107 | 36 | 2-3 | 320/480/640 | 1/2/4/8 | fp16/fp32/int8 | 678 | 832 |
| YOLOv8 | yolov8s | det | TensorRT (Ultralytics) | 11.2 | 108 | 36 | 3-3 | 320/480/640 | 1/2/4/8 | fp16/fp32/int8 | 700 | 828 |
| YOLOv8 | yolov8n-obb | obb | TensorRT (Ultralytics) | 3.1 | 107 | 36 | 2-3 | 320/480/640 | 1/2/4/8 | fp16/fp32/int8 | 1448 | 2214 |
| YOLOv8 | yolov8s-obb | obb | TensorRT (Ultralytics) | 11.4 | 96 | 32 | 3-3 | 320/480/640 | 1/2/4/8 | fp16/fp32/int8 | 1774 | 2197 |
| YOLOv8 | yolov8n-pose | pose | TensorRT (Ultralytics) | 3.3 | 107 | 36 | 2-3 | 320/480/640 | 1/2/4/8 | fp16/fp32/int8 | 678 | 849 |
| YOLOv8 | yolov8s-pose | pose | TensorRT (Ultralytics) | 11.6 | 96 | 32 | 3-3 | 320/480/640 | 1/2/4/8 | fp16/fp32/int8 | 701 | 837 |
| YOLOv8 | yolov8n-seg | seg | TensorRT (Ultralytics) | 3.4 | 107 | 36 | 2-3 | 320/480/640 | 1/2/4/8 | fp16/fp32/int8 | 763 | 2166 |
| YOLOv8 | yolov8s-seg | seg | TensorRT (Ultralytics) | 11.8 | 108 | 36 | 3-3 | 320/480/640 | 1/2/4/8 | fp16/fp32/int8 | 919 | 2330 |

Non-YOLO model names are inferred from the profiling scripts and parameter counts.

## All rows (YOLOv8 + non-YOLO)

**MAE (MB)**

| protocol | mean of other repeats | Random forest | XGBoost | LightGBM | SVR | Linear on weight size (baseline) | Mean (baseline) |
|---|---|---|---|---|---|---|---|
| repeat-measurement noise (reference) | 25.8 |  |  |  |  |  |  |
| random rows 8:2 |  | 23.7 | 25.9 | 28.1 | 384.2 | 626.3 | 861.7 |
| grouped by configuration |  | 64.9 | 60.6 | 65.2 | 441.5 | 637.7 | 888.3 |
| leave one model out |  | 232.5 | 247.6 | 244.2 | 569.1 | 734.7 | 927.2 |
| train YOLOv8, test non-YOLO |  | 1805.5 | 1847.3 | 2015.1 | 1984.7 | 1102.7 | 1728.0 |
| train non-YOLO, test YOLOv8 |  | 1442.3 | 1486.5 | 1485.1 | 1572.9 | 1672.2 | 1728.0 |

**MAPE (%)**

| protocol | mean of other repeats | Random forest | XGBoost | LightGBM | SVR | Linear on weight size (baseline) | Mean (baseline) |
|---|---|---|---|---|---|---|---|
| repeat-measurement noise (reference) | 1.9 |  |  |  |  |  |  |
| random rows 8:2 |  | 1.9 | 2.2 | 2.5 | 31.5 | 53.1 | 79.3 |
| grouped by configuration |  | 4.3 | 4.3 | 4.7 | 32.7 | 51.2 | 76.1 |
| leave one model out |  | 11.2 | 12.1 | 11.9 | 46.7 | 57.3 | 80.4 |
| train YOLOv8, test non-YOLO |  | 64.3 | 64.6 | 71.1 | 69.2 | 38.2 | 59.9 |
| train non-YOLO, test YOLOv8 |  | 180.4 | 185.4 | 185.2 | 195.7 | 206.5 | 212.7 |

**R²**

| protocol | mean of other repeats | Random forest | XGBoost | LightGBM | SVR | Linear on weight size (baseline) | Mean (baseline) |
|---|---|---|---|---|---|---|---|
| repeat-measurement noise (reference) | 0.996 |  |  |  |  |  |  |
| random rows 8:2 |  | 0.998 | 0.998 | 0.997 | 0.750 | 0.437 | -0.015 |
| grouped by configuration |  | 0.991 | 0.992 | 0.991 | 0.688 | 0.434 | -0.027 |
| leave one model out |  | 0.706 | 0.734 | 0.718 | 0.542 | 0.162 | -0.110 |
| train YOLOv8, test non-YOLO |  | -16.218 | -17.481 | -20.435 | -20.252 | -6.063 | -15.387 |
| train non-YOLO, test YOLOv8 |  | -6.501 | -6.948 | -6.950 | -7.778 | -8.788 | -9.387 |

**Share of under-predictions (%)**

| protocol | mean of other repeats | Random forest | XGBoost | LightGBM | SVR | Linear on weight size (baseline) | Mean (baseline) |
|---|---|---|---|---|---|---|---|
| repeat-measurement noise (reference) | 54 |  |  |  |  |  |  |
| random rows 8:2 |  | 54 | 58 | 56 | 45 | 42 | 45 |
| grouped by configuration |  | 52 | 51 | 50 | 50 | 46 | 50 |
| leave one model out |  | 54 | 52 | 58 | 49 | 46 | 49 |
| train YOLOv8, test non-YOLO |  | 100 | 100 | 100 | 100 | 88 | 100 |
| train non-YOLO, test YOLOv8 |  | 0 | 0 | 0 | 0 | 0 | 0 |

## YOLOv8 rows only

**MAE (MB)**

| protocol | mean of other repeats | Random forest | XGBoost | LightGBM | SVR | Linear on weight size (baseline) | Mean (baseline) |
|---|---|---|---|---|---|---|---|
| repeat-measurement noise (reference) | 34.1 |  |  |  |  |  |  |
| random rows 8:2 |  | 34.4 | 34.6 | 37.1 | 352.1 | 479.3 | 492.3 |
| grouped by configuration |  | 48.2 | 50.7 | 54.2 | 344.3 | 481.9 | 488.4 |
| leave one model out |  | 90.5 | 71.7 | 72.0 | 412.1 | 581.7 | 534.3 |
| leave one task out |  | 543.2 | 507.6 | 514.4 | 565.8 | 590.8 | 591.4 |
| leave one scale out |  | 109.9 | 74.1 | 73.9 | 392.6 | 1549.0 | 488.9 |

**MAPE (%)**

| protocol | mean of other repeats | Random forest | XGBoost | LightGBM | SVR | Linear on weight size (baseline) | Mean (baseline) |
|---|---|---|---|---|---|---|---|
| repeat-measurement noise (reference) | 2.7 |  |  |  |  |  |  |
| random rows 8:2 |  | 2.7 | 2.7 | 3.1 | 21.5 | 44.6 | 45.7 |
| grouped by configuration |  | 4.5 | 4.8 | 5.2 | 21.5 | 46.1 | 46.9 |
| leave one model out |  | 7.8 | 6.4 | 6.5 | 28.9 | 55.4 | 51.4 |
| leave one task out |  | 49.5 | 41.3 | 41.4 | 45.1 | 57.1 | 57.0 |
| leave one scale out |  | 11.1 | 6.7 | 6.6 | 25.3 | 162.8 | 47.0 |

**R²**

| protocol | mean of other repeats | Random forest | XGBoost | LightGBM | SVR | Linear on weight size (baseline) | Mean (baseline) |
|---|---|---|---|---|---|---|---|
| repeat-measurement noise (reference) | 0.986 |  |  |  |  |  |  |
| random rows 8:2 |  | 0.989 | 0.989 | 0.989 | -0.023 | 0.034 | -0.003 |
| grouped by configuration |  | 0.983 | 0.982 | 0.982 | -0.026 | 0.012 | -0.003 |
| leave one model out |  | 0.937 | 0.960 | 0.961 | -0.196 | -0.451 | -0.194 |
| leave one task out |  | -0.636 | -0.552 | -0.679 | -0.670 | -0.463 | -0.460 |
| leave one scale out |  | 0.924 | 0.959 | 0.959 | -0.253 | -9.778 | -0.003 |

**Share of under-predictions (%)**

| protocol | mean of other repeats | Random forest | XGBoost | LightGBM | SVR | Linear on weight size (baseline) | Mean (baseline) |
|---|---|---|---|---|---|---|---|
| repeat-measurement noise (reference) | 47 |  |  |  |  |  |  |
| random rows 8:2 |  | 49 | 48 | 49 | 52 | 36 | 36 |
| grouped by configuration |  | 51 | 51 | 50 | 49 | 30 | 31 |
| leave one model out |  | 49 | 48 | 52 | 46 | 31 | 32 |
| leave one task out |  | 43 | 42 | 52 | 51 | 31 | 32 |
| leave one scale out |  | 39 | 45 | 51 | 55 | 51 | 31 |

## Random forest, leave one model out, per held-out model (all rows)

| family | model | rows | actual_mean_MB | predicted_mean_MB | MAE_MB | MAPE_pct | under_pct |
|---|---|---|---|---|---|---|---|
| YOLOv8 | yolov8n | 107.0 | 737.7 | 736.8 | 16.5 | 2.2 | 48.6 |
| YOLOv8 | yolov8n-cls | 107.0 | 723.3 | 699.1 | 41.1 | 5.5 | 61.7 |
| YOLOv8 | yolov8n-obb | 107.0 | 2009.0 | 2108.8 | 115.9 | 6.1 | 9.3 |
| YOLOv8 | yolov8n-pose | 107.0 | 737.0 | 745.9 | 25.7 | 3.5 | 42.1 |
| YOLOv8 | yolov8n-seg | 107.0 | 1267.3 | 1255.4 | 115.8 | 9.7 | 46.7 |
| YOLOv8 | yolov8s | 108.0 | 747.5 | 752.5 | 30.4 | 4.1 | 44.4 |
| YOLOv8 | yolov8s-cls | 108.0 | 701.7 | 725.8 | 46.2 | 7.0 | 32.4 |
| YOLOv8 | yolov8s-obb | 96.0 | 2090.0 | 2027.9 | 90.8 | 4.4 | 75.0 |
| YOLOv8 | yolov8s-pose | 96.0 | 756.9 | 747.6 | 33.8 | 4.4 | 62.5 |
| YOLOv8 | yolov8s-seg | 108.0 | 1428.5 | 1271.2 | 160.7 | 11.7 | 93.5 |
| non-YOLO | DeepLabV3-ResNet50 | 62.0 | 2619.7 | 2584.6 | 47.3 | 1.8 | 96.8 |
| non-YOLO | EfficientNet-B0 | 62.0 | 2540.3 | 606.2 | 1934.1 | 76.1 | 100.0 |
| non-YOLO | Faster R-CNN-R50-FPN | 62.0 | 2794.4 | 3101.1 | 306.6 | 11.0 | 0.0 |
| non-YOLO | Keypoint R-CNN-R50-FPN | 62.0 | 3422.5 | 3461.2 | 42.7 | 1.3 | 37.1 |
| non-YOLO | Mask R-CNN-R50-FPN | 62.0 | 3622.1 | 3055.7 | 566.4 | 15.6 | 100.0 |
| non-YOLO | ResNet-50 | 62.0 | 2427.7 | 2581.4 | 153.8 | 6.4 | 0.0 |
| non-YOLO | RetinaNet-R50-FPN | 62.0 | 2966.9 | 2991.2 | 48.3 | 1.6 | 50.0 |
| non-YOLO | ViT-B/16 | 62.0 | 2334.6 | 782.2 | 1552.5 | 66.5 | 100.0 |

