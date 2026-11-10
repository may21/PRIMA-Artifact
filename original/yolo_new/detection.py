from ultralytics import YOLO

# TensorRT 모델을 불러옵니다
tensorrt_model = YOLO("yolov8n.engine", task='detect')
#model = YOLO("yolov8n.pt", task='detect')
#model.to("cpu")

# 이미지 경로를 불러옵니다
image_path = "./expanded_coco_images"

# 모델을 이미지 경로에 적용하고 GPU에서 실행
result = tensorrt_model.predict(source=image_path)
#result = model(image_path)    
