from ultralytics import YOLO

# Load the exported TensorRT model
#model = YOLO("yolov8n-obb.pt", task='obb')
#model.export(format="engine", imgsz=640)
tensorrt_model = YOLO("yolov8n-obb.engine", task='obb')
#model.to("cuda")


# Load the image path
image_path = "./expanded_dota_images"

result = tensorrt_model.predict(image_path)
#result = model(image_path)


