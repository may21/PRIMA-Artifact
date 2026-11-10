from ultralytics import YOLO
  
# Load the exported TensorRT model
#model = YOLO("yolov8n-cls.pt", task='classify')
#model.export(format="engine" , imgsz=640)
tensorrt_model = YOLO("yolov8n-cls.engine", task='classify')
#model.to("cuda")


# Load the image path
image_path = "./expanded_imagenet_images"

result = tensorrt_model.predict(image_path)
#result = model(image_path)

