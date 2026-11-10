from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")
model.export(
    format="engine",
    imgsz=640
    #dynamic=True,  
    #batch=8,  
    workspace=4,  
    #data="coco.yaml",  
)

# Load the exported TensorRT INT8 model
#model = YOLO("yolov8n.engine", task="detect")
