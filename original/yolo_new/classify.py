from ultralytics import YOLO
  
model = YOLO("yolo26n-cls.pt", task='classify')
model.to("cuda")

# Load the image path
image_path = "./imagenet_images"

result = model(image_path)
