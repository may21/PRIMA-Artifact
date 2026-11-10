from ultralytics import YOLO
import os

model = YOLO("yolov8n-obb.pt", task='obb')
model.to("cpu")

image_path = "./dota_images"

image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

selected_images = image_files[:25]

results = []
for image in selected_images:
    result = model(image)
    results.append(result)
