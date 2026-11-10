from ultralytics import YOLO
import time
start_time = time.time()
model_obb = YOLO("yolov8n-obb.pt", task='obb')
image_path_obb = "./dota_images"
result_obb = model_obb(image_path_obb)
print("OBB task completed.")
end_time = time.time()
print(f"Sequential execution time: {end_time - start_time:.5f} seconds")

