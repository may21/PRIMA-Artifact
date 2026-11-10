import subprocess
from ultralytics import YOLO
from glob import glob

def get_res_memory_usage():
    result = subprocess.run(['top', '-b', '-n', '1'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    
    for line in output.split('\n'):
        if 'python3' in line:
            res_value = line.split()[5].lower()
            if 'k' in res_value:
                res_memory = float(res_value.replace('k', '')) / 1024
            elif 'm' in res_value:
                res_memory = float(res_value.replace('m', ''))
            elif 'g' in res_value:
                res_memory = float(res_value.replace('g', '')) * 1024
            else:
                res_memory = float(res_value) / 1024
            return res_memory
    return 0.0

mem_usage_before_model_load = get_res_memory_usage()
print(f"Before loading model - RES Memory usage: {mem_usage_before_model_load:.2f} MB")

model = YOLO("yolov8n-pose.pt", task='pose')
tensorrt_model = YOLO("yolov8n-pose.engine", task='pose')
model.to("cpu")

image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']

image_paths = []
for ext in image_extensions:
    image_paths.extend(glob(f"./coco_images/{ext}"))

memory_usage = []

mem_usage_before = get_res_memory_usage()
print(f"Before starting inference - RES Memory usage: {mem_usage_before:.2f} MB")

for i, image_path in enumerate(image_paths):
    result = tensorrt_model(image_path)
    mem_usage = get_res_memory_usage()
    memory_usage.append(mem_usage)
    print(f"Processed image {i+1}/{len(image_paths)} - RES Memory usage: {mem_usage:.2f} MB")

