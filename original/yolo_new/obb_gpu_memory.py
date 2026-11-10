import os
import subprocess
from ultralytics import YOLO
from glob import glob
import time
import csv

def get_python_memory_usage():
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

def start_jtop_logging(log_path="/var/log/jtop/"):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, "jtop_log.csv")
    subprocess.Popen(['sudo', 'jtop', '--log', log_file])
    return log_file

def get_latest_gpu_utilization(log_file):
    with open(log_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        if rows:
            latest_row = rows[-1]
            gpu_utilization = int(latest_row.get('GR3D_FREQ', 0))
            return gpu_utilization
    return 0

used_mem_before_model_load = get_python_memory_usage()
print(f"Before loading model - Python Process Memory Usage: {used_mem_before_model_load:.2f} MB")

model = YOLO("yolov8n-obb.pt", task='obb')
tensorrt_model = YOLO("yolov8n-obb.engine", task='obb')
model.to("cpu")

image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob(f"./dota_images/{ext}"))

used_mem_before_inference = get_python_memory_usage()
print(f"Before starting inference - Python Process Memory Usage: {used_mem_before_inference:.2f} MB")

jtop_log_file = start_jtop_logging()
memory_usage = []

try:
    for i, image_path in enumerate(image_paths):
        result = tensorrt_model(image_path)
        mem_usage = get_python_memory_usage()
        gpu_utilization = get_latest_gpu_utilization(jtop_log_file)
        
        memory_usage.append((mem_usage, gpu_utilization))
        print(f"Processed image {i+1}/{len(image_paths)} - Python Process Memory Usage: {mem_usage:.2f} MB, GPU Utilization: {gpu_utilization}%")
finally:
    subprocess.run(['sudo', 'pkill', '-f', 'jtop'])

