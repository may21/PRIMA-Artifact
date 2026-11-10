import psutil
import os
from ultralytics import YOLO
from glob import glob
import matplotlib.pyplot as plt

# 현재 프로세스의 PID 기록
pid = os.getpid()
print(f"Process PID: {pid}")

# PID 출력 또는 기록
with open("pid.txt", "w") as f:
    f.write(str(pid))
    
# YOLO 모델 로드
model = YOLO("yolov8n-obb.pt", task='obb')
model.to("cpu")  # CPU 메모리 사용량을 측정하므로 CPU로 모델 이동

# 폴더 내의 모든 이미지 경로 불러오기
image_paths = glob("./dota_images/*.png")

# 각 이미지를 순차적으로 추론
for i, image_path in enumerate(image_paths):
    result = model(image_path)
    print(f"Processed image {i+1}/{len(image_paths)}")



