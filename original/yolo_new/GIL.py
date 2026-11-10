import os
import subprocess
import threading
from ultralytics import YOLO
import time

# 결과 저장 폴더 설정
output_folder = "./GIL_result"
os.makedirs(output_folder, exist_ok=True)

# GPU 사용량과 메모리 사용량을 전역 변수로 설정
gpu_usage = "N/A"
gpu_memory = "N/A"
lock = threading.Lock()

# tegrastats를 백그라운드에서 실행하고 GPU 사용량과 메모리 사용량을 업데이트하는 함수
def update_gpu_stats():
    global gpu_usage, gpu_memory
    process = subprocess.Popen(['tegrastats'], stdout=subprocess.PIPE, text=True)
    
    for line in iter(process.stdout.readline, ''):
        with lock:
            for part in line.split():
                if 'GR3D_FREQ' in part:
                    gpu_usage = part.split('@')[1].replace('%', '').strip()
                elif 'RAM' in part:
                    mem_info = part.split('/')
                    gpu_memory = mem_info[0].replace('RAM', '').replace('MB', '').strip()
        time.sleep(1)  # 1초마다 업데이트

# 결과를 텍스트 파일에 저장하는 함수
def save_results(thread_name, image_path, result, gpu_usage, gpu_memory):
    result_file = os.path.join(output_folder, f"{thread_name}_results.txt")
    with open(result_file, "a") as file:
        file.write(f"Image: {image_path}, Result: {result}, GPU Usage: {gpu_usage}%, GPU Memory: {gpu_memory} MB\n")

# 스레드별 예측 함수
def thread_safe_predict(image_paths, model_path, task, thread_name):
    local_model = YOLO(model_path, task=task)
    for image_path in image_paths:
        with lock:
            gpu_usage_before = gpu_usage
            mem_usage_before = gpu_memory
        results = local_model.predict(image_path)  # 예측 수행
        with lock:
            gpu_usage_after = gpu_usage
            mem_usage_after = gpu_memory

        gpu_usage_result = f"{gpu_usage_before}% -> {gpu_usage_after}%"
        mem_usage_result = f"{mem_usage_before} MB -> {mem_usage_after} MB"
        save_results(thread_name, image_path, results, gpu_usage_result, mem_usage_result)

# GPU 상태 업데이트 스레드를 시작
gpu_stats_thread = threading.Thread(target=update_gpu_stats, daemon=True)
gpu_stats_thread.start()

# 이미지 폴더 경로 설정
image_folder = "./coco_images"
image_folder2 = "./dota_images"
image_folder3 = "./imagenet_images"

# 이미지 파일 리스트 생성
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]
image_paths_2 = [os.path.join(image_folder2, img) for img in os.listdir(image_folder2) if img.endswith(('.jpg', '.png'))]
image_paths_3 = [os.path.join(image_folder3, img) for img in os.listdir(image_folder3) if img.endswith(('.jpg', '.png'))]

# 이미지 경로 분할
quarter = len(image_paths) // 4
quarter2 = len(image_paths_2) // 4
quarter3 = len(image_paths_3) // 4

image_paths1 = image_paths[:quarter]
image_paths2 = image_paths[quarter:quarter*2]
image_paths3 = image_paths[quarter*2:quarter*3]
image_paths4 = image_paths_2[:quarter2]
image_paths5 = image_paths_3[:quarter3]

# 스레드 생성 및 시작
thread1 = threading.Thread(target=thread_safe_predict, args=(image_paths1, "yolov8n.engine", 'detect', "Model1Thread"))
thread2 = threading.Thread(target=thread_safe_predict, args=(image_paths2, "yolov8n-seg.engine", 'segment', "Model2Thread"))
thread3 = threading.Thread(target=thread_safe_predict, args=(image_paths3, "yolov8n-pose.engine", 'pose', "Model3Thread"))
thread4 = threading.Thread(target=thread_safe_predict, args=(image_paths4, "yolov8n-obb.engine", 'obb', "Model4Thread"))
thread5 = threading.Thread(target=thread_safe_predict, args=(image_paths5, "yolov8n-cls.engine", 'classify', "Model5Thread"))

thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread5.start()

# 스레드 완료 대기
thread1.join()
thread2.join()
thread3.join()
thread4.join()
thread5.join()

print(f"모든 결과는 '{output_folder}' 폴더에 저장되었습니다.")

