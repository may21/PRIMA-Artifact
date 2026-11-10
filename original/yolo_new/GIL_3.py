import os
import shutil
from threading import Thread
from ultralytics import YOLO
import time

def thread_safe_predict(image_paths, model_name, task_type, output_file):
    local_model = YOLO(model_name, task=task_type)
    with open(output_file, 'w') as f:
        for image_path in image_paths:
            start_time = time.time()
            results = local_model.predict(image_path)
            end_time = time.time()
            inference_time = end_time - start_time
            f.write(f"Image: {image_path}, Inference Time: {inference_time:.4f} seconds, Results: {results}\n")

# 결과를 저장할 폴더 설정
results_folder = "./model_results"

# 폴더가 이미 존재하면 삭제 후 새로 생성
if os.path.exists(results_folder):
    shutil.rmtree(results_folder)
os.makedirs(results_folder)

# 이미지가 저장된 폴더 경로
#image_folder = "./coco_images" #detect.pose,segment
image_folder = "./dota_images" #obb
#image_folder = "./imagenet_images" #classify

# 이미지 파일 리스트 가져오기
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]

# 스레드 개수 입력받기
num_threads = 4 

# 이미지 할당 크기 (25개씩 할당)
images_per_thread = 25

# 이미지 경로 리스트를 재사용하도록 조정
splits = []
for i in range(num_threads):
    start_idx = (i * images_per_thread) % len(image_paths)
    end_idx = start_idx + images_per_thread
    splits.append(image_paths[start_idx:end_idx])

# 스레드 생성 및 시작
threads = []
for i in range(num_threads):
    thread_name = f"Model{i+1}Thread"
    output_file = os.path.join(results_folder, f"results_model{i+1}.txt")
    threads.append(Thread(target=thread_safe_predict, args=(splits[i], "yolov8n-obb.pt", "obb", output_file), name=thread_name))

for thread in threads:
    thread.start()

# 스레드 완료 대기
for thread in threads:
    thread.join()

