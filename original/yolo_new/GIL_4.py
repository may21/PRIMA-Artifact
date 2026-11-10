import os
import shutil
from threading import Thread
from ultralytics import YOLO
import time
import numpy as np

#device = 'cuda' if torch.cuda.is_available() else 'cpu'

def thread_safe_predict(image_paths, model_name, task_type, output_file):
    local_model = YOLO(model_name, task=task_type).to('cuda')
    with open(output_file, 'w') as f:
        for image_path in image_paths:
            #start_time = time.time()
            results = local_model.predict(image_path)
            #end_time = time.time()
            #inference_time = end_time - start_time
            #f.write(f"Image: {image_path}, Inference Time: {inference_time:.4f} seconds, Results: {results}\n")

# 결과를 저장할 폴더 설정
results_folder = "./model_results"

# 폴더가 이미 존재하면 삭제 후 새로 생성
if os.path.exists(results_folder):
    shutil.rmtree(results_folder)
os.makedirs(results_folder)

# 이미지가 저장된 폴더 경로
#image_folder = "./coco_images" #detect.pose,segment
image_folder = "./expanded_dota_images" #obb
#image_folder = "./imagenet_images" #classify

# 이미지 파일 리스트 가져오기
#image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]
image_paths = [os.path.abspath(os.path.join(image_folder, img)) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]

# 총 이미지 수
total_images = len(image_paths)

# 사용할 스레드 수
num_threads = 1  # 원하는 스레드 수 (예: 1, 2, 4, 6, 8)

# np.array_split을 사용하여 고르게 이미지 분배
splits = np.array_split(image_paths, num_threads)

# 스레드 생성및 시작
threads = []
start= time.time()
for i, split in enumerate(splits):
    output_file = os.path.join(results_folder, f"results_model_{i+1}.txt")
    thread = Thread(target=thread_safe_predict, args=(split, "yolov8n-obb.pt", "obb", output_file), name=f"ModelThread_{i+1}")
    #thread = Thread(target=thread_safe_predict, args=(split, "yolov8n-obb.engine", "obb", output_file), name=f"ModelThread_{i+1}")
    threads.append(thread)

for thread in threads:
    thread.start()

# 스레드 완료 대기
for thread in threads:
    thread.join()

print(time.time()-start,"s")

