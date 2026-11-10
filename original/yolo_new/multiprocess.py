import os
import shutil
from multiprocessing import Process
from ultralytics import YOLO
import time
import numpy as np

#device = 'cuda' if torch.cuda.is_available() else 'cpu'

def process_safe_predict(image_paths, model_name, task_type, output_file):
    local_model = YOLO(model_name, task=task_type)#.to('cpu')
    with open(output_file, 'w') as f:
        for image_path in image_paths:
            #start_time = time.time()
            results = local_model.predict(image_path)
            #end_time = time.time()
            #inference_time = end_time - start_time
            #f.write("Image: {}, Inference Time: {:.4f} seconds, Results: {}\n".format(image_path, inference_time, results))

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

# 이미지 폴더 경로 지정
#image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]
image_paths = [os.path.abspath(os.path.join(image_folder, img)) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]

# 총 이미지 수
total_images = len(image_paths)

# 사용할 프로세스 수
num_processes = 4  # 원하는 프로세스 수 (예: 1, 2, 4, 6, 8)

# np.array_split을 사용하여 고르게 이미지 분배
splits = np.array_split(image_paths, num_processes)

# 프로세스 생성 및 시작
processes = []
start = time.time()
for i, split in enumerate(splits):
    output_file = os.path.join(results_folder, f"results_model_{i+1}.txt")
    #process = Process(target=process_safe_predict, args=(split, "yolov8n-obb.pt", "obb", output_file), name=f"ModelProcess_{i+1}")
    process = Process(target=process_safe_predict, args=(split, "yolov8n-obb.engine", "obb", output_file), name=f"ModelProcess_{i+1}")
    processes.append(process)

for process in processes:
    process.start()

for process in processes:
    process.join()
print(time.time()-start,"s")
