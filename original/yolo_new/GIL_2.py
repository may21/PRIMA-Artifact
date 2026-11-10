import os
import shutil
from threading import Thread
from ultralytics import YOLO
import time

# 전역 리스트로 스레드 정보를 저장
threads_info = []

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
image_folder = "./coco_images"
image_folder2 = "./dota_images"
image_folder3 = "./imagenet_images"

# 이미지 파일 리스트 가져오기
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]
image_paths_2 = [os.path.join(image_folder2, img) for img in os.listdir(image_folder2) if img.endswith(('.jpg', '.png'))]
image_paths_3 = [os.path.join(image_folder3, img) for img in os.listdir(image_folder3) if img.endswith(('.jpg', '.png'))]

quarter = len(image_paths) // 4
quarter2 = len(image_paths_2) // 4
half = len(image_paths_3) // 2

image_paths1 = image_paths[:quarter]
image_paths2 = image_paths[quarter:quarter*3]
image_paths3 = image_paths[quarter*2:quarter*3]
image_paths4 = image_paths_2[:quarter2]
image_paths5 = image_paths_3[:half]
image_paths6 = image_paths_3[half:]
image_paths7 = image_paths_2[quarter2:quarter2*2]
image_paths8 = image_paths_2[quarter2*2:quarter2*3]
image_paths9 = image_paths_2[quarter2*3:]

# 스레드 생성 및 시작
threads = []

threads.append(Thread(target=thread_safe_predict, args=(image_paths1, "yolov8n.pt", "detect", os.path.join(results_folder, "results_model1.txt")), name="Model1Thread"))
threads.append(Thread(target=thread_safe_predict, args=(image_paths2, "yolov8n-seg.pt", "segment", os.path.join(results_folder, "results_model2.txt")), name="Model2Thread"))
threads.append(Thread(target=thread_safe_predict, args=(image_paths3, "yolov8n-pose.pt", "pose", os.path.join(results_folder, "results_model3.txt")), name="Model3Thread"))
threads.append(Thread(target=thread_safe_predict, args=(image_paths4, "yolov8n-obb.pt", "obb", os.path.join(results_folder, "results_model4.txt")), name="Model4Thread"))
threads.append(Thread(target=thread_safe_predict, args=(image_paths5, "yolov8n-cls.pt", "classify", os.path.join(results_folder, "results_model5.txt")), name="Model5Thread"))
threads.append(Thread(target=thread_safe_predict, args=(image_paths6, "yolov8n-cls.pt", "classify", os.path.join(results_folder, "results_model6.txt")), name="Model6Thread"))
threads.append(Thread(target=thread_safe_predict, args=(image_paths7, "yolov8n-obb.pt", "obb", os.path.join(results_folder, "results_model7.txt")), name="Model7Thread"))
threads.append(Thread(target=thread_safe_predict, args=(image_paths8, "yolov8n-obb.pt", "obb", os.path.join(results_folder, "results_model8.txt")), name="Model8Thread"))
#threads.append(Thread(target=thread_safe_predict, args=(image_paths9, "yolov8n-obb.pt", "obb", os.path.join(results_folder, "results_model9.txt")), name="Model9Thread"))

for thread in threads:
    thread.start()

# 스레드 완료 대기
for thread in threads:
    thread.join()

