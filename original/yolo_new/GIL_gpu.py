import os
import psutil
import subprocess
import time
from threading import Thread
from ultralytics import YOLO

def get_gpu_usage():
    try:
        result = subprocess.run(
            ["tegrastats", "--interval", "100", "--logfile", "/tmp/tegrastats.log"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
        )
        with open("/tmp/tegrastats.log", "r") as file:
            lines = file.readlines()
            for line in lines:
                if "GR3D_FREQ" in line:
                    gpu_usage = line.split("GR3D_FREQ")[1].split("%")[0].strip()
                    return f"{gpu_usage}%"
    except subprocess.TimeoutExpired:
        return "Timeout"
    return "N/A"

def log_usage_and_predict(image_paths, model_name, task, output_file):
    local_model = YOLO(model_name, task=task)
    process = psutil.Process()
    
    with open(output_file, "w") as f:
        for image_path in image_paths:
            results = local_model.predict(image_path)
            memory_usage = process.memory_info().rss / (1024 * 1024)
            gpu_usage = get_gpu_usage()
            f.write(f"Image: {image_path}\n{results}\n")
            f.write(f"GPU Usage: {gpu_usage}\n")
            f.write(f"Memory Usage (RES): {memory_usage} MB\n\n")
            time.sleep(0.1)

image_folder = "./coco_images"
image_folder2 = "./dota_images"
image_folder3 = "./imagenet_images"

image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]
image_paths_2 = [os.path.join(image_folder2, img) for img in os.listdir(image_folder2) if img.endswith(('.jpg', '.png'))]
image_paths_3 = [os.path.join(image_folder3, img) for img in os.listdir(image_folder3) if img.endswith(('.jpg', '.png'))]

quarter = len(image_paths) // 4
quarter2 = len(image_paths_2) // 4
quarter3 = len(image_paths_3) // 4

image_paths1 = image_paths[:quarter]
image_paths2 = image_paths[:quarter]
image_paths3 = image_paths[:quarter]
image_paths4 = image_paths_2[:quarter2]
image_paths5 = image_paths_3[:quarter3]

thread1 = Thread(target=log_usage_and_predict, args=(image_paths1, "yolov8n.engine", 'detect', "detect_results_model1.txt"))
thread2 = Thread(target=log_usage_and_predict, args=(image_paths2, "yolov8n-seg.engine", 'segment', "seg_results_model2.txt"))
thread3 = Thread(target=log_usage_and_predict, args=(image_paths3, "yolov8n-pose.engine", 'pose', "pose_results_model3.txt"))
thread4 = Thread(target=log_usage_and_predict, args=(image_paths4, "yolov8n-obb.engine", 'obb', "obb_results_model4.txt"))
thread5 = Thread(target=log_usage_and_predict, args=(image_paths5, "yolov8n-cls.engine", 'classify', "cls_results_model5.txt"))

thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread5.start()

thread1.join()
thread2.join()
thread3.join()
thread4.join()
thread5.join()

