import os
import psutil
from threading import Thread
from ultralytics import YOLO


def log_usage_and_predict(image_paths, model_name, task, output_file):
    local_model = YOLO(model_name, task=task)
    process = psutil.Process()
   
    with open(output_file, "w") as f:
        for image_path in image_paths:
            results = local_model.predict(image_path)
            cpu_usage = process.cpu_percent(interval=0.1)
            memory_usage = process.memory_info().rss / (1024 * 1024)
            f.write(f"Image: {image_path}\n{results}\n")
            f.write(f"CPU Usage: {cpu_usage}%\n")
            f.write(f"Memory Usage (RES): {memory_usage} MB\n\n")


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
image_paths2 = image_paths[quarter:quarter*2]
image_paths3 = image_paths[quarter*2:quarter*3]
image_paths4 = image_paths_2[:quarter2]
image_paths5 = image_paths_3[:quarter3]


thread1 = Thread(target=log_usage_and_predict, args=(image_paths1, "yolov8n.pt", 'detect', "results_model1.txt"))
thread2 = Thread(target=log_usage_and_predict, args=(image_paths2, "yolov8n-seg.pt", 'segment', "results_model2.txt"))
thread3 = Thread(target=log_usage_and_predict, args=(image_paths3, "yolov8n-pose.pt", 'pose', "results_model3.txt"))
thread4 = Thread(target=log_usage_and_predict, args=(image_paths4, "yolov8n-obb.pt", 'obb', "results_model4.txt"))
thread5 = Thread(target=log_usage_and_predict, args=(image_paths5, "yolov8n-cls.pt", 'classify', "results_model5.txt"))


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

