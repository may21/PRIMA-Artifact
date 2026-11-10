import os
import shutil
from threading import Thread
from ultralytics import YOLO
import time
import subprocess
import numpy as np

def thread_safe_predict(image_paths, model_name, task_type, output_file):
    local_model = YOLO(model_name, task=task_type)
    with open(output_file, 'w') as f:
        for image_path in image_paths:
            start_time = time.time()
            results = local_model.predict(image_path)
            end_time = time.time()
            inference_time_ms = (end_time - start_time) * 1000

            for result in results:
                if hasattr(result, 'probs') and isinstance(result.probs, np.ndarray):
                    probs_array = result.probs
                    labels_and_confidences = ', '.join([f"{local_model.names[i]} {conf:.2f}" for i, conf in enumerate(probs_array) if conf > 0])
                    f.write(f"Image: {image_path}, Labels: {labels_and_confidences}, Inference Time: {inference_time_ms:.1f} ms\n")
                else:
                    f.write(f"Image: {image_path}, Inference Time: {inference_time_ms:.1f} ms, No probabilities found\n")

def save_gr3d_freq_and_res(gpu_log_file, top_log_file, output_file):
    with open(gpu_log_file, 'r') as gpu_file, open(top_log_file, 'r') as top_file, open(output_file, 'w') as outfile:
        gpu_lines = gpu_file.readlines()
        top_lines = top_file.readlines()

        outfile.write("GPU_Usage(%)\tCPU_RES(MB)\n")
        for gpu_line, top_line in zip(gpu_lines, top_lines):
            if "GR3D_FREQ" in gpu_line:
                gpu_usage = gpu_line.split("GR3D_FREQ")[1].split("%")[0].strip()
            else:
                gpu_usage = "N/A"

            if "python" in top_line:
                res_value = top_line.split()[5]
                res_value_mb = round(int(res_value) / 1024, 2)
            else:
                res_value_mb = "N/A"

            outfile.write(f"{gpu_usage}\t{res_value_mb}\n")

results_folder = "./gpu_model_results"

if os.path.exists(results_folder):
    shutil.rmtree(results_folder)
os.makedirs(results_folder)

image_folder = "./dota_images"
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]

total_images = len(image_paths)
num_threads = 4
splits = np.array_split(image_paths, num_threads)

gpu_log_file = os.path.join(results_folder, 'gpu_usage_log_raw.txt')
top_log_file = os.path.join(results_folder, 'cpu_usage_log_raw.txt')

tegrastats_process = subprocess.Popen(['tegrastats', '--interval', '1000'], stdout=open(gpu_log_file, 'w'), stderr=subprocess.PIPE)
top_process = subprocess.Popen(['top', '-b', '-d', '1'], stdout=open(top_log_file, 'w'), stderr=subprocess.PIPE)

threads = []
for i, split in enumerate(splits):
    output_file = os.path.join(results_folder, f"results_model_{i+1}.txt")
    thread = Thread(target=thread_safe_predict, args=(split, "yolov8n-obb.engine", "obb", output_file), name=f"ModelThread_{i+1}")
    threads.append(thread)

for thread in threads:
    thread.start()
    
for thread in threads:
    thread.join()

tegrastats_process.terminate()
top_process.terminate()

filtered_log_file = os.path.join(results_folder, 'gpu_cpu_usage_summary.txt')
save_gr3d_freq_and_res(gpu_log_file, top_log_file, filtered_log_file)

