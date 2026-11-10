import os
import shutil
from threading import Thread
from ultralytics import YOLO
import time
import subprocess
import numpy as np

def thread_safe_predict(image_paths, model_name, task_type, output_file, batch_size=16):
    local_model = YOLO(model_name, task=task_type)
    with open(output_file, 'w') as f:
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            if len(batch) == 0:  # 빈 배치 건너뛰기
                continue
            
            valid_batch = [img for img in batch if os.path.exists(img)]
            if len(valid_batch) == 0:  # 유효한 이미지가 없으면 건너뛰기
                print(f"Warning: No valid images in batch {batch}")
                continue
            
            try:
                start_time = time.time()
                results = local_model.predict(source=valid_batch, batch=batch_size)
                end_time = time.time()
                inference_time_ms = (end_time - start_time) * 1000

                for result, image_path in zip(results, valid_batch):
                    if hasattr(result, 'probs') and isinstance(result.probs, np.ndarray):
                        probs_array = result.probs
                        labels_and_confidences = ', '.join([f"{local_model.names[i]} {conf:.2f}" for i, conf in enumerate(probs_array) if conf > 0])
                        f.write(f"Image: {image_path}, Labels: {labels_and_confidences}, Inference Time: {inference_time_ms:.1f} ms\n")
                    else:
                        f.write(f"Image: {image_path}, Inference Time: {inference_time_ms:.1f} ms, No probabilities found\n")
            except Exception as e:
                print(f"Error processing batch {batch}: {e}")

def save_gr3d_freq(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if "GR3D_FREQ" in line:
                freq_value = line.split("GR3D_FREQ")[1].split("%")[0].strip()
                outfile.write(f"{freq_value}%\n")

results_folder = "./gpu_model_results"

if os.path.exists(results_folder):
    shutil.rmtree(results_folder)
os.makedirs(results_folder)

image_folder = "./dota_images"
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png'))]

total_images = len(image_paths)
num_threads = 1
splits = np.array_split(image_paths, num_threads)

gpu_log_file = os.path.join(results_folder, 'gpu_usage_log_raw.txt')
tegrastats_process = subprocess.Popen(['tegrastats', '--interval', '1000'], stdout=open(gpu_log_file, 'w'), stderr=subprocess.PIPE)

threads = []
for i, split in enumerate(splits):
    output_file = os.path.join(results_folder, f"results_model_{i+1}.txt")
    thread = Thread(target=thread_safe_predict, args=(split, "yolov8n-obb.engine", "obb", output_file, 16), name=f"ModelThread_{i+1}")
    threads.append(thread)

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

tegrastats_process.terminate()

filtered_gpu_log_file = os.path.join(results_folder, 'gpu_usage_gr3d_freq_only.txt')
save_gr3d_freq(gpu_log_file, filtered_gpu_log_file)

