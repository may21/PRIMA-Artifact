import threading
import os
from ultralytics import YOLO
import subprocess

# Function to load model and predict
def thread_safe_inference(image_paths, result_file):
    model = YOLO("yolov8n-obb.engine", task='obb')
    results = []
    for image_path in image_paths:
        result = model.predict(source=image_path, device=0)
        results.append(result)
    with open(result_file, "a") as f:
        for result in results:
            f.write(str(result) + "\n")

# Function to run pidstat for Python3 processes
def monitor_cpu_usage(result_file):
    pidstat_command = f"pidstat -u -t -h 2 1 -C python3"
    with open(result_file, "w") as pidstat_file:
        subprocess.run(pidstat_command, shell=True, stdout=pidstat_file)

# Image directory
image_dir = "./dota_images"
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]

# Split image files among threads
num_cores = os.cpu_count()
split_image_files = [image_files[i::num_cores] for i in range(num_cores)]
result_file = "GIL_result.txt"

# Clear result file
with open(result_file, "w") as f:
    f.write("")

# Create threads equal to the number of CPU cores
threads = []
for i in range(num_cores):
    t = threading.Thread(target=thread_safe_inference, args=(split_image_files[i], result_file))
    threads.append(t)
    t.start()

# Start pidstat monitoring in a separate thread
pidstat_result_file = "GIL_pidstat_result.txt"
monitor_thread = threading.Thread(target=monitor_cpu_usage, args=(pidstat_result_file,))
monitor_thread.start()

# Join threads
for t in threads:
    t.join()
monitor_thread.join()

