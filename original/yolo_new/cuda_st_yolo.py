import torch
import time
from ultralytics import YOLO

# CUDA stream
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

# YOLOv8
model = YOLO('yolov8n-obb.pt').to('cuda')


def async_yolo_inference(stream, image):
    with torch.cuda.stream(stream): 
        result = model(image) 
    return result

def sequential_inference(images):
    start = time.time()
    for img in images:
        model(img) 
    torch.cuda.synchronize()  
    end = time.time()
    print(f"{end - start:.5f}초")

# CUDA 스트림을 사용한 병렬 YOLO 추론
def parallel_inference(images):
    start = time.time()
    result1 = async_yolo_inference(stream1, images[0]) 
    result2 = async_yolo_inference(stream2, images[1]) 
    torch.cuda.synchronize()  
    end = time.time()
    print(f"{end - start:.5f}초")

if __name__ == "__main__":
    images = ['dota_images/P0045.png', 'dota_images/P0265.png']

    print("1. Sequential")
    sequential_inference(images)

    print("\n2. CUDA stream")
    parallel_inference(images)
