import torch
import time
from ultralytics import YOLO
import os

# CUDA 스트림 생성
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

# 모델 로드
model_seg = YOLO("yolov8n-seg.pt", task='segment').to('cuda')
model_obb = YOLO("yolov8n-obb.pt", task='obb').to('cuda')

# 세그먼트 및 OBB 모델 함수 (CUDA 스트림 사용)
def run_models_with_stream(batch_size=20):
    start_time = time.time()

    # 세그먼트 이미지 리스트 (batch_size 장씩만 처리)
    image_path_seg = "./coco_images"
    images_seg = [os.path.join(image_path_seg, f) for f in os.listdir(image_path_seg) if f.endswith('.jpg') or f.endswith('.png')]
    batch_seg = images_seg[:batch_size]

    # 세그먼트 모델 실행
    with torch.cuda.stream(stream1):
        result_seg = [model_seg(img) for img in batch_seg]
        print("Segmentation task completed.")

    # OBB 이미지 리스트 (batch_size 장씩만 처리)
    image_path_obb = "./dota_images"
    images_obb = [os.path.join(image_path_obb, f) for f in os.listdir(image_path_obb) if f.endswith('.jpg') or f.endswith('.png')]
    batch_obb = images_obb[:batch_size]

    # OBB 모델 실행
    with torch.cuda.stream(stream2):
        result_obb = [model_obb(img) for img in batch_obb]
        print("OBB task completed.")

    torch.cuda.synchronize()  # 모든 연산이 완료될 때까지 대기
    end_time = time.time()
    print(f"Stream execution time: {end_time - start_time:.5f} seconds")

# 세그먼트 및 OBB 모델 함수 (순차 실행)
def run_models_sequential(batch_size=20):
    start_time = time.time()

    # 세그먼트 이미지 리스트 (batch_size 장씩만 처리)
    image_path_seg = "./coco_images"
    images_seg = [os.path.join(image_path_seg, f) for f in os.listdir(image_path_seg) if f.endswith('.jpg') or f.endswith('.png')]
    batch_seg = images_seg[:batch_size]

    # 세그먼트 모델 실행
    result_seg = [model_seg(img) for img in batch_seg]
    print("Segmentation task completed.")

    # OBB 이미지 리스트 (batch_size 장씩만 처리)
    image_path_obb = "./dota_images"
    images_obb = [os.path.join(image_path_obb, f) for f in os.listdir(image_path_obb) if f.endswith('.jpg') or f.endswith('.png')]
    batch_obb = images_obb[:batch_size]

    # OBB 모델 실행
    result_obb = [model_obb(img) for img in batch_obb]
    print("OBB task completed.")

    end_time = time.time()
    print(f"Sequential execution time: {end_time - start_time:.5f} seconds")

# 메인 함수
if __name__ == "__main__":
    batch_size = 20
    print("1. CUDA 스트림을 사용한 병렬 실행:")
    run_models_with_stream(batch_size=batch_size)

    #print("\n2. 순차 실행:")
    #run_models_sequential(batch_size=batch_size)

