import torch
import time
from ultralytics import YOLO

# CUDA 스트림 생성
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

# 모델 로드를 한 번만 수행
model_seg = YOLO("yolov8n-seg.pt", task='segment').to('cuda')
model_obb = YOLO("yolov8n-obb.pt", task='obb').to('cuda')

# 세그먼트 및 OBB 모델 함수 (CUDA 스트림 사용)
def run_models_with_stream():
    start_time = time.time()
    # 세그먼트 모델 실행
    with torch.cuda.stream(stream1):
        image_path_seg = "./coco_images"
        result_seg = model_seg(image_path_seg)  # 모델 로드는 스트림 밖에서 한 번만 수행
        print("Segmentation task completed.")

    # OBB 모델 실행
    with torch.cuda.stream(stream2):
        image_path_obb = "./dota_images"
        result_obb = model_obb(image_path_obb)  # 모델 로드는 스트림 밖에서 한 번만 수행
        print("OBB task completed.")

    torch.cuda.synchronize()  # 모든 연산이 완료될 때까지 대기
    end_time = time.time()
    print(f"Stream execution time: {end_time - start_time:.5f} seconds")

# 세그먼트 및 OBB 모델 함수 (순차 실행)
def run_models_sequential():
    start_time = time.time()
    # 세그먼트 모델 실행
    image_path_seg = "./coco_images"
    result_seg = model_seg(image_path_seg)
    print("Segmentation task completed.")

    # OBB 모델 실행
    image_path_obb = "./dota_images"
    result_obb = model_obb(image_path_obb)
    print("OBB task completed.")

    end_time = time.time()
    print(f"Sequential execution time: {end_time - start_time:.5f} seconds")

# 메인 함수
if __name__ == "__main__":
    print("1. CUDA 스트림을 사용한 병렬 실행:")
    run_models_with_stream()

    #print("\n2. 순차 실행:")
    #run_models_sequential()

