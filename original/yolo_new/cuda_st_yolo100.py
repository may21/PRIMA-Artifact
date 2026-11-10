import os
import torch
import time
from ultralytics import YOLO

# CUDA 스트림 생성
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()
stream3 = torch.cuda.Stream()
stream4 = torch.cuda.Stream()
# YOLOv8 모델 로드 (GPU 사용)
model = YOLO('yolov8n-obb.pt',task='obb')

# 비동기 YOLO 추론 함수 (CUDA 스트림 사용)
def async_yolo_inference(stream, images):
    with torch.cuda.stream(stream):  # CUDA 스트림 설정
        results = [model(img) for img in images]  # 각 이미지 추론 수행
    return results

# GPU에서 순차적으로 YOLO 추론 수행
def sequential_inference(images):
    dota_images = "./dota_images"
    start = time.time()
    model(dota_images) # 순차적으로 추론 수행
    # torch.cuda.synchronize()  # 모든 연a산 완료 대기
    end = time.time()
    print(f"순차 연산 시간: {end - start:.5f}초")

# CUDA 스트림을 사용한 병렬 YOLO 추론
def parallel_inference(images):
    # 이미지 리스트를 두 개의 배치로 나눔
    batch1, batch2,batch3,batch4 = images[:25], images[25:50],images[50:75],images[75:]

    start = time.time()
    result1 = async_yolo_inference(stream1, batch1)  # 첫 번째 스트림에서 추론
    result2 = async_yolo_inference(stream2, batch2)  # 두 번째 스트림에서 추론
    result3 = async_yolo_inference(stream3, batch3)
    result2 = async_yolo_inference(stream4, batch4)
    torch.cuda.synchronize()  # 모든 연산 완료 대기
    end = time.time()
    
    print(f"병렬(스트림) 연산 시간: {end - start:.5f}초")

    return result1 + result2  # 두 배치의 결과 결합

if __name__ == "__main__":
    # dota_images 폴더 내 모든 PNG 파일 읽기
    folder_path = "dota_images/"
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]

    # 이미지가 100장인지 확인
    assert len(images) == 100, "100장의 이미지가 필요합니다."

    print("1. 순차 연산 수행")
    sequential_inference(images)

    #print("\n2. CUDA 스트림을 사용한 병렬 연산 수행")
    #results = parallel_inference(images)
    #print(f"총 {len(results)}개의 이미지에 대한 추론 완료")

