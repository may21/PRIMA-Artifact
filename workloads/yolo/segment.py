import ctypes
import os
from ultralytics import YOLO
import time
libc = ctypes.cdll.LoadLibrary("libc.so.6")

# pthread_self()는 pthread_t (void*) 반환
libc.pthread_self.restype = ctypes.c_void_p

# pthread_setname_np(pthread_t thread, const char *name)
libc.pthread_setname_np.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

# 이름 설정 (최대 15자까지 + null terminator)
libc.pthread_setname_np(libc.pthread_self(), b"segment")
#start = time.time()
# TensorRT 모델을 불러옵니다
tensorrt_model = YOLO("yolov8n-seg.engine", task='segment')
#model = YOLO("yolov8n-seg.pt", task='segment')
#model.to("cpu")

# 이미지 경로를 불러옵니다
image_limit = os.environ.get("IMAGE_LIMIT", "110")
image_path = f"./image_lists/coco_{image_limit}.txt"
start = time.time()
# 모델을 이미지 경로에 적용하고 GPU에서 실행
result = tensorrt_model.predict(
    image_path,
    save=True,
    project="/usr/src/ultralytics/yolo_new/runs3",  # 저장할 상위 경로
    name="seg",                                       # 폴더 이름
    exist_ok=True                                     # 같은 폴더에 덮어쓰기(번호 증가 방지)
)
#result = tensorrt_model.predict(source=image_path,save=True)
print(time.time()-start)
#result = model(image_path)
