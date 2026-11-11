import ctypes
import os
from ultralytics import YOLO
import time
# 스레드 이름 설정
libc = ctypes.cdll.LoadLibrary("libc.so.6")

# pthread_self()는 pthread_t (void*) 반환
libc.pthread_self.restype = ctypes.c_void_p

#pthread_setname_np(pthread_t thread, const char *name)
libc.pthread_setname_np.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

# 이름 설정 (최대 15자까지 + null terminator)
libc.pthread_setname_np(libc.pthread_self(), b"classify")
#start = time.time()
# Load the exported TensorRT model
#model = YOLO("yolov8n-cls.pt", task='classify')
#model.export(format="engine" , imgsz=640)
tensorrt_model = YOLO("yolov8n-cls.engine", task='classify')
#model.to("cuda")

# Load the image path
image_limit = os.environ.get("IMAGE_LIMIT", "110")
image_path = f"./image_lists/imagenet_{image_limit}.txt"
start = time.time()
result = tensorrt_model.predict(
    image_path,
    save=True,
    project="/usr/src/ultralytics/yolo_new/runs3",  # 저장할 상위 경로
    name="cls",                                       # 폴더 이름
    exist_ok=True                                     # 같은 폴더에 덮어쓰기(번호 증가 방지)
)
#result = tensorrt_model.predict(image_path, save=True)
#result = model(image_path)
print(time.time()-start)
