import psutil
import time

# PID 파일에서 PID 읽기
with open("pid.txt", "r") as f:
    pid = int(f.read().strip())

# psutil을 사용하여 해당 PID의 프로세스 추적
process = psutil.Process(pid)

print(f"Tracking memory usage for PID: {pid}")

# 실시간으로 메모리 사용량 추적
try:
    while True:
        mem_info = process.memory_info()
        mem_usage_mb = mem_info.rss / (1024 ** 2)  # MB 단위로 변환
        print(f"Memory usage: {mem_usage_mb:.2f} MB")

        # 1초 간격으로 메모리 사용량 업데이트
        time.sleep(1)

except psutil.NoSuchProcess:
    print("Process ended.")

