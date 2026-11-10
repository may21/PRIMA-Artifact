import psutil
import subprocess
import time
import os

# 결과 저장할 폴더 생성
output_folder = "cpu_logs"
os.makedirs(output_folder, exist_ok=True)
output_file_path = os.path.join(output_folder, "cpu_usage_log.txt")

# target_script.py를 별도 프로세스로 실행
process = subprocess.Popen(["/usr/bin/python3", "multiprocess.py"])
pid = process.pid
main_process = psutil.Process(pid)

try:
    with open(output_file_path, "w") as log_file:
        while process.poll() is None:  # 프로세스가 종료될 때까지 반복
            # 시스템 전체 CPU 사용량 측정
            total_cpu_usage = psutil.cpu_percent(interval=1)
            log_file.write(f"전체 CPU 사용량: {total_cpu_usage}%\n")
            print(f"전체 CPU 사용량: {total_cpu_usage}%")
            
            # target_script.py가 생성한 모든 하위 프로세스를 추적
            children = main_process.children(recursive=True)
            for child in children:
                cpu_usage = child.cpu_percent(interval=1)
                log_file.write(f"PID {child.pid}의 CPU 사용량: {cpu_usage}%\n")
                print(f"PID {child.pid}의 CPU 사용량: {cpu_usage}%")
            time.sleep(1)
except psutil.NoSuchProcess:
    print("프로세스가 종료되었습니다.")

