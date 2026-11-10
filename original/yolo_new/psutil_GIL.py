import psutil
import subprocess
import time
import os  # os 모듈을 import

# 'GIL_4.py' 실행하여 프로세스 생성
process = subprocess.Popen(['/usr/bin/python3', 'GIL_4.py'])
pid = process.pid  # 프로세스 ID 가져오기
p = psutil.Process(pid)  # psutil 프로세스 객체 생성

# 결과 저장할 폴더 생성
output_folder = "cpu_logs"
os.makedirs(output_folder, exist_ok=True)  # 폴더가 존재하지 않으면 생성
output_file_path = os.path.join(output_folder, "cpu_usage_log.txt")

# 결과를 파일로 기록
with open(output_file_path, "w") as f:
    try:
        while process.poll() is None:  # 프로세스가 종료되지 않았을 때
            total_cpu_usage = psutil.cpu_percent(interval=1)  # 전체 CPU 사용량
            f.write(f"전체 CPU 사용량: {total_cpu_usage}%\n")

            # 프로세스 내 모든 스레드에 대해 CPU 사용량 출력
            threads = p.threads()  # 스레드 목록 가져오기
            for thread in threads:
                thread_id = thread.id
                thread_process = psutil.Process(thread.id)
                thread_cpu_usage = thread_process.cpu_percent(interval=1)  # 1초 간격으로 CPU 사용량 계산
                f.write(f"스레드 {thread_id} CPU 사용량: {thread_cpu_usage}%\n")


            f.flush()  # 파일에 즉시 기록
    except psutil.NoSuchProcess:
        f.write("프로세스가 종료되었습니다.\n")

# 프로세스 종료 후 리소스 해제
process.wait()
with open(output_file_path, "a") as f:
    f.write("CPU 모니터링 종료\n")

