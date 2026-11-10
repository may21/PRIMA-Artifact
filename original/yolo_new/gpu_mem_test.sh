#!/bin/bash
  
# 필요한 디렉토리를 확인하고 없으면 생성
if [ ! -d "gpu_results" ]; then
    mkdir gpu_results
else
    rm -f gpu_results/*.txt
fi

if [ ! -d "memory_results" ]; then
    mkdir memory_results
else
    rm -f memory_results/*.txt
fi

# GPU 사용량을 모니터링하는 함수
monitor_gpu() {
    local script_name=$1
    local gpu_file="gpu_results/${script_name%.py}_gpu.txt"
    local result_file="memory_results/${script_name%.py}_memory_result.txt"

    # tegrastats를 백그라운드에서 시작하고 GR3D_FREQ 부분만 필터링하여 저장
    (tegrastats --interval 5000 | grep --line-buffered -oP 'GR3D_FREQ \K\d+%@\[\d+\]' > $gpu_file) &
    TEGRA_PID=$!

    # Python 스크립트 실행 및 결과 저장
    python3 $script_name > $result_file

    # tegrastats 모니터링 중지
    kill $TEGRA_PID
}

# 각 스크립트를 GPU 모니터링과 함께 실행
monitor_gpu "obb_memory.py"
monitor_gpu "pose_memory.py"
monitor_gpu "detect_memory.py"
monitor_gpu "classify_memory.py"
monitor_gpu "segment_memory.py"
