import os
import csv

# 텍스트 파일에서 데이터를 읽어 CSV 파일로 변환하는 함수
def convert_txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r', encoding='utf-8') as infile, open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        # CSV 헤더 작성
        writer.writerow(["Image", "Inference Time (seconds)", "Results"])
        
        # 각 줄의 데이터를 CSV로 변환하여 작성
        for line in infile:
            # "Image: path, Inference Time: 0.1234 seconds, Results: ..."
            line = line.strip()
            if line:
                parts = line.split(", ")
                # 예외 처리: parts 리스트 길이가 충분한지 확인
                if len(parts) >= 3:
                    image = parts[0].split(": ")[1]
                    inference_time = parts[1].split(": ")[1].split(" ")[0]
                    results = parts[2].split(": ", 1)[1]
                    writer.writerow([image, inference_time, results])
                else:
                    # 로그 출력 또는 예외 처리
                    print(f"Warning: Unexpected line format in file {txt_file}: {line}")

# 결과가 저장된 폴더 경로
results_folder = "./mp_results"

# 폴더 내의 모든 텍스트 파일을 찾아 CSV 파일로 변환
for filename in os.listdir(results_folder):
    if filename.endswith(".txt"):
        txt_file_path = os.path.join(results_folder, filename)
        csv_file_path = os.path.join(results_folder, filename.replace(".txt", ".csv"))
        convert_txt_to_csv(txt_file_path, csv_file_path)

print("모든 텍스트 파일이 CSV 파일로 변환되었습니다.")

