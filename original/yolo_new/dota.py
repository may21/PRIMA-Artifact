import os
import shutil

# 폴더 경로 설정
folder_path = "dota_images"

# 파일 이름 정렬
files = sorted(os.listdir(folder_path))

# 복사할 파일 인덱스
copy_indices = [0, 1, 2]  # 1, 2, 3번째 파일

# 복사 작업
for index in copy_indices:
    if 0 <= index < len(files):  # 인덱스가 파일 리스트 범위 내인지 확인
        original_file = os.path.join(folder_path, files[index])
        new_file_name = f"{os.path.splitext(files[index])[0]}_copy{os.path.splitext(files[index])[1]}"
        new_file = os.path.join(folder_path, new_file_name)
        shutil.copy(original_file, new_file)
        print(f"Copied: {original_file} to {new_file}")

