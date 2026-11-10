import os
import shutil

# coco_images 폴더 경로
folder = "coco_images"
extension = ".jpg"
target_delete_index = 80

# 파일 리스트 가져오기 및 정렬
files = sorted([f for f in os.listdir(folder) if f.endswith(extension)])

# 80번째 파일 삭제
if len(files) >= target_delete_index:
    file_to_delete = os.path.join(folder, files[target_delete_index - 1])
    print(f"Deleting file: {file_to_delete}")
    os.remove(file_to_delete)
else:
    print(f"Error: Less than {target_delete_index} files found.")
    exit(1)

# 파일 리스트 갱신
files = sorted([f for f in os.listdir(folder) if f.endswith(extension)])

# 첫 번째 파일 복사
if files:
    first_file = os.path.join(folder, files[0])
    new_file = os.path.join(folder, f"copy_of_{files[0]}")
    print(f"Copying file: {first_file} to {new_file}")
    shutil.copy(first_file, new_file)
else:
    print("Error: No files found in the folder.")
    exit(1)

# 최종 파일 개수 확인
final_count = len([f for f in os.listdir(folder) if f.endswith(extension)])
print(f"Total .jpg files now: {final_count}")

