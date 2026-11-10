import os
import shutil

original_image_path = "./coco_images"  # 100장의 이미지가 있는 폴더
expanded_image_path = "./expanded_coco_images"  # 복사된 1000장의 이미지를 저장할 폴더

# 이미지 폴더 생성
if not os.path.exists(expanded_image_path):
    os.makedirs(expanded_image_path)

# 100장의 이미지를 10번 복사하여 1000장의 이미지로 확장
image_files = os.listdir(original_image_path)
for i in range(100):  # 10번 복사
    for img_file in image_files:
        src = os.path.join(original_image_path, img_file)
        dst = os.path.join(expanded_image_path, f"{i}_{img_file}")
        shutil.copy(src, dst)
