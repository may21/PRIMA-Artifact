import os
import shutil

# Folder path
folder_path = "./expanded_dota_images"

# Check if folder exists
if not os.path.exists(folder_path):
    print(f"Folder '{folder_path}' does not exist.")
else:
    # List all files in the folder, sorted to ensure consistency
    files = sorted(os.listdir(folder_path))

    # Ensure there are at least 47 images
    if len(files) < 47:
        print("Not enough images in the folder to perform the operation.")
    else:
        # Delete the 47th image (index 46)
        image_to_delete = os.path.join(folder_path, files[46])
        os.remove(image_to_delete)
        print(f"Deleted: {image_to_delete}")

        # Copy the first image
        first_image = os.path.join(folder_path, files[0])
        new_image_copy = os.path.join(folder_path, f"copy_of_{files[0]}")
        shutil.copy2(first_image, new_image_copy)
        print(f"Copied: {first_image} to {new_image_copy}")

    # Verify the final image count
    total_images = len(os.listdir(folder_path))
    print(f"Total images in folder after operation: {total_images}")

