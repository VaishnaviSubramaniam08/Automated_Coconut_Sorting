import os
import random
import shutil

# Paths
dataset_path = "C:/Users/VAISHNAVI S/Downloads/coconut/coconut/coconuts_dataset/full_data"  # root folder
base_output = "C:/Users/VAISHNAVI S/Downloads/coconut/coconut/new_dataset"

# Classes (subfolders under full_data)
classes = ["crack", "healthy"]

def make_dirs():
    for split in ["train", "val"]:
        for cls in classes:
            os.makedirs(os.path.join(base_output, split, cls), exist_ok=True)

make_dirs()

def collect_images(cls):
    cls_path = os.path.join(dataset_path, cls)
    images = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.endswith((".jpg", ".png"))]
    return images

def move_files(file_list, dest_folder):
    for file_path in file_list:
        file_name = os.path.basename(file_path)
        
        # Copy image
        shutil.copy(file_path, os.path.join(dest_folder, file_name))
        
        # Copy label (same folder, same name but .txt)
        label_file = file_name.rsplit(".", 1)[0] + ".txt"
        label_path = os.path.join(os.path.dirname(file_path), label_file)
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(dest_folder, label_file))

for cls in classes:
    all_images = collect_images(cls)
    random.shuffle(all_images)
    
    split_idx = int(0.8 * len(all_images))  # 80% train, 20% val
    train_files = all_images[:split_idx]
    val_files = all_images[split_idx:]
    
    move_files(train_files, os.path.join(base_output, "train", cls))
    move_files(val_files, os.path.join(base_output, "val", cls))
    
    print(f"{cls} Train: {len(train_files)} | Val: {len(val_files)}")

print("Dataset split completed!")
