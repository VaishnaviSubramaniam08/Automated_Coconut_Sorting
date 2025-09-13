import os
import random
import shutil

# Base dataset path
dataset_path = r"C:\Users\VAISHNAVI S\OneDrive\Documents\coconut\coconuts_dataset"

# Paths to images and labels inside the dataset
train_img_dir = os.path.join(dataset_path, "train")  # images inside 'train' folder
train_lbl_dir = os.path.join(dataset_path, "train")  # labels in same folder, adjust if different

# Step 1: Rename label files ending with ".xml.txt" to ".txt"
for filename in os.listdir(train_lbl_dir):
    if filename.endswith(".xml.txt"):
        old_path = os.path.join(train_lbl_dir, filename)
        new_filename = filename.replace(".xml.txt", ".txt")
        new_path = os.path.join(train_lbl_dir, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")
print("All label files renamed successfully!")

# Step 2: Create output directories for splitting
base_output = r"C:\Users\VAISHNAVI S\OneDrive\Documents\coconut\new_dataset"
output_train_img = os.path.join(base_output, "train/images")
output_train_lbl = os.path.join(base_output, "train/labels")
output_val_img = os.path.join(base_output, "val/images")
output_val_lbl = os.path.join(base_output, "val/labels")

os.makedirs(output_train_img, exist_ok=True)
os.makedirs(output_train_lbl, exist_ok=True)
os.makedirs(output_val_img, exist_ok=True)
os.makedirs(output_val_lbl, exist_ok=True)

# Step 3: Collect all images from the train folder
all_images = [f for f in os.listdir(train_img_dir) if f.endswith((".jpg", ".png"))]
random.shuffle(all_images)

# Step 4: Split 80% train, 20% val
split_idx = int(0.8 * len(all_images))
train_files = all_images[:split_idx]
val_files = all_images[split_idx:]

# Step 5: Function to copy images and their corresponding label files
def copy_files(file_list, img_dest, lbl_dest):
    for file in file_list:
        # Copy image
        shutil.copy(os.path.join(train_img_dir, file), os.path.join(img_dest, file))
        # Copy label file with same name but .txt extension
        label_file = file.rsplit(".", 1)[0] + ".txt"
        label_path = os.path.join(train_lbl_dir, label_file)
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(lbl_dest, label_file))

# Step 6: Copy train and validation files
copy_files(train_files, output_train_img, output_train_lbl)
copy_files(val_files, output_val_img, output_val_lbl)

# Step 7: Summary print
print(f"✅ Train: {len(train_files)} images, Val: {len(val_files)} images")
