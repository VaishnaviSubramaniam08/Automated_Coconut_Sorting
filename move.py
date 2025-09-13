import os
import shutil

# Paths
all_labels = "C:/Users/VAISHNAVI S/Downloads/coconut/coconut/all_labels"
images_train = "C:/Users/VAISHNAVI S/Downloads/coconut/coconut/coconut_detect/images/train"
images_val = "C:/Users/VAISHNAVI S/Downloads/coconut/coconut/coconut_detect/images/val"
labels_train = "C:/Users/VAISHNAVI S/Downloads/coconut/coconut/coconut_detect/labels/train"
labels_val = "C:/Users/VAISHNAVI S/Downloads/coconut/coconut/coconut_detect/labels/val"

# Make sure label folders exist
os.makedirs(labels_train, exist_ok=True)
os.makedirs(labels_val, exist_ok=True)

# Function to copy labels based on images
def copy_labels(image_folder, label_folder):
    for img_file in os.listdir(image_folder):
        # Get base name without extension
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + ".txt"
        label_path = os.path.join(all_labels, label_file)

        # If label exists, copy it
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(label_folder, label_file))
            print(f"Copied: {label_file} → {label_folder}")
        else:
            print(f"Label not found for {img_file}")

# Copy for train and val
copy_labels(images_train, labels_train)
copy_labels(images_val, labels_val)
