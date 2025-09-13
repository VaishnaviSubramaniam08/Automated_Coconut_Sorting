import os
import shutil
import random

# ---------------- CONFIG ----------------
dataset_dir = r"C:/Users/VAISHNAVI S/Downloads/coconut/coconut/view"  # your dataset path
classes = ["bottom", "side", "top"]   # class folders inside 'view'
splits = {"train": 0.7, "val": 0.2, "test": 0.1}  # percentages
# ----------------------------------------

# Create output directories
for split in splits.keys():
    for cls in classes:
        out_dir = os.path.join(dataset_dir, split, cls)
        os.makedirs(out_dir, exist_ok=True)

# Split data for each class
for cls in classes:
    cls_dir = os.path.join(dataset_dir, cls)   # original folder (view/bottom, view/side, etc.)
    images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)

    total = len(images)
    train_end = int(total * splits["train"])
    val_end = train_end + int(total * splits["val"])

    split_data = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    # Move files
    for split, files in split_data.items():
        for f in files:
            src = os.path.join(cls_dir, f)
            dst = os.path.join(dataset_dir, split, cls, f)
            shutil.copy2(src, dst)  # copy instead of move, so original remains safe

print("✅ Dataset split complete!")
