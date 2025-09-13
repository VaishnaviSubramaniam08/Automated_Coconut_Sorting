import os
import glob

# Folder path
folder = "C:/Users/VAISHNAVI S/Downloads/coconut/coconut/new_dataset/val/healthy"

# Remove all .txt files
for file in glob.glob(os.path.join(folder, "*.txt")):
    os.remove(file)
    print(f"Removed: {file}")
