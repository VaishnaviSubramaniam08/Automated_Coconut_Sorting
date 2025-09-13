import os

dataset_folder = r"C:/Users/VAISHNAVI S/Downloads/coconut/coconut/view"
yaml_file = os.path.join(dataset_folder, "view.yaml")

# Detect class folders automatically
class_folders = [f for f in os.listdir(dataset_folder)
                 if os.path.isdir(os.path.join(dataset_folder, f))]
class_folders = [f for f in class_folders if f.lower() in ["top", "bottom", "side"]]

with open(yaml_file, "w") as f:
    f.write(f"path: {dataset_folder}\n")
    f.write("names:\n")
    for i, cls in enumerate(sorted(class_folders)):
        f.write(f"  {i}: {cls}\n")

print(f"[INFO] YAML created at {yaml_file}")
print(f"[INFO] Classes detected: {class_folders}")
