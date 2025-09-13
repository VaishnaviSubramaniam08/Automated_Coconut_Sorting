import os

# folder containing your files
folder = r"C:/Users/VAISHNAVI S/Downloads/coconut/coconut/coconuts_dataset/train/healthy"

for filename in os.listdir(folder):
    if filename.endswith(".xml.txt"):
        old_path = os.path.join(folder, filename)
        new_filename = filename.replace(".xml.txt", ".txt")
        new_path = os.path.join(folder, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")

print("All files renamed successfully!")


