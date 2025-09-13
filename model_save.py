import shutil

# Path to YOLO saved weights
src = "runs/detect/train3/weights/best.pt"

# Rename and save as a reusable detection model
dst = "coconut_detect_best.pt"
shutil.copy(src, dst)

print(f"Model saved as {dst}")
