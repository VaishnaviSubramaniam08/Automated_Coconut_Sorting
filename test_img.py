from ultralytics import YOLO

# Load YOUR trained model, not the default one
model = YOLO("runs/classify/train5/weights/best.pt")

# Path to test image
img_path = "C:/Users/VAISHNAVI S/Downloads/coconut/coconut/new_dataset/healthy/healthy53.jpg"  # change path if needed

# Run prediction
results = model(img_path)

# Show only "Cracked" or "Healthy"
for r in results:
    class_id = r.probs.top1
    class_name = model.names[class_id]
    print(f"Prediction: {class_name}")
