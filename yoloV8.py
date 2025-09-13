from ultralytics import YOLO

# Load YOLOv8 model (classification)
model = YOLO("yolov8n-cls.pt")   # small model, can use yolov8s-cls.pt for more accuracy

# Train
model.train(
    data="C:/Users/VAISHNAVI S/Downloads/coconut/coconut/new_dataset",  # path to dataset
    epochs=20,
    imgsz=224,
    batch=16
)
metrics = model.val()
print(metrics)
results = model.predict("C:/Users/VAISHNAVI S/Downloads/coconut/coconut/new_dataset/val/crack/cracked1.png")
print(results[0].probs)   # probabilities for each class
