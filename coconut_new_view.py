from ultralytics import YOLO

# 1. Load your trained classification model
model = YOLO(r"C:\Users\VAISHNAVI S\Downloads\coconut\coconut\runs\classify\train15\weights\best.pt")

# 2. Path to the test image
img_path = r"C:\Users\VAISHNAVI S\Downloads\coconut\coconut\view\train\side\IMG-20250912-WA0228.jpg"

# 3. Run prediction
results = model.predict(img_path)

# 4. Process results
for r in results:
    # Show result image (with predicted label drawn on it)
    r.show()

    # Save result image
    r.save(filename="output.jpg")

    # Get class predictions
    top_class_index = r.probs.top1                # index of top predicted class
    confidence = float(r.probs.top1conf)          # confidence of top class
    class_name = r.names[top_class_index]         # class name (bottom/top/side)

    print(f"Predicted Class: {class_name}")
    print(f"Confidence: {confidence:.2f}")
