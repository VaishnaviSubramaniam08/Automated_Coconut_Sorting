import cv2
from ultralytics import YOLO

# Load your trained classification model
model = YOLO("C:/Users/VAISHNAVI S/Downloads/coconut/coconut/coconut_model.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run prediction
    results = model.predict(frame, imgsz=224, verbose=False)

    # Get top prediction
    probs = results[0].probs  # classification probabilities
    class_id = probs.top1
    conf = probs.top1conf.cpu().numpy()  # confidence
    class_name = model.names[class_id]

    # If confidence too low → assume no coconut
    if conf < 0.6:   # threshold (tweak as needed)
        text = f"No coconut"
        color = (0, 0, 255)  # red
    else:
        text = f"{class_name}: {conf:.2f}"
        color = (0, 255, 0) if class_name == "healthy" else (0, 0, 255)

    # Show prediction on frame
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Camera Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
