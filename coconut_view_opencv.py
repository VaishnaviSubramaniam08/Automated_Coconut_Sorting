import cv2
from ultralytics import YOLO

# 1. Load your trained classification model
model = YOLO(r"C:\Users\VAISHNAVI S\Downloads\coconut\coconut\runs\classify\train15\weights\best.pt")

# 2. Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # 3. Run prediction on the frame
    results = model.predict(frame)

    for r in results:
        top_class_index = r.probs.top1
        confidence = float(r.probs.top1conf)
        class_name = r.names[top_class_index]

        # 4. Draw prediction on frame
        text = f"{class_name} ({confidence:.2f})"
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 2, cv2.LINE_AA)

    # 5. Show frame in OpenCV window
    cv2.imshow("Coconut View Classification", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
