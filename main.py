import cv2
import numpy as np
from ultralytics import YOLO

# Load detection (object detection)
detect_model = YOLO(r"C:/Users/VAISHNAVI S/Downloads/coconut/coconut/coconut_detect_best.pt")

# Load classification (healthy / crack)
classify_model = YOLO(r"C:/Users/VAISHNAVI S/Downloads/coconut/coconut/coconut_model_crac_hea.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1️⃣ Detect coconuts
    det_results = detect_model.predict(frame, imgsz=640, conf=0.25, verbose=False)
    boxes = det_results[0].boxes.xyxy.cpu().numpy() if det_results[0].boxes is not None else []

    if len(boxes) == 0:
        cv2.putText(frame, "No coconut detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            # Clip to frame size
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

            coconut_crop = frame[y1:y2, x1:x2]

            if coconut_crop.size == 0:
                continue  # skip invalid crop

            # 2️⃣ Classify coconut
            cls_results = classify_model.predict(coconut_crop, imgsz=224, verbose=False)
            # get probabilities
            probs = cls_results[0].probs.data.cpu().numpy()
            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])

            # get class name
            if hasattr(classify_model, 'names'):
                label = classify_model.names[class_id]
            else:
                label = str(class_id)

            # Draw box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Coconut Detection + Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
