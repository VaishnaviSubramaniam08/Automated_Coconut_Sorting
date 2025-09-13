import cv2
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO

# ---------------- Load models ----------------
detect_model = YOLO("coconut_detect_best.pt")  # YOLO for detection
reg = joblib.load("coconut_weight_model.pkl")  # Random Forest weight model
scaler = joblib.load("scaler.pkl")            # Feature scaler

feature_cols = ['pixel_area', 'perimeter', 'circularity', 'aspect_ratio']

# ---------------- Feature extraction ----------------
def extract_features_from_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return [0, 0, 0, 0]

    # take largest contour in crop
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h if h != 0 else 0
    circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter != 0 else 0

    return [area, perimeter, circularity, aspect_ratio]

# ---------------- Webcam prediction ----------------
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Detect coconuts
    results = detect_model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()  # x1,y1,x2,y2

    if len(detections) == 0:
        cv2.putText(frame, "No coconut", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
    else:
        for box in detections:
            x1, y1, x2, y2 = map(int, box[:4])
            crop = frame[y1:y2, x1:x2]

            # Step 2: Extract features from crop
            features = extract_features_from_crop(crop)
            features_scaled = scaler.transform([features])

            # Step 3: Predict weight
            weight = reg.predict(features_scaled)[0]

            # Draw bounding box + weight
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Weight: {int(weight)}g", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            print(f"Predicted Weight: {int(weight)}g")

    cv2.imshow("Coconut Detection + Weight", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
