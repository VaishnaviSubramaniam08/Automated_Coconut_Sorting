import cv2
import numpy as np
from ultralytics import YOLO
import joblib
import pandas as pd

# ---------------- Load YOLO models ----------------
detect_model = YOLO(r"C:/Users/VAISHNAVI S/Downloads/coconut/coconut/coconut_detect_best.pt")
classify_model = YOLO(r"C:/Users/VAISHNAVI S/Downloads/coconut/coconut/coconut_model_crac_hea.pt")

# ---------------- Load weight prediction models ----------------
scaler = joblib.load("coconut_scaler.pkl")
regressor = joblib.load("coconut_weight_model.pkl")

# ---------------- Feature Extraction ----------------
def get_coconut_features(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = w / h if h > 0 else 0

    features = pd.DataFrame([[area, perimeter, circularity, aspect_ratio]],
                            columns=["area", "perimeter", "circularity", "aspect_ratio"])
    return features

# ---------------- Weight Prediction ----------------
def predict_coconut_weight(crop):
    features = get_coconut_features(crop)
    if features is None:
        return None
    scaled_features = scaler.transform(features)
    weight = regressor.predict(scaled_features)[0]
    return weight

# ---------------- Maturity Prediction ----------------
def predict_maturity_level(crop):
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    avg_hue = np.mean(hsv_crop[:, :, 0])
    avg_saturation = np.mean(hsv_crop[:, :, 1])
    avg_value = np.mean(hsv_crop[:, :, 2])

    # Thresholds (Adjust based on your dataset)
    if avg_value > 150 and avg_saturation < 50:
        maturity = "New Coconut"
    elif avg_value > 100 and avg_value <= 150 and avg_hue < 20:
        maturity = "Weak Coconut"
    else:
        maturity = "Mature Coconut"

    return maturity

# ---------------- Main Real-time Loop ----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect coconuts
    det_results = detect_model.predict(frame, imgsz=640, conf=0.25, verbose=False)
    boxes = det_results[0].boxes.xyxy.cpu().numpy() if det_results[0].boxes is not None else []

    if len(boxes) == 0:
        cv2.putText(frame, "No coconut detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            x1, y1 = max(x1,0), max(y1,0)
            x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

            coconut_crop = frame[y1:y2, x1:x2]
            if coconut_crop.size == 0:
                continue

            # Classify coconut (healthy/cracked)
            cls_results = classify_model.predict(coconut_crop, imgsz=224, verbose=False)
            probs = cls_results[0].probs.data.cpu().numpy()
            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])
            label = classify_model.names[class_id] if hasattr(classify_model, 'names') else str(class_id)

            # Predict maturity level
            maturity_label = predict_maturity_level(coconut_crop)

            # Predict weight only if healthy
            if label.lower() in ["healthy", "good", "intact"]:  # Adjust according to your labels
                weight = predict_coconut_weight(coconut_crop)
                if weight is not None:
                    weight = int(weight)
                    if weight < 300:
                        weight_category = "Low"
                    elif 300 <= weight <= 500:
                        weight_category = "Medium"
                    else:
                        weight_category = "High"
                    weight_label = f"Weight: {weight} g ({weight_category})"
                else:
                    weight_label = "Weight N/A"
                color = (0, 255, 0)  # Green for healthy
            else:
                weight_label = "Cracked - No Weight"
                color = (0, 0, 255)  # Red for cracked

            # --- Position of coconut (centre) ---
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            W = frame.shape[1]
            H = frame.shape[0]

            # Decide top/bottom/left/right based on thirds of frame
            if cy < H/3:
                vert = "Top"
            elif cy > 2*H/3:
                vert = "Bottom"
            else:
                vert = "Middle"

            if cx < W/3:
                horiz = "Left"
            elif cx > 2*W/3:
                horiz = "Right"
            else:
                horiz = "Center"

            position_label = f"{horiz}-{vert}"  # e.g. 'Left-Top'

            # Draw bounding box + centre + labels
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # blue dot at centre
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, max(y1-35, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Maturity: {maturity_label}", (x1, max(y1-10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, weight_label, (x1, max(y1+15, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Pos: {position_label} ({cx},{cy})", (x1, max(y1+60, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Console output including position
            print(f"Coconut {i+1}: Class={label}, Confidence={confidence:.2f}, "
                  f"Maturity={maturity_label}, {weight_label}, Pixel=({cx},{cy}), Region={position_label}")

    cv2.imshow("Coconut Detection + Classification + Weight + Maturity + Position", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
