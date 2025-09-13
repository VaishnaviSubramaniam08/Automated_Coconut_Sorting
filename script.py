import cv2
import numpy as np
import pandas as pd
import joblib

# ------------------------ Load Models ------------------------
clf = joblib.load('coconut_category_model.pkl')
reg = joblib.load('coconut_weight_model.pkl')
feature_cols = ['pixel_area', 'perimeter', 'circularity', 'aspect_ratio']

# ------------------------ Feature Extraction ------------------------
def extract_features(frame, min_area=500):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding for robust detection
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_contour = None
    best_score = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > best_score:
            best_score = circularity
            best_contour = cnt
    
    if best_contour is not None:
        # Compute features
        area = cv2.contourArea(best_contour)
        perimeter = cv2.arcLength(best_contour, True)
        x, y, w, h = cv2.boundingRect(best_contour)
        aspect_ratio = float(w)/h
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        features = [area, perimeter, circularity, aspect_ratio]
        return best_contour, features
    else:
        return None, [0,0,0,0]

# ------------------------ Real-time Webcam ------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    contour, features = extract_features(frame)
    
    if contour is not None:
        # Draw contour
        cv2.drawContours(frame, [contour], -1, (0,255,0), 2)
        
        # Predict
        X_input = pd.DataFrame([features], columns=feature_cols)
        category = clf.predict(X_input)[0]
        weight = reg.predict(X_input)[0]
        
        # Display on contour (center of bounding rectangle)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(frame, f"{category}, {int(weight)}g", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        
        # Print in terminal
        print(f"Coconut detected - Area: {int(features[0])}, Category: {category}, Weight: {int(weight)}g")

    # Show frame
    cv2.imshow("Coconut Weight Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
