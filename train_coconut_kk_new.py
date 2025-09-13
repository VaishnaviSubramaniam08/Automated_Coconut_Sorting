import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# -------------------- Load Dataset --------------------
df = pd.read_csv(r"C:\Users\VAISHNAVI S\Downloads\coconut\coconut\coconut_dataset_with_features.csv")  # replace with your dataset path

feature_cols = ['pixel_area', 'perimeter', 'circularity', 'aspect_ratio']
X = df[feature_cols]
y = df['whole_weight']

# Optional: Scale features (good for KNN, harmless for Random Forest)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -------------------- Train RandomForest --------------------
reg = RandomForestRegressor(n_estimators=200, random_state=42)
reg.fit(X_train, y_train)

# Test model
y_pred = reg.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model trained! MAE on test data: {mae:.2f} grams")

# Save model and scaler
joblib.dump(reg, "coconut_weight_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# -------------------- Feature Extraction --------------------
def extract_features(frame, min_area=500):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        area = cv2.contourArea(best_contour)
        perimeter = cv2.arcLength(best_contour, True)
        x, y, w, h = cv2.boundingRect(best_contour)
        aspect_ratio = float(w)/h
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        features = [area, perimeter, circularity, aspect_ratio]
        return best_contour, features
    else:
        return None, [0,0,0,0]

# -------------------- Real-time Webcam Prediction --------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Load trained model and scaler
reg = joblib.load("coconut_weight_model.pkl")
scaler = joblib.load("scaler.pkl")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    contour, features = extract_features(frame)
    
    if contour is not None:
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Predict weight
        weight = reg.predict(features_scaled)[0]
        
        # Draw contour and show prediction
        cv2.drawContours(frame, [contour], -1, (0,255,0), 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(frame, f"Weight: {int(weight)}g", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        
        print(f"Detected coconut - Predicted Weight: {int(weight)}g")

    cv2.imshow("Coconut Weight Prediction", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
