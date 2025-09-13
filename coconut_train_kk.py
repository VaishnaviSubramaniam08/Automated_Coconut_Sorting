import cv2
import numpy as np
import joblib
import pandas as pd
import os

# ---------------- Load scaler + trained regressor ----------------
scaler = joblib.load("coconut_scaler.pkl")
regressor = joblib.load("coconut_weight_model.pkl")

# ---------------- Feature Extraction ----------------
def get_coconut_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding (inverse since coconut is darker on light background)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("❌ No contours found")
        return None

    c = max(contours, key=cv2.contourArea)

    # Feature calculations
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = w / h if h > 0 else 0

    # ✅ Make sure column names match training data
    features = pd.DataFrame(
        [[area, perimeter, circularity, aspect_ratio]],
        columns=["area", "perimeter", "circularity", "aspect_ratio"],
    )
    return features

# ---------------- Prediction ----------------
def predict_coconut_weight(image_path):
    features = get_coconut_features(image_path)
    if features is None:
        print("❌ No coconut detected")
        return None

    # Scale features
    scaled_features = scaler.transform(features)
    
    # Predict weight
    predicted_weight = regressor.predict(scaled_features)[0]
    return predicted_weight

# ---------------- Main Test ----------------
if __name__ == "__main__":
    test_image = r"C:\Users\VAISHNAVI S\Downloads\coconut\coconut\weightcalculation\wgt27.jpg"
    if os.path.exists(test_image):
        weight = predict_coconut_weight(test_image)
        if weight is not None:
            print(f"✅ Estimated Coconut Weight: {int(weight)} g")
    else:
        print(f"❌ Image not found: {test_image}")
