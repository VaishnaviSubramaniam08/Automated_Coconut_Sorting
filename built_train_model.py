import pandas as pd
import os
import cv2
import numpy as np

# ---- Feature Extraction ----
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = w / h if h > 0 else 0

    return [area, perimeter, circularity, aspect_ratio]

# ---- Load Labels CSV ----
labels = pd.read_csv(r"C:\Users\VAISHNAVI S\Downloads\coconut\coconut\coconut_dataset_with_features.csv")

data = []
for _, row in labels.iterrows():
    img_name = row["image_name"]     # make sure column exists
    weight = row["whole_weight"]     # make sure column exists

    img_path = os.path.join(
        r"C:\Users\VAISHNAVI S\Downloads\coconut\coconut\weightcalculation",
        img_name
    )
    features = extract_features(img_path)

    if features:
        data.append(features + [weight])

df = pd.DataFrame(data, columns=["area","perimeter","circularity","aspect_ratio","weight"])
df.to_csv(r"C:\Users\VAISHNAVI S\Downloads\coconut\coconut\coconut_dataset_with_features.csv", index=False)

print("✅ Dataset created with", len(df), "samples")
