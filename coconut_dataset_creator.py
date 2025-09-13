import cv2
import os
import pandas as pd

# Step 1: Calculate pixel areas for all images
input_folder = r'C:\Users\VAISHNAVI S\Downloads\coconut\coconut\weightcalculation_gray'  # grayscale images folder
image_names = []
areas = []

print("Calculating pixel areas...")
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.jpg'):  # handle .JPG too
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Could not read {filename}")
            continue

        _, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
        else:
            area = 0

        image_names.append(filename)
        areas.append(area)
        print(f"{filename}: {area}")

# Step 2: Create DataFrame with areas
area_df = pd.DataFrame({"cocconut_id": image_names, "pixel_area": areas})

# Step 3: Load weight data (your CSV)
weight_csv_path = r'C:\Users\VAISHNAVI S\Downloads\coconut\coconut\Untitled spreadsheet - Sheet1 (2).csv'  # change to your file path
weight_df = pd.read_csv(weight_csv_path)

# Step 4: Merge area data with weight data
merged_df = pd.merge(weight_df, area_df, on='cocconut_id', how='inner')

# Step 5: Add weight categories (low/medium/high)
merged_df['category'] = pd.qcut(
    merged_df['whole_weight'], 3, labels=['low', 'medium', 'high']
)

# Step 6: Add image paths
merged_df['image_path'] = merged_df['cocconut_id'].apply(
    lambda x: os.path.join(input_folder, x)
)

# Step 7: Save final merged dataset
output_csv = r'C:\Users\VAISHNAVI S\Downloads\coconut\coconut_complete_dataset.csv'
merged_df.to_csv(output_csv, index=False)

print(f"\nFinal dataset saved to: {output_csv}")
print("Columns:", list(merged_df.columns))
print(merged_df.head())
