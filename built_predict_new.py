from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\VAISHNAVI S\Downloads\coconut\coconut\coconut_dataset_with_features.csv")
X = df[["area","perimeter","circularity","aspect_ratio"]]
y = df["weight"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Save
joblib.dump(model, "coconut_weight_model.pkl")
joblib.dump(scaler, "coconut_scaler.pkl")

print("✅ Model & scaler saved")
