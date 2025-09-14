# app.py
import streamlit as st
import numpy as np
from ultralytics import YOLO
import joblib
import pandas as pd
from PIL import Image
import plotly.express as px
import time
import pyttsx3
import tempfile

# ---------------- Load Models ----------------
detect_model = YOLO("C:/Users/VAISHNAVI S/Downloads/coconut/coconut/coconut_detect_best.pt")
classify_model = YOLO("C:/Users/VAISHNAVI S/Downloads/coconut/coconut/coconut_model_crac_hea.pt")
view_model = YOLO(r"C:/Users/VAISHNAVI S/Downloads/coconut/coconut/runs/classify/train15/weights/best.pt")
scaler = joblib.load("coconut_scaler.pkl")
regressor = joblib.load("coconut_weight_model.pkl")

# ---------------- Feature Functions ----------------
def get_coconut_features(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    circularity = (4 * np.pi * area) / (perimeter*perimeter) if perimeter>0 else 0
    x,y,w,h = cv2.boundingRect(c)
    aspect_ratio = w/h if h>0 else 0
    return pd.DataFrame([[area, perimeter, circularity, aspect_ratio]],
                        columns=["area","perimeter","circularity","aspect_ratio"])

def predict_coconut_weight(crop):
    features = get_coconut_features(crop)
    if features is None: return None
    scaled_features = scaler.transform(features)
    return int(regressor.predict(scaled_features)[0])

def predict_maturity_level(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    avg_hue = np.mean(hsv[:,:,0])
    avg_sat = np.mean(hsv[:,:,1])
    avg_val = np.mean(hsv[:,:,2])
    if avg_val>150 and avg_sat<50: return "New Coconut"
    elif avg_val>100 and avg_val<=150 and avg_hue<20: return "Weak Coconut"
    else: return "Mature Coconut"

def predict_coconut_view(crop):
    results = view_model.predict(crop, imgsz=224, verbose=False)
    top_class_index = results[0].probs.top1
    confidence = float(results[0].probs.top1conf)
    class_name = results[0].names[top_class_index]
    return class_name, confidence

# ---------------- Voice Function ----------------
def speak_csv_data(df):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    
    text_to_speak = "Coconut Data Summary:\n"
    for i, row in df.iterrows():
        text_to_speak += f"Coconut {row['Coconut #']}: Class {row['Class']}, " \
                         f"Confidence {row['Confidence']}, " \
                         f"Maturity {row['Maturity']}, " \
                         f"Weight {row['Weight']}, " \
                         f"View {row['View']}.\n"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        temp_file = f.name
    engine.save_to_file(text_to_speak, temp_file)
    engine.runAndWait()
    return temp_file

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Coconut Dashboard", layout="wide", page_icon="🥥")
st.markdown("""
    <style>
    .stApp .block-container {
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
""", unsafe_allow_html=True)
st.title("🥥 Coconut Detection & Dashboard")

# --- Session state ---
if 'all_coconuts' not in st.session_state:
    st.session_state.all_coconuts = []

# --- Main Layout: Add Spacer Column for more space ---
col1, spacer, col2 = st.columns([1.2, 0.2, 1])

# ====================================================
# LEFT: Upload / Camera Input + Detection Preview
# ====================================================
with col1:
    st.header("📸 Input & Detection")

    uploaded_file = st.file_uploader("📤 Upload Coconut Image", type=["jpg","png","jpeg"])
    camera_file = st.camera_input("📷 Capture Photo")
    use_webcam = st.checkbox("🎥 Live Webcam Feed")

    if st.button("🗑 Clear All Data"):
        st.session_state.all_coconuts = []
        st.success("All data cleared!")

    st.markdown("---")
    st.subheader("🔍 Detection Preview")

    frame = None
    if uploaded_file:
        frame = np.array(Image.open(uploaded_file))[:,:,::-1]
    elif camera_file:
        frame = np.array(Image.open(camera_file))[:,:,::-1]

    if use_webcam:
        st.info("Streaming from webcam... Press stop to end.")
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret: break
            annotated = frame.copy()
            det_results = detect_model.predict(annotated, imgsz=640, conf=0.25, verbose=False)
            boxes = det_results[0].boxes.xyxy.cpu().numpy() if det_results[0].boxes is not None else []
            for box in boxes:
                x1,y1,x2,y2 = map(int, box[:4])
                crop = annotated[y1:y2, x1:x2]
                if crop.size == 0: continue
                cls_results = classify_model.predict(crop, imgsz=224, verbose=False)
                probs = cls_results[0].probs.data.cpu().numpy()
                class_id = int(np.argmax(probs))
                label = classify_model.names[class_id]
                maturity = predict_maturity_level(crop)
                if label.lower() in ["healthy","good","intact"]:
                    weight = predict_coconut_weight(crop)
                    category = "Low" if weight<300 else "Medium" if weight<=500 else "High"
                    weight_label = f"{weight} g ({category})"
                    view, conf = predict_coconut_view(crop)
                else:
                    weight_label, category, view = "Cracked - No Weight","N/A","N/A"
                color = (0,255,0) if "healthy" in label.lower() else (0,0,255)
                cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
                cv2.putText(annotated,f"{label}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
            stframe.image(annotated[:,:,::-1], channels="RGB")
            time.sleep(0.05)
        cap.release()

    elif frame is not None:
        annotated = frame.copy()
        det_results = detect_model.predict(annotated, imgsz=640, conf=0.25, verbose=False)
        boxes = det_results[0].boxes.xyxy.cpu().numpy() if det_results[0].boxes is not None else []
        coconut_data = []
        for i, box in enumerate(boxes):
            x1,y1,x2,y2 = map(int, box[:4])
            crop = annotated[y1:y2, x1:x2]
            if crop.size == 0: continue
            cls_results = classify_model.predict(crop, imgsz=224, verbose=False)
            probs = cls_results[0].probs.data.cpu().numpy()
            class_id = int(np.argmax(probs))
            label = classify_model.names[class_id]
            maturity = predict_maturity_level(crop)
            if label.lower() in ["healthy","good","intact"]:
                weight = predict_coconut_weight(crop)
                category = "Low" if weight<300 else "Medium" if weight<=500 else "High"
                weight_label = f"{weight} g ({category})"
                view, conf = predict_coconut_view(crop)
            else:
                weight_label, category, view = "Cracked - No Weight","N/A","N/A"
            color = (0,255,0) if "healthy" in label.lower() else (0,0,255)
            cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
            cv2.putText(annotated,f"{label}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
            coconut_data.append({
                "Coconut #": len(st.session_state.all_coconuts)+i+1,
                "Class": label,
                "Confidence": f"{probs[class_id]:.2f}",
                "Maturity": maturity,
                "Weight": weight_label,
                "Weight Category": category,
                "View": view
            })
        st.session_state.all_coconuts.extend(coconut_data)
        st.image(annotated[:,:,::-1], caption="Detection Preview", use_container_width=True)
    else:
        st.info("Upload or capture an image, or enable webcam.")

# ====================================================
# RIGHT: Dashboard Insights
# ====================================================
with col2:
    st.header("📊 Dashboard Insights")

    total = len(st.session_state.all_coconuts)
    st.metric("🥥 Total Coconuts Detected", total)

    if total > 0:
        df = pd.DataFrame(st.session_state.all_coconuts)
        st.dataframe(df, use_container_width=True)

        # CSV download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", csv, "coconut_data.csv", "text/csv")

        # Button to speak CSV
        if st.button("🔊 Speak CSV Data"):
            audio_file = speak_csv_data(df)
            audio_bytes = open(audio_file, 'rb').read()
            st.audio(audio_bytes, format='audio/mp3')

        st.markdown("---")
        st.subheader("📈 Charts Overview")

        # --- Chart Layout: 2 per row ---
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)

        # 1️⃣ Weight Distribution - BAR
        weight_counts = df['Weight Category'].value_counts().reindex(['Low','Medium','High'], fill_value=0)
        fig_weight = px.bar(weight_counts, x=weight_counts.index, y=weight_counts.values,
                            title="Coconut Weight Distribution",
                            color=weight_counts.index,
                            color_discrete_map={'Low':'orange','Medium':'blue','High':'green'})
        row1_col1.plotly_chart(fig_weight, use_container_width=True)

        # 2️⃣ Healthy vs Cracked - PIE
        health_counts = df['Class'].apply(lambda x: 'Healthy' if x.lower() in ['healthy','good','intact'] else 'Cracked').value_counts()
        fig_health = px.pie(values=health_counts.values, names=health_counts.index,
                            title="Coconut Condition",
                            color=health_counts.index, color_discrete_map={'Healthy':'green','Cracked':'red'})
        row1_col2.plotly_chart(fig_health, use_container_width=True)

        # 3️⃣ View Distribution - PIE
        view_counts = df['View'].value_counts()
        fig_view = px.pie(values=view_counts.values, names=view_counts.index,
                          title="Coconut View Distribution")
        row2_col1.plotly_chart(fig_view, use_container_width=True)

        # 4️⃣ Maturity Levels - BAR
        maturity_counts = df['Maturity'].value_counts()
        fig_maturity = px.bar(maturity_counts, x=maturity_counts.index, y=maturity_counts.values,
                              title="Maturity Level Distribution", color=maturity_counts.index)
        row2_col2.plotly_chart(fig_maturity, use_container_width=True)

    else:
        st.info("No detection results yet.")
