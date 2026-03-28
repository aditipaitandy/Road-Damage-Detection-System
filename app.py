import streamlit as st
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import os
from datetime import datetime
from PIL import Image
from tensorflow.keras.models import load_model

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Road Damage Detection System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# File paths
# ----------------------------
MODEL_PATH = "road_damage_model.h5"
LOG_FILE = "prediction_logs.csv"
UPLOAD_DIR = "saved_uploads"
LOGO_PATH = "logo.png"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

model = load_trained_model()

# ----------------------------
# Session state
# ----------------------------
if "page_mode" not in st.session_state:
    st.session_state.page_mode = "home"

# ----------------------------
# Helper functions
# ----------------------------
def preprocess_image(image: Image.Image):
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_damage(image: Image.Image):
    processed = preprocess_image(image)
    prediction = model.predict(processed, verbose=0)[0][0]

    # Expected mapping: crack = 0, no_crack = 1
    if prediction < 0.5:
        label = "Crack Detected"
        confidence = (1 - prediction) * 100
        health = "Poor"
        severity = "High Risk"
    else:
        label = "No Crack"
        confidence = prediction * 100
        health = "Good"
        severity = "Safe Surface"

    return label, confidence, health, severity

def save_uploaded_image(uploaded_file):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{uploaded_file.name}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

def append_log(filename, prediction, confidence, health, severity, saved_path):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "road_health": health,
        "severity": severity,
        "saved_path": saved_path
    }

    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)

        # Ensure old CSVs get missing columns
        if "severity" not in df.columns:
            df["severity"] = df["prediction"].apply(
                lambda x: "High Risk" if x == "Crack Detected" else "Safe Surface"
            )
        if "road_health" not in df.columns:
            df["road_health"] = df["prediction"].apply(
                lambda x: "Poor" if x == "Crack Detected" else "Good"
            )
        if "confidence" not in df.columns:
            df["confidence"] = 0.0
        if "saved_path" not in df.columns:
            df["saved_path"] = ""

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(LOG_FILE, index=False)

def load_logs():
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)

        if "severity" not in df.columns:
            df["severity"] = df["prediction"].apply(
                lambda x: "High Risk" if x == "Crack Detected" else "Safe Surface"
            )

        if "road_health" not in df.columns:
            df["road_health"] = df["prediction"].apply(
                lambda x: "Poor" if x == "Crack Detected" else "Good"
            )

        if "confidence" not in df.columns:
            df["confidence"] = 0.0

        if "saved_path" not in df.columns:
            df["saved_path"] = ""

        return df

    return pd.DataFrame(columns=[
        "timestamp", "filename", "prediction",
        "confidence", "road_health", "severity", "saved_path"
    ])

def get_dashboard_metrics(logs_df):
    total = len(logs_df)
    crack_cases = int((logs_df["prediction"] == "Crack Detected").sum()) if total else 0
    safe_cases = int((logs_df["prediction"] == "No Crack").sum()) if total else 0
    avg_conf = float(logs_df["confidence"].mean()) if total else 0.0
    return total, crack_cases, safe_cases, avg_conf

# ----------------------------
# Custom CSS - lighter theme
# ----------------------------
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at top left, rgba(96,165,250,0.22), transparent 26%),
        radial-gradient(circle at top right, rgba(192,132,252,0.18), transparent 28%),
        radial-gradient(circle at bottom left, rgba(52,211,153,0.14), transparent 24%),
        linear-gradient(180deg, #071427 0%, #0b1d35 45%, #08182c 100%);
    color: #f8fafc;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1280px;
}

header[data-testid="stHeader"] {
    background: transparent;
}

div.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(21, 34, 58, 0.78);
    color: #f8fafc;
    font-weight: 600;
    transition: 0.25s ease;
}

div.stButton > button:hover {
    border: 1px solid rgba(125,211,252,0.42);
    background: rgba(37, 52, 80, 0.96);
    color: white;
}

.nav-wrap {
    background: rgba(20, 32, 56, 0.62);
    border: 1px solid rgba(255,255,255,0.10);
    backdrop-filter: blur(14px);
    border-radius: 22px;
    padding: 1rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 12px 30px rgba(0,0,0,0.16);
}

.hero-card {
    background: linear-gradient(135deg, rgba(21,34,58,0.88), rgba(14,28,48,0.82));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 30px;
    padding: 2.4rem;
    backdrop-filter: blur(18px);
    box-shadow: 0 18px 50px rgba(0,0,0,0.20);
    margin-bottom: 1.4rem;
}

.main-title {
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.05;
    color: #f8fafc;
    margin-bottom: 0.75rem;
}

.sub-text {
    font-size: 1.03rem;
    color: #d6e2f0;
    line-height: 1.8;
}

.tag {
    display: inline-block;
    padding: 0.45rem 0.85rem;
    border-radius: 999px;
    background: rgba(96,165,250,0.18);
    border: 1px solid rgba(147,197,253,0.28);
    color: #e0f2fe;
    margin-right: 0.5rem;
    margin-top: 0.7rem;
    font-size: 0.9rem;
}

.glass-card {
    background: rgba(20,32,56,0.68);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 24px;
    padding: 1.4rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 14px 34px rgba(0,0,0,0.14);
}

.metric-card {
    background: linear-gradient(135deg, rgba(24,39,67,0.84), rgba(38,57,92,0.72));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 22px;
    padding: 1.2rem;
    min-height: 140px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.14);
}

.section-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.45rem;
}

.small-note {
    color: #c8d6e5;
    font-size: 0.93rem;
    line-height: 1.75;
}

.result-good {
    background: linear-gradient(135deg, rgba(22,101,52,0.30), rgba(34,197,94,0.12));
    border: 1px solid rgba(134,239,172,0.24);
    color: #dcfce7;
    padding: 1.2rem;
    border-radius: 20px;
    font-weight: 600;
}

.result-bad {
    background: linear-gradient(135deg, rgba(153,27,27,0.34), rgba(248,113,113,0.12));
    border: 1px solid rgba(252,165,165,0.24);
    color: #fee2e2;
    padding: 1.2rem;
    border-radius: 20px;
    font-weight: 600;
}

.info-strip {
    background: rgba(51, 65, 85, 0.38);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
}

.footer-note {
    text-align: center;
    color: #b9c7d8;
    font-size: 0.9rem;
    padding-top: 1.4rem;
}

div[data-testid="stFileUploader"] {
    background: rgba(20,32,56,0.52);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 0.35rem;
}

div[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}

@media (max-width: 992px) {
    .main-title {
        font-size: 2.5rem;
    }
    .hero-card {
        padding: 1.7rem;
    }
}

@media (max-width: 768px) {
    .main-title {
        font-size: 2rem;
    }
    .hero-card {
        padding: 1.2rem;
        border-radius: 22px;
    }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Navbar
# ----------------------------
st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)
nav1, nav2, nav3, nav4, nav5 = st.columns(5, gap="small")

with nav1:
    if st.button("Home", use_container_width=True):
        st.session_state.page_mode = "home"

with nav2:
    if st.button("Detect", use_container_width=True):
        st.session_state.page_mode = "detect"

with nav3:
    if st.button("Dashboard", use_container_width=True):
        st.session_state.page_mode = "history"

with nav4:
    if st.button("Cloud Ready", use_container_width=True):
        st.session_state.page_mode = "cloud"

with nav5:
    if st.button("About", use_container_width=True):
        st.session_state.page_mode = "about"

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Home Page
# ----------------------------
if st.session_state.page_mode == "home":
    left_logo, right_hero = st.columns([0.12, 0.88], gap="small")

    with left_logo:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=80)

    with right_hero:
        st.markdown("""
        <div class="hero-card">
            <div class="main-title">Road Damage Detection System</div>
            <div class="sub-text">
                A final-year-project-ready AI platform for automated road surface inspection.
                This system detects visible cracks from uploaded images, generates confidence-based
                predictions, stores detection history, and is structured for future cloud analytics integration.
            </div>
            <div>
                <span class="tag">Final Year Project</span>
                <span class="tag">MobileNetV2</span>
                <span class="tag">Responsive UI</span>
                <span class="tag">Prediction Logging</span>
                <span class="tag">Cloud Ready</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="section-title">Project Overview</div>
            <div class="small-note">
                This application automates the process of road damage analysis through deep learning.
                Instead of manual inspection, users can upload road images and receive a prediction
                indicating whether road damage is present or not.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.markdown("""
        <div class="glass-card">
            <div class="section-title">Workflow</div>
            <div class="small-note">
                1. Upload road image<br>
                2. Preprocess image<br>
                3. Predict using MobileNetV2 model<br>
                4. Display road health and confidence<br>
                5. Save result to history log
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        if st.button("Start Detection", use_container_width=True):
            st.session_state.page_mode = "detect"
            st.rerun()

    with col2:
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("""
            <div class="metric-card">
                <div class="section-title">Detection</div>
                <div class="small-note">Classifies uploaded road images as crack or no crack.</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown("""
            <div class="metric-card">
                <div class="section-title">Confidence</div>
                <div class="small-note">Displays model confidence for transparency in results.</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        r3, r4 = st.columns(2)
        with r3:
            st.markdown("""
            <div class="metric-card">
                <div class="section-title">Tracking</div>
                <div class="small-note">Stores predictions for review, reporting, and analytics.</div>
            </div>
            """, unsafe_allow_html=True)
        with r4:
            st.markdown("""
            <div class="metric-card">
                <div class="section-title">Scalability</div>
                <div class="small-note">Designed to later connect with Azure and IBM Cognos.</div>
            </div>
            """, unsafe_allow_html=True)

# ----------------------------
# Detect Page
# ----------------------------
elif st.session_state.page_mode == "detect":
    st.markdown("""
    <div class="hero-card">
        <div class="section-title">Road Image Analysis</div>
        <div class="small-note">
            Upload a road image to perform automated surface inspection using the trained MobileNetV2 model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Road Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1.2, 1], gap="large")

        with left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            with st.spinner("Analyzing road surface..."):
                prediction, confidence, health, severity = predict_damage(image)
                saved_path = save_uploaded_image(uploaded_file)
                append_log(uploaded_file.name, prediction, confidence, health, severity, saved_path)

            if prediction == "Crack Detected":
                st.markdown(f"""
                <div class="result-bad">
                    Prediction: {prediction}<br><br>
                    Confidence: {confidence:.2f}%<br>
                    Road Health: {health}<br>
                    Severity Level: {severity}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-good">
                    Prediction: {prediction}<br><br>
                    Confidence: {confidence:.2f}%<br>
                    Road Health: {health}<br>
                    Severity Level: {severity}
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.markdown("""
            <div class="info-strip">
                <b>Inspection Insight:</b> This version performs binary classification.
                Future improvements can include crack segmentation, severity analysis,
                GPS-based monitoring, and advanced analytics dashboards.
            </div>
            """, unsafe_allow_html=True)

# ----------------------------
# Dashboard / History Page
# ----------------------------
elif st.session_state.page_mode == "history":
    st.markdown("""
    <div class="hero-card">
        <div class="section-title">Detection Dashboard</div>
        <div class="small-note">
            Review detection history, monitor trends, and export structured records for reports and analytics.
        </div>
    </div>
    """, unsafe_allow_html=True)

    logs_df = load_logs()
    total, crack_cases, safe_cases, avg_conf = get_dashboard_metrics(logs_df)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Predictions", total)
    with m2:
        st.metric("Crack Cases", crack_cases)
    with m3:
        st.metric("Safe Cases", safe_cases)
    with m4:
        st.metric("Avg Confidence", f"{avg_conf:.2f}%")

    st.write("")

    if logs_df.empty:
        st.info("No predictions logged yet.")
    else:
        chart_col1, chart_col2 = st.columns(2, gap="large")

        with chart_col1:
            prediction_counts = logs_df["prediction"].value_counts().reset_index()
            prediction_counts.columns = ["Prediction", "Count"]

            pie_fig = px.pie(
                prediction_counts,
                names="Prediction",
                values="Count",
                hole=0.45,
                title="Crack vs No Crack Distribution",
                color="Prediction",
                color_discrete_map={
                    "Crack Detected": "#f97373",
                    "No Crack": "#4ade80"
                }
            )
            pie_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
                title_font=dict(size=18),
                legend=dict(orientation="h", y=-0.15)
            )
            st.plotly_chart(pie_fig, use_container_width=True)

        with chart_col2:
            severity_counts = logs_df["severity"].value_counts().reset_index()
            severity_counts.columns = ["Severity", "Count"]

            bar_fig = px.bar(
                severity_counts,
                x="Severity",
                y="Count",
                title="Severity Distribution",
                text="Count",
                color="Severity",
                color_discrete_sequence=["#60a5fa", "#f59e0b", "#ef4444", "#34d399"]
            )
            bar_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
                title_font=dict(size=18),
                xaxis_title="Severity Level",
                yaxis_title="Number of Images"
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        st.write("")

        if "timestamp" in logs_df.columns:
            logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"], errors="coerce")
            trend_df = logs_df.dropna(subset=["timestamp"]).copy()

            if not trend_df.empty:
                trend_df["date"] = trend_df["timestamp"].dt.date
                daily_counts = trend_df.groupby("date").size().reset_index(name="Count")

                line_fig = px.line(
                    daily_counts,
                    x="date",
                    y="Count",
                    markers=True,
                    title="Predictions Over Time"
                )
                line_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f8fafc"),
                    title_font=dict(size=18),
                    xaxis_title="Date",
                    yaxis_title="Predictions"
                )
                st.plotly_chart(line_fig, use_container_width=True)

        st.write("")
        st.dataframe(logs_df, use_container_width=True)

        csv = logs_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Prediction Log CSV",
            data=csv,
            file_name="prediction_logs.csv",
            mime="text/csv",
            use_container_width=True
        )

# ----------------------------
# Cloud Ready Page
# ----------------------------
elif st.session_state.page_mode == "cloud":
    st.markdown("""
    <div class="hero-card">
        <div class="section-title">Cloud and BI Integration Readiness</div>
        <div class="small-note">
            This app has been structured so it can later integrate with Azure services and IBM Cognos Analytics.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <div class="glass-card">
            <div class="section-title">Azure Path</div>
            <div class="small-note">
                Uploaded images and logs can later be sent to Azure Blob Storage or cloud databases
                for scalable storage and deployment.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="glass-card">
            <div class="section-title">IBM Cognos Path</div>
            <div class="small-note">
                Prediction logs can later be used for dashboards such as crack trends,
                inspection frequency, confidence analysis, and summary reporting.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------
# About Page
# ----------------------------
elif st.session_state.page_mode == "about":
    st.markdown("""
    <div class="hero-card">
        <div class="section-title">About the Project</div>
        <div class="small-note">
            <b>Project Title:</b> AI-Based Road Damage Detection System Using Deep Learning<br><br>
            <b>Model Used:</b> MobileNetV2<br><br>
            <b>Frontend:</b> Streamlit<br><br>
            <b>Purpose:</b> To automate road inspection and reduce manual effort by classifying road images as damaged or safe.<br><br>
            <b>Future Scope:</b> Crack segmentation, severity analysis, GPS-based monitoring, Azure deployment, and IBM Cognos dashboards.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
<div class="footer-note">
Built for Final Year Project • AI-Based Road Inspection Platform
</div>
""", unsafe_allow_html=True)