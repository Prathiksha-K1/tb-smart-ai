import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

from utils.auth import login_user, logout
from utils.model_utils import load_model, predict_image
from utils.risk_score import calculate_symptom_score
from utils.db import init_db, insert_case, seed_demo_data
from utils.report import generate_report
from utils.gradcam_utils import generate_simple_heatmap
from utils.preprocess import apply_dip_pipeline, get_xray_validity_score, is_valid_chest_xray
from utils.recommendation import get_ai_recommendation
from utils.dashboard_utils import get_dashboard_stats, load_cases_df
import numpy as np

def fix_display_image(img):
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img

    img_np = img_np.astype("float32")
    img_np = img_np - img_np.min()

    if img_np.max() > 0:
        img_np = img_np / (img_np.max() + 1e-8)

    img_np = (img_np * 255).astype("uint8")

    return Image.fromarray(img_np)
# ----------------------------
# APP CONFIG
# ----------------------------
st.set_page_config(
    page_title="TB-SMART AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# STYLING
# ----------------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0b1f4d 0%, #08142e 40%, #020617 100%);
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #081120 0%, #0b1730 100%);
    border-right: 1px solid rgba(96,165,250,0.22);
}

/* Titles */
.main-title {
    font-size: 48px;
    font-weight: 900;
    color: #ffffff;
    text-shadow: 0 0 18px #60a5fa, 0 0 36px rgba(96,165,250,0.35);
    text-align: center;
}

.sub-title {
    font-size: 18px;
    color: #dbeafe;
    margin-bottom: 20px;
    text-align: center;
}

.section-title {
    font-size: 30px;
    font-weight: 900;
    margin-bottom: 8px;
    color: #ffffff;
    text-shadow: 0 0 14px rgba(96,165,250,0.28);
}

.small-caption {
    color: #cbd5e1;
    font-size: 14px;
    margin-bottom: 18px;
}

/* Cards */
.metric-box {
    background: linear-gradient(145deg, rgba(10,20,40,0.95), rgba(15,35,70,0.88));
    backdrop-filter: blur(14px);
    padding: 22px;
    border-radius: 22px;
    text-align: center;
    border: 1px solid rgba(96,165,250,0.25);
    box-shadow: 0 0 20px rgba(96,165,250,0.14);
    margin-bottom: 15px;
}

.metric-title {
    font-size: 15px;
    color: #bfdbfe;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 34px;
    font-weight: 900;
    color: #ffffff;
}

.report-card {
    background: linear-gradient(145deg, rgba(10,20,40,0.96), rgba(15,35,70,0.90));
    border: 1px solid rgba(96,165,250,0.25);
    box-shadow: 0 0 22px rgba(96,165,250,0.12);
    border-radius: 22px;
    padding: 32px;
    max-width: 980px;
    margin: auto;
}

.stButton > button {
    background: linear-gradient(90deg, #2563eb, #38bdf8);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 10px 22px;
    font-weight: 800;
    box-shadow: 0 0 14px rgba(56,189,248,0.24);
}

.stDownloadButton > button {
    background: linear-gradient(90deg, #0ea5e9, #38bdf8);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 10px 22px;
    font-weight: 800;
    box-shadow: 0 0 14px rgba(56,189,248,0.24);
}

hr {
    border: 0;
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(96,165,250,0.3), transparent);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# INIT
# ----------------------------
init_db()
seed_demo_data()

if not os.path.exists("uploads"):
    os.makedirs("uploads")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "last_report" not in st.session_state:
    st.session_state.last_report = None

# ----------------------------
# LOGIN PAGE
# ----------------------------
if not st.session_state.logged_in:
    st.markdown("<div class='main-title'>🫁 TB-SMART AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>AI-Based Pulmonary Tuberculosis Detection Using Transfer Learning Models</div>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:
        st.markdown("## 🔐 Secure Clinical Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user_role = user["role"]
                st.session_state.user_name = user["name"]
                st.success(f"Welcome, {user['name']}!")
                st.rerun()
            else:
                st.error("Invalid username or password")

        st.markdown("### Demo Credentials")
        st.code("Doctor Login:\nUsername: doctor1\nPassword: doc123")
        st.code("Admin Login:\nUsername: admin1\nPassword: admin123")

# ----------------------------
# MAIN APP
# ----------------------------
else:
    st.sidebar.markdown("## 🫁 TB-SMART AI")
    st.sidebar.markdown(f"### 👤 {st.session_state.user_name}")
    st.sidebar.caption(f"Role: {st.session_state.user_role}")
    st.sidebar.markdown("---")

    menu_options = ["Dashboard", "New Screening", "Patient History", "Report Center", "About Project"]
    if st.session_state.user_role == "Admin":
        menu_options.insert(4, "Admin Panel")

    selected_page = st.sidebar.radio("Navigation", menu_options)

    if st.sidebar.button("Logout"):
        logout()
        st.rerun()

    # ----------------------------
    # DASHBOARD
    # ----------------------------
    if selected_page == "Dashboard":
        st.markdown("<div class='section-title'>🏥 Hospital Dashboard</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-caption'>Overview of TB screening activity and AI-assisted clinical triage</div>", unsafe_allow_html=True)

        stats = get_dashboard_stats()

        cols = st.columns(3)
        metrics = [
            ("Total Cases", stats["total_cases"]),
            ("TB Suspected", stats["tb_cases"]),
            ("Normal Cases", stats["normal_cases"]),
            ("High Risk", stats["high_risk"]),
            ("Moderate Risk", stats["moderate_risk"]),
            ("Low Risk", stats["low_risk"]),
        ]

        for i, (title, value) in enumerate(metrics):
            with cols[i % 3]:
                st.markdown(f"""
                    <div class='metric-box'>
                        <div class='metric-title'>{title}</div>
                        <div class='metric-value'>{value}</div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📊 3D Clinical Risk Visualization")

        fig = go.Figure(data=[
            go.Scatter3d(
                x=[1, 2, 3],
                y=[1, 1, 1],
                z=[stats["high_risk"], stats["moderate_risk"], stats["low_risk"]],
                mode='markers+text',
                marker=dict(
                    size=[
                        stats["high_risk"] * 10 + 16,
                        stats["moderate_risk"] * 10 + 16,
                        stats["low_risk"] * 10 + 16
                    ],
                    color=[
                        stats["high_risk"],
                        stats["moderate_risk"],
                        stats["low_risk"]
                    ],
                    colorscale='Blues',
                    opacity=0.92,
                    line=dict(width=2, color='white')
                ),
                text=["High Risk", "Moderate Risk", "Low Risk"],
                textposition="top center"
            )
        ])

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="Risk Category",
                    tickvals=[1, 2, 3],
                    ticktext=["High", "Moderate", "Low"],
                    backgroundcolor="rgba(0,0,0,0)"
                ),
                yaxis=dict(title="", visible=False, backgroundcolor="rgba(0,0,0,0)"),
                zaxis=dict(title="Case Count", backgroundcolor="rgba(0,0,0,0)")
            ),
            margin=dict(l=0, r=0, b=0, t=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🕒 Recent Cases")

        df = load_cases_df()
        if not df.empty:
            display_df = df[[
                "patient_id", "patient_name", "age", "gender",
                "prediction", "confidence", "final_risk", "created_at"
            ]].head(10)
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No patient cases available yet.")

    # ----------------------------
    # NEW SCREENING
    # ----------------------------
    elif selected_page == "New Screening":
        st.markdown("<div class='section-title'>🩺 New TB Screening Case</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-caption'>Doctor screening module for AI-assisted pulmonary TB analysis</div>", unsafe_allow_html=True)

        model = load_model("tb_model.pth")

        col1, col2 = st.columns([1.15, 1])

        with col1:
            patient_id = st.text_input("Patient ID")
            patient_name = st.text_input("Patient Name")
            age = st.number_input("Age", min_value=1, max_value=120, step=1)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])

            st.markdown("### Symptoms & Risk Factors")
            cough = st.checkbox("Cough > 2 weeks")
            fever = st.checkbox("Fever")
            night_sweats = st.checkbox("Night Sweats")
            weight_loss = st.checkbox("Weight Loss")
            smoking = st.checkbox("Smoking History")
            previous_tb = st.checkbox("Previous TB History")
            immunocompromised = st.checkbox("Immunocompromised Condition")

        with col2:
            uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            valid_xray, xray_score = is_valid_chest_xray(image)

            st.markdown("---")
            st.subheader("🧾 Input Validation")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Chest X-ray Validity Score", f"{xray_score:.2f}%")
            with c2:
                status = "Valid / Acceptable" if valid_xray else "INVALID INPUT"
                st.metric("Input Status", status)

            if not valid_xray:
                st.error("❌ Invalid Input: This image does not appear to be a valid frontal chest X-ray for TB screening.")
                st.warning("Please upload a proper PA/AP chest X-ray image only.")
                st.stop()

            # DIP
            stage_images, dip_scores, final_dip_image = apply_dip_pipeline(image)

            st.markdown("---")
            st.subheader("🧪 Digital Image Processing Pipeline")

            # show 8 methods in rows of 4
            for i in range(0, len(stage_images), 4):
                cols = st.columns(4)
                batch = stage_images[i:i+4]
                for j, (title, img) in enumerate(batch):
                    with cols[j]:
                        fixed_img = fix_display_image(img)
                        st.image(fixed_img, caption=title, use_container_width=True)

            st.markdown("### 📈 DIP Method Contribution")
            dip_df = pd.DataFrame({
                "Method": list(dip_scores.keys()),
                "Improvement %": list(dip_scores.values())
            })
            st.dataframe(dip_df, use_container_width=True)
            st.bar_chart(dip_df.set_index("Method"))

            if st.button("Analyze"):
                prediction, confidence, input_tensor = predict_image(model, image)

                tb_confidence = confidence if prediction == "Tuberculosis" else (100 - confidence)
                symptom_score = calculate_symptom_score(cough, fever, night_sweats, weight_loss, smoking)

                if tb_confidence > 85 and (symptom_score >= 4 or previous_tb or immunocompromised):
                    final_risk = "High Risk"
                elif tb_confidence > 50 or symptom_score >= 3:
                    final_risk = "Moderate Risk"
                else:
                    final_risk = "Low Risk"

                recommendations = get_ai_recommendation(
                    prediction, confidence, final_risk,
                    age, smoking, previous_tb, immunocompromised
                )

                heatmap, overlay = generate_simple_heatmap(input_tensor)

                st.success("Analysis Completed")

                st.markdown("### 🧾 Clinical Summary")
                st.info("AI-assisted screening result based on chest X-ray analysis and symptom evaluation.")

                colA, colB = st.columns([1, 1], gap="large")

                with colA:
                    st.subheader("🧠 AI Prediction Result")
                    st.write(f"**Prediction:** {prediction}")
                    st.write(f"**Confidence:** {confidence:.2f}%")
                    st.write(f"**Symptom Score:** {symptom_score}")
                    st.write(f"**Clinical Priority:** {final_risk}")

                    if confidence >= 90:
                        st.success("High confidence prediction")
                    elif confidence >= 70:
                        st.warning("Moderate confidence prediction")
                    else:
                        st.error("Low confidence prediction")

                with colB:
                    st.subheader("🩻 AI Attention Map")
                    st.image(overlay, caption="Suspicious Region Highlight", width=360)

                st.markdown("---")
                st.subheader("🧠 AI Recommendation Engine")
                for rec in recommendations:
                    st.write(f"- {rec}")

                report = generate_report(
                    patient_name, age, gender,
                    prediction, confidence,
                    symptom_score, final_risk,
                    dip_scores,
                    xray_score
                )

                st.session_state.last_report = report

                st.markdown("---")
                st.markdown("<div class='section-title'>📄 Clinical TB Report</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='report-card'>
                    <pre style="color:#e2e8f0; font-size:15px; white-space:pre-wrap;">{report}</pre>
                </div>
                """, unsafe_allow_html=True)

                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"{patient_name}_tb_report.txt",
                    mime="text/plain"
                )

                image_path = os.path.join("uploads", uploaded_file.name)
                image.save(image_path)

                insert_case((
                    patient_id,
                    patient_name,
                    age,
                    gender,
                    int(cough),
                    int(fever),
                    int(night_sweats),
                    int(weight_loss),
                    int(smoking),
                    int(previous_tb),
                    int(immunocompromised),
                    prediction,
                    confidence,
                    symptom_score,
                    final_risk,
                    "\n".join(recommendations),
                    image_path
                ))

                st.success("Case saved successfully and report added to Report Center.")

    # ----------------------------
    # PATIENT HISTORY
    # ----------------------------
    elif selected_page == "Patient History":
        st.markdown("<div class='section-title'>📁 Patient History</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-caption'>Review previously screened patient records</div>", unsafe_allow_html=True)

        df = load_cases_df()
        if df.empty:
            st.info("No patient records found.")
        else:
            search = st.text_input("Search by Patient Name or ID")
            if search:
                df = df[
                    df["patient_name"].astype(str).str.contains(search, case=False, na=False) |
                    df["patient_id"].astype(str).str.contains(search, case=False, na=False)
                ]

            st.dataframe(df[[
                "patient_id", "patient_name", "age", "gender",
                "prediction", "confidence", "final_risk", "created_at"
            ]], use_container_width=True)

    # ----------------------------
    # REPORT CENTER
    # ----------------------------
    elif selected_page == "Report Center":
        st.markdown("<div class='section-title'>📄 Report Center</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-caption'>Centralized hospital-style screening report view</div>", unsafe_allow_html=True)

        if st.session_state.last_report:
            st.markdown(f"""
            <div class='report-card'>
                <pre style="color:#e2e8f0; font-size:15px; white-space:pre-wrap;">{st.session_state.last_report}</pre>
            </div>
            """, unsafe_allow_html=True)

            st.download_button(
                label="Download Latest Report",
                data=st.session_state.last_report,
                file_name="latest_tb_report.txt",
                mime="text/plain"
            )
        else:
            st.info("No report generated yet. Analyze a patient case first from New Screening.")

    # ----------------------------
    # ADMIN PANEL
    # ----------------------------
    elif selected_page == "Admin Panel":
        if st.session_state.user_role != "Admin":
            st.error("Access denied. Admin only.")
        else:
            st.markdown("<div class='section-title'>🛠️ Hospital Admin Panel</div>", unsafe_allow_html=True)
            st.markdown("<div class='small-caption'>Administrative overview and exported patient data access</div>", unsafe_allow_html=True)

            df = load_cases_df()
            if df.empty:
                st.info("No hospital records available.")
            else:
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download All Cases as CSV", csv, "tb_cases_export.csv", "text/csv")

    # ----------------------------
    # ABOUT PROJECT
    # ----------------------------
    elif selected_page == "About Project":
        st.markdown("<div class='section-title'>ℹ️ About Project</div>", unsafe_allow_html=True)

        st.markdown("""
        ## 🫁 AI-Based Pulmonary Tuberculosis Detection Using Transfer Learning Models

        ### Project Objective
        To develop a clinical AI-assisted screening platform for pulmonary tuberculosis using chest X-ray analysis, digital image processing, and intelligent risk prioritization.

        ### Core Subjects Covered
        - **Digital Image Processing**
          - Grayscale normalization
          - Histogram Equalization
          - CLAHE enhancement
          - Median filtering
          - Gaussian denoising
          - Edge enhancement
          - Morphological cleanup
          - Image sharpening
        - **Artificial Intelligence**
          - Transfer Learning using DenseNet121
          - Chest X-ray binary classification
        - **Design and Analysis of Algorithms (DAA)**
          - Risk scoring logic
          - Decision support recommendation engine
        - **Computer Vision**
          - Chest X-ray validation
          - AI attention map / explainability

        ### Innovative Features
        - Strict chest X-ray input validation
        - Multi-stage DIP pipeline visualization
        - 3D risk dashboard
        - AI explainability heatmap
        - Multi-role hospital access
        - Centralized report center

        ### Disclaimer
        This system is intended for educational and AI-assisted screening purposes only.
        """)