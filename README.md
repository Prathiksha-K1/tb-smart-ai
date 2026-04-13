# 🫁 AI-Based Pulmonary Tuberculosis Detection System

## 📌 Overview
This project presents an AI-based system for detecting pulmonary tuberculosis (TB) using chest X-ray images. The system leverages transfer learning models along with Digital Image Processing (DIP) techniques to provide accurate, fast, and explainable diagnostic support.

---

## 🚀 Features
- 🔍 TB Detection using Deep Learning  
- 🧠 Models: MobileNetV2, ResNet50, EfficientNet  
- 🖼️ Image preprocessing (CLAHE, denoising, sharpening)  
- 📊 Risk classification (Low / Moderate / High)  
- 📈 Patient history tracking using SQLite database  
- 🌐 Web-based interface built with Streamlit  
- 🔥 Grad-CAM visualization for explainability  
- 📁 Kaggle dataset used for training and evaluation  

---

## 🏗️ Tech Stack
- Python  
- Streamlit  
- PyTorch  
- OpenCV  
- Pandas / NumPy  
- SQLite  

---

## 📂 Project Structure
tb-smart-ai/
│
├── app.py
├── requirements.txt
├── tb_cases.db
├── tb_model.pth
├── uploads/
├── utils/
│   ├── auth.py
│   ├── dashboard_utils.py
│   ├── db.py
│   ├── gradcam_utils.py
│   ├── model_utils.py
│   ├── preprocess.py
│   ├── recommendation.py
│   ├── report.py
│   ├── risk_score.py

---

## ⚙️ Installation
git clone https://github.com/Prathiksha-K1/tb-smart-ai.git  
cd tb-smart-ai  
pip install -r requirements.txt  

---

## ▶️ Run the App
streamlit run app.py  

Then open in browser:  
http://localhost:8501  

---

## 📊 Dataset
- Source: Kaggle Chest X-ray Dataset  
- Contains TB-positive and normal chest X-ray images  

---

## 🧠 Model Details
- Transfer learning improves performance on limited medical data  
- Pre-trained CNN models are fine-tuned for TB detection  
- Achieves high classification accuracy (~99%)  
- Uses Grad-CAM for visual explanation  

---

## 📸 Screenshots
- Patient Dashboard  
- Prediction Output  
- Grad-CAM Visualization  

(You can add screenshots here later)

---

## 🔬 Explainable AI
Grad-CAM is used to highlight important regions in chest X-ray images. This improves transparency and helps doctors understand how the AI model makes decisions.

---

## ⚠️ Note
This system is intended for clinical decision support only and should not replace professional medical diagnosis.

---

## 👩‍💻 Authors
- Prathiksha K  
- Akshaya S  
- Charulatha R  

Guided by:  
Dr. R. Jagadeesh Kannan  

---

## 🌍 Future Scope
- Multi-disease detection (Pneumonia, Lung Cancer, COVID-19)  
- Mobile application development  
- Integration with hospital systems (EHR)  
- Real-time patient monitoring  
- Deployment on edge devices  

---

## ⭐ Acknowledgment
- World Health Organization (WHO)  
- Kaggle Dataset Contributors  
- Open-source AI community  
