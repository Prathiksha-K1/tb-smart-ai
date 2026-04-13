from datetime import datetime

def generate_report(patient_name, age, gender, prediction, confidence, symptom_score, final_risk, dip_scores, xray_score):
    now = datetime.now().strftime("%d-%m-%Y %I:%M %p")

    report = f"""
============================================================
                  TB-SMART AI HOSPITAL REPORT
============================================================

Report Generated On : {now}

---------------------- PATIENT DETAILS ----------------------

Patient Name        : {patient_name}
Age                 : {age}
Gender              : {gender}

--------------------- AI SCREENING RESULT -------------------

AI Prediction       : {prediction}
Prediction Confidence : {confidence:.2f}%
Chest X-ray Validity  : {xray_score:.2f}%
Symptom Score         : {symptom_score}
Final Clinical Risk   : {final_risk}

---------------- DIGITAL IMAGE PROCESSING -------------------

1. CLAHE Contrast Enhancement : {dip_scores.get("CLAHE Contrast Enhancement", 0)}%
2. Gaussian Denoising         : {dip_scores.get("Gaussian Denoising", 0)}%
3. Image Sharpening           : {dip_scores.get("Image Sharpening", 0)}%

-------------------- CLINICAL INTERPRETATION ----------------

This AI-assisted screening result is based on:
- Transfer learning model prediction
- Digital image enhancement pipeline
- Symptom-based risk prioritization
- Chest X-ray structural validation

----------------------- RECOMMENDATION ----------------------

This report is intended for AI-assisted clinical screening only.
It must not be used as the sole basis for final diagnosis.
Further confirmatory evaluation is advised if risk is moderate or high.

Suggested follow-up:
- Sputum examination / GeneXpert
- Clinical radiologist review
- Physician consultation

============================================================
                  END OF SCREENING REPORT
============================================================
"""
    return report