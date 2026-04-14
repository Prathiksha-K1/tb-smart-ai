from datetime import datetime

def center_text(text, width=72):
    return text.center(width)

def generate_report(patient_name, age, gender, prediction, confidence, symptom_score, final_risk, dip_scores, xray_score):
    
    now = datetime.now().strftime("%d-%m-%Y %I:%M %p")

    # helper for alignment
    def line(label, value):
        return f"{label:<30}: {value}"

    report = f"""
{center_text("======================================")}
{center_text("TB-SMART AI HOSPITAL REPORT")}
{center_text("======================================")}

{line("Report Generated On", now)}

{center_text("-----------------------------  PATIENT DETAILS ---------------------------------------")}

{line("Patient Name", patient_name)}
{line("Age", age)}
{line("Gender", gender)}

{center_text("----------------------------- AI SCREENING RESULT ------------------------------------")}

{line("AI Prediction", prediction)}
{line("Prediction Confidence", f"{confidence:.2f}%")}
{line("Chest X-ray Validity", f"{xray_score:.2f}%")}
{line("Symptom Score", symptom_score)}
{line("Final Clinical Risk", final_risk)}

{center_text("--------------------------- DIGITAL IMAGE PROCESSING ---------------------------------")}

{line("CLAHE Contrast Enhancement", f"{dip_scores.get('CLAHE Contrast Enhancement', 0)}%")}
{line("Gaussian Denoising", f"{dip_scores.get('Gaussian Denoising', 0)}%")}
{line("Image Sharpening", f"{dip_scores.get('Image Sharpening', 0)}%")}

{center_text("--------------------------- CLINICAL INTERPRETATION ----------------------------------")}

This AI-assisted screening result is based on:
- Transfer learning model prediction
- Digital image enhancement pipeline
- Symptom-based risk prioritization
- Chest X-ray structural validation

{center_text("-------------------------------- RECOMMENDATION --------------------------------------")}

This report is intended for AI-assisted clinical screening only.
It must not be used as the sole basis for final diagnosis.

Further confirmatory evaluation is advised if risk is moderate or high.

Suggested follow-up:
- Sputum examination / GeneXpert
- Clinical radiologist review
- Physician consultation

"""
    return report