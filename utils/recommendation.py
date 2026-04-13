def get_ai_recommendation(prediction, confidence, final_risk, age, smoking, previous_tb, immunocompromised):
    recommendations = []

    if prediction == "Tuberculosis":
        recommendations.append("AI model suggests possible pulmonary tuberculosis involvement.")
        recommendations.append("Recommend confirmatory testing such as sputum test / GeneXpert.")
    else:
        recommendations.append("AI model does not strongly suggest tuberculosis in current X-ray.")

    if final_risk == "High Risk":
        recommendations.append("Urgent clinical review is recommended due to high-risk screening outcome.")
    elif final_risk == "Moderate Risk":
        recommendations.append("Moderate-risk triage advised. Clinical follow-up is recommended.")
    else:
        recommendations.append("Low-risk AI screening outcome. Routine monitoring may be sufficient.")

    if smoking:
        recommendations.append("Smoking history may increase pulmonary complication risk.")
    if previous_tb:
        recommendations.append("Previous TB history increases the need for closer evaluation.")
    if immunocompromised:
        recommendations.append("Immunocompromised condition may increase clinical vulnerability.")
    if age >= 60:
        recommendations.append("Advanced age may require additional clinical caution.")

    if confidence < 70:
        recommendations.append("Prediction confidence is limited. Radiologist review is strongly advised.")

    return recommendations