def calculate_symptom_score(cough, fever, night_sweats, weight_loss, smoking):
    score = 0
    if cough:
        score += 2
    if fever:
        score += 1
    if night_sweats:
        score += 2
    if weight_loss:
        score += 2
    if smoking:
        score += 1
    return score

def final_risk_level(tb_confidence, symptom_score):
    if tb_confidence > 80 and symptom_score >= 4:
        return "High Risk"
    elif tb_confidence > 50 or symptom_score >= 3:
        return "Moderate Risk"
    else:
        return "Low Risk"