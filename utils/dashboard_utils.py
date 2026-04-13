import sqlite3
import pandas as pd

DB_NAME = "tb_cases.db"

def load_cases_df():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM cases ORDER BY created_at DESC", conn)
    conn.close()
    return df

def get_dashboard_stats():
    df = load_cases_df()

    if df.empty:
        return {
            "total_cases": 0,
            "tb_cases": 0,
            "normal_cases": 0,
            "high_risk": 0,
            "moderate_risk": 0,
            "low_risk": 0
        }

    return {
        "total_cases": len(df),
        "tb_cases": len(df[df["prediction"] == "Tuberculosis"]),
        "normal_cases": len(df[df["prediction"] == "Normal"]),
        "high_risk": len(df[df["final_risk"] == "High Risk"]),
        "moderate_risk": len(df[df["final_risk"] == "Moderate Risk"]),
        "low_risk": len(df[df["final_risk"] == "Low Risk"]),
    }