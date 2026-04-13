import sqlite3

DB_NAME = "tb_cases.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            patient_name TEXT,
            age INTEGER,
            gender TEXT,
            cough INTEGER,
            fever INTEGER,
            night_sweats INTEGER,
            weight_loss INTEGER,
            smoking INTEGER,
            previous_tb INTEGER,
            immunocompromised INTEGER,
            prediction TEXT,
            confidence REAL,
            symptom_score INTEGER,
            final_risk TEXT,
            recommendation TEXT,
            image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

def insert_case(data):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        INSERT INTO cases (
            patient_id, patient_name, age, gender,
            cough, fever, night_sweats, weight_loss, smoking,
            previous_tb, immunocompromised,
            prediction, confidence, symptom_score, final_risk,
            recommendation, image_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)

    conn.commit()
    conn.close()

def get_all_cases():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM cases ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def seed_demo_data():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM cases")
    count = c.fetchone()[0]

    if count == 0:
        demo_cases = [
            ("TB001", "Ramesh", 54, "Male", 1, 1, 1, 1, 1, 0, 0, "Tuberculosis", 98.7, 5, "High Risk", "Urgent clinical review recommended", "uploads/demo1.png"),
            ("TB002", "Priya", 32, "Female", 1, 0, 0, 0, 0, 0, 0, "Normal", 94.3, 1, "Low Risk", "Routine follow-up advised", "uploads/demo2.png"),
            ("TB003", "Arun", 46, "Male", 1, 1, 0, 1, 1, 1, 0, "Tuberculosis", 91.2, 4, "Moderate Risk", "Further confirmatory testing recommended", "uploads/demo3.png"),
            ("TB004", "Sneha", 28, "Female", 0, 0, 0, 0, 0, 0, 0, "Normal", 97.1, 0, "Low Risk", "No strong evidence of TB", "uploads/demo4.png"),
            ("TB005", "Karthik", 61, "Male", 1, 1, 1, 1, 1, 1, 1, "Tuberculosis", 99.1, 5, "High Risk", "Immediate pulmonologist review advised", "uploads/demo5.png"),
            ("TB006", "Divya", 39, "Female", 1, 1, 0, 0, 0, 0, 0, "Tuberculosis", 88.5, 3, "Moderate Risk", "GeneXpert and sputum test suggested", "uploads/demo6.png"),
        ]

        for case in demo_cases:
            c.execute("""
                INSERT INTO cases (
                    patient_id, patient_name, age, gender,
                    cough, fever, night_sweats, weight_loss, smoking,
                    previous_tb, immunocompromised,
                    prediction, confidence, symptom_score, final_risk,
                    recommendation, image_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, case)

    conn.commit()
    conn.close()