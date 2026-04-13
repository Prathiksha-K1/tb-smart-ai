import streamlit as st

# Dummy users for project demo
USERS = {
    "doctor1": {"password": "doc123", "role": "Doctor", "name": "Dr. Arjun Kumar"},
    "admin1": {"password": "admin123", "role": "Admin", "name": "Hospital Admin"}
}

def login_user(username, password):
    if username in USERS and USERS[username]["password"] == password:
        return USERS[username]
    return None

def logout():
    for key in ["logged_in", "user_role", "user_name", "username"]:
        if key in st.session_state:
            del st.session_state[key]