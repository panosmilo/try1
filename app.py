import streamlit as st
import pandas as pd
import joblib
import os
import sqlite3
from datetime import datetime

# --- Ρυθμίσεις Σελίδας ---
st.set_page_config(page_title="Restaurant Predictor", layout="wide")

# --- Σύνδεση με SQLite ---
def init_db():
    conn = sqlite3.connect('restaurant.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            Ημερομηνία TEXT PRIMARY KEY,
            Ημέρα_Εβδομάδας INTEGER,
            Θερμοκρασία REAL,
            Βροχόπτωση REAL,
            Καιρός TEXT,
            Αργία INTEGER,
            Ειδική_Ημέρα TEXT,
            Ώρες_Αιχμής TEXT,
            Παραγγελίες_Delivery INTEGER,
            Συνολικός_Πελάτες INTEGER,
            Εποχή INTEGER,
            Διαφημίσεις INTEGER,
            Τουριστική_Περίοδος INTEGER
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS food (
            Ημερομηνία TEXT,
            Φαγητό TEXT,
            Πωλήσεις_Σύνολο INTEGER,
            Πωλήσεις_Delivery INTEGER,
            Εξαντλήθηκε INTEGER,
            Ώρα_Εξάντλησης TEXT,
            Κατηγορία TEXT,
            Τιμή REAL,
            Διαθεσιμότητα INTEGER,
            FOREIGN KEY (Ημερομηνία) REFERENCES attendance (Ημερομηνία)
        )
    ''')
    
    conn.commit()
    return conn

# --- Φόρτωση Μοντέλων ---
@st.cache_resource
def load_models():
    try:
        model_att = joblib.load("models/attendance_model.pkl")
        model_food = joblib.load("models/food_model.pkl")
        return model_att, model_food
    except FileNotFoundError:
        st.error("Δεν βρέθηκαν μοντέλα. Εκτελέστε πρώτα το train_models.py")
        return None, None

# --- Αρχική Σελίδα ---
def home():
    st.title("🍽️ Restaurant Prediction System")
    st.write("Επιλέξτε μια καρτέλα από το μενού στα αριστερά.")

# --- Πρόβλεψη Προσέλευσης ---
def attendance_prediction():
    st.header("📊 Πρόβλεψη Προσέλευσης")
    with st.form("attendance_form"):
        date = st.date_input("Ημερομηνία")
        temp = st.number_input("Θερμοκρασία (°C)", min_value=-10, max_value=50, value=20)
        rain = st.number_input("Βροχόπτωση (mm)", min_value=0, value=0)
        is_holiday = st.selectbox("Αργία", ["Όχι", "Ναι"])
        season = (date.month % 12 + 3) // 3  # 1=Χειμώνας, ..., 4=Φθινόπωρο
        
        if st.form_submit_button("Πρόβλεψη"):
            model_att, _ = load_models()
            if model_att:
                X = [[
                    date.weekday(), temp, rain, 
                    1 if is_holiday == "Ναι" else 0, 
                    season, 0, 0  # Διαφημίσεις=0, Τουρισμός=0 (προσωρινά)
                ]]
                pred = int(model_att.predict(X)[0])
                st.success(f"**Προβλεπόμενοι πελάτες:** {pred}")
            else:
                st.error("Το μοντέλο δεν είναι διαθέσιμο.")

# --- Πρόβλεψη Φαγητών ---
def food_prediction():
    st.header("🍲 Πρόβλεψη Πωλήσεων Φαγητού")
    with st.form("food_form"):
        food = st.text_input("Όνομα Φαγητού")
        date = st.date_input("Ημερομηνία")
        temp = st.number_input("Θερμοκρασία (°C)", value=20)
        is_holiday = st.selectbox("Αργία", ["Όχι", "Ναι"])
        price = st.number_input("Τιμή (€)", min_value=0.0, value=10.0)
        
        if st.form_submit_button("Πρόβλεψη"):
            model_att, model_food = load_models()
            if model_att and model_food:
                # Πρόβλεψη προσέλευσης πρώτα
                season = (date.month % 12 + 3) // 3
                X_att = [[
                    date.weekday(), temp, 0, 
                    1 if is_holiday == "Ναι" else 0, 
                    season, 0, 0
                ]]
                pred_customers = int(model_att.predict(X_att)[0])
                
                # Πρόβλεψη φαγητού
                X_food = [[
                    date.weekday(), temp, 
                    1 if is_holiday == "Ναι" else 0, 
                    season, pred_customers, int(pred_customers * 0.3), price, 1
                ]]
                pred_sales = int(model_food.predict(X_food)[0])
                st.success(f"**Προβλεπόμενες πωλήσεις για {food}:** {pred_sales}")
            else:
                st.error("Τα μοντέλα δεν είναι διαθέσιμα.")

# --- Υποβολή CSV ---
def upload_csv():
    st.header("📤 Υποβολή CSV")
    conn = init_db()
    
    # CSV Προσέλευσης
    st.subheader("Δεδομένα Προσέλευσης")
    uploaded_att = st.file_uploader("Ανέβασμα CSV Προσέλευσης", type="csv", key="att")
    if uploaded_att:
        df_att = pd.read_csv(uploaded_att)
        df_att.to_sql("attendance", conn, if_exists="append", index=False)
        st.success("Αποθηκεύτηκε!")
    
    # CSV Φαγητών
    st.subheader("Δεδομένα Φαγητών")
    uploaded_food = st.file_uploader("Ανέβασμα CSV Φαγητών", type="csv", key="food")
    if uploaded_food:
        df_food = pd.read_csv(uploaded_food)
        df_food.to_sql("food", conn, if_exists="append", index=False)
        st.success("Αποθηκεύτηκε!")
    
    conn.close()

# --- Βάση Δεδομένων ---
def database():
    st.header("🗄️ Βάση Δεδομένων")
    conn = init_db()
    
    st.subheader("Δεδομένα Προσέλευσης")
    df_att = pd.read_sql("SELECT * FROM attendance", conn)
    st.dataframe(df_att)
    
    st.subheader("Δεδομένα Φαγητών")
    df_food = pd.read_sql("SELECT * FROM food", conn)
    st.dataframe(df_food)
    
    if st.button("Εκκαθάριση Δεδομένων"):
        conn.cursor().execute("DELETE FROM attendance")
        conn.cursor().execute("DELETE FROM food")
        conn.commit()
        st.warning("Τα δεδομένα διαγράφηκαν.")
    
    conn.close()

# --- Κύριο Μενού ---
menu = {
    "Αρχική": home,
    "Πρόβλεψη Προσέλευσης": attendance_prediction,
    "Πρόβλεψη Φαγητών": food_prediction,
    "Υποβολή CSV": upload_csv,
    "Βάση Δεδομένων": database
}

selected = st.sidebar.selectbox("Επιλογή", menu.keys())
menu[selected]()