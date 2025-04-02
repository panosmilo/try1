import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
from datetime import datetime

def log_message(message):
    os.makedirs("logs", exist_ok=True)
    with open("logs/training_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()}: {message}\n")

def load_data():
    try:
        # Φόρτωση CSV προσέλευσης
        attendance_files = [f"attendance_data/{f}" for f in os.listdir("attendance_data") if f.startswith("attendance_")]
        df_attendance = pd.concat([pd.read_csv(f) for f in attendance_files], ignore_index=True)
        
        # Φόρτωση CSV φαγητών
        food_files = [f"food_data/{f}" for f in os.listdir("food_data") if f.startswith("food_")]
        df_food = pd.concat([pd.read_csv(f) for f in food_files], ignore_index=True)
        
        # Συγχώνευση
        df_merged = pd.merge(df_food, df_attendance, on="Ημερομηνία", how="left")
        return df_attendance, df_merged
    
    except Exception as e:
        log_message(f"Σφάλμα φόρτωσης: {str(e)}")
        raise

def train_models():
    try:
        df_attendance, df_merged = load_data()
        
        # 1. Μοντέλο Προσέλευσης
        X_att = df_attendance[['Ημέρα_Εβδομάδας', 'Θερμοκρασία', 'Βροχόπτωση', 'Αργία', 'Εποχή', 'Διαφημίσεις', 'Τουριστική_Περίοδος']]
        y_att = df_attendance['Συνολικός_Πελάτες']
        
        model_att = RandomForestRegressor(n_estimators=100, random_state=42)
        model_att.fit(X_att, y_att)
        joblib.dump(model_att, "models/attendance_model.pkl")
        
        # 2. Μοντέλο Φαγητών
        X_food = df_merged[['Ημέρα_Εβδομάδας', 'Θερμοκρασία', 'Αργία', 'Εποχή', 'Συνολικός_Πελάτες', 'Παραγγελίες_Delivery', 'Τιμή', 'Διαθεσιμότητα']]
        y_food = df_merged['Πωλήσεις_Σύνολο']
        
        model_food = RandomForestRegressor(n_estimators=100, random_state=42)
        model_food.fit(X_food, y_food)
        joblib.dump(model_food, "models/food_model.pkl")
        
        log_message("Εκπαίδευση ολοκληρώθηκε!")
    
    except Exception as e:
        log_message(f"Σφάλμα εκπαίδευσης: {str(e)}")
        raise

if __name__ == "__main__":
    train_models()