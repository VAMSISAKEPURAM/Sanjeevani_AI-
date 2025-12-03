import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import mysql.connector
from mysql.connector import Error

# ---------------- LOGGER ----------------
logger = logging.getLogger("sanjeevani_ml")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(ch)

# ---------------- CONFIG PATHS ----------------
BASE_DIR = os.path.dirname(__file__)

ML_MODEL_PATH = os.path.join(BASE_DIR, "models", "weather_prediction_model.pkl")
FEATURES_JSON_PATH = os.path.join(BASE_DIR, "models", "ml_feature_names.json")
SCALER_PATH = os.path.join(BASE_DIR, "models", "ml_scaler.pkl")

DB_CONFIG = {
    "host": "localhost",
    "user": "flask_user",
    "password": "Flask@232",
    "database": "sanjeevani_ai",
    "auth_plugin": "mysql_native_password"
}

RESULTS_TABLE = "weather_prediction_results"

# ---------------- DB UTIL ----------------
def get_db_connection():
    """Return MySQL connection."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error:
        logger.exception("Unable to connect to DB")
        raise

def ensure_results_table(conn):
    """Creates the result table if missing. Adds day_index to store per-day predictions."""
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {RESULTS_TABLE} (
        id INT AUTO_INCREMENT PRIMARY KEY,
        diagnosis_id INT NOT NULL,
        day_index INT DEFAULT 0,
        prediction VARCHAR(255),
        confidence FLOAT,
        features_json JSON,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        INDEX (diagnosis_id),
        INDEX (day_index)
    ) ENGINE=InnoDB;
    """
    cur = conn.cursor()
    cur.execute(create_sql)
    conn.commit()
    cur.close()
    logger.info("Results table ensured (with day_index).")

# ---------------- MODEL WRAPPER ----------------
class MLModelWrapper:
    """
    A wrapper for sklearn ML model for weather-based pesticide prediction.
    Handles predict_proba fallbacks safely.
    """

    def __init__(self, path: str):
        self.path = path
        self.model = None
        self.feature_names = None
        self.scaler = None
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"ML model file not found: {self.path}")

        logger.info(f"Loading ML model: {self.path}")
        self.model = joblib.load(self.path)

        # Load feature names
        if hasattr(self.model, "feature_names_in_"):
            self.feature_names = list(self.model.feature_names_in_)
        elif os.path.exists(FEATURES_JSON_PATH):
            with open(FEATURES_JSON_PATH, "r") as f:
                self.feature_names = json.load(f)

        # Load scaler
        if os.path.exists(SCALER_PATH):
            try:
                self.scaler = joblib.load(SCALER_PATH)
                logger.info("Scaler loaded.")
            except:
                logger.warning("Scaler load failed.")

    # Softmax + sigmoid for fallback conversions
    @staticmethod
    def _sigmoid(x): return 1 / (1 + np.exp(-x))
    @staticmethod
    def _softmax(x):
        ex = np.exp(x - np.max(x, axis=1, keepdims=True))
        return ex / np.sum(ex, axis=1, keepdims=True)

    def predict_proba(self, X: pd.DataFrame):
        """Safe probability prediction wrapper."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)

        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            if scores.ndim == 1:
                p = self._sigmoid(scores)
                return np.vstack([1 - p, p]).T
            return self._softmax(scores)

        preds = self.model.predict(X)
        probs = np.zeros((len(preds), len(np.unique(preds))))
        for i, p in enumerate(preds):
            probs[i, p] = 1
        return probs

# ---------------- PREPROCESSOR ----------------
def prepare_features(df, disease_type, crop_type=None, wrapper=None):
    df = df.copy()

    rename_map = {
        "temp": "temperature",
        "wind": "wind_speed",
        "rain": "rainfall"
    }
    df = df.rename(columns=rename_map)

    for col in ["temperature", "humidity", "wind_speed", "rainfall"]:
        if col not in df:
            df[col] = np.nan

    df["disease_type"] = disease_type
    df["crop_type"] = crop_type if crop_type else "unknown"

    X = df[["temperature", "humidity", "wind_speed", "rainfall", "crop_type", "disease_type"]]

    X = pd.get_dummies(X, drop_first=True)

    if wrapper and wrapper.feature_names:
        for col in wrapper.feature_names:
            if col not in X:
                X[col] = 0
        X = X[wrapper.feature_names]

    if wrapper and wrapper.scaler:
        num_cols = X.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            X[num_cols] = wrapper.scaler.transform(X[num_cols])

    return X.astype(float)

# ---------------- MAIN PIPELINE ----------------
def run_weather_prediction(diagnosis_id: int, wrapper: Optional[MLModelWrapper] = None):
    conn = get_db_connection()
    ensure_results_table(conn)
    cur = conn.cursor(dictionary=True)

    # 1. Read the diagnosis row
    cur.execute("SELECT * FROM plant_diagnosis WHERE id=%s", (diagnosis_id,))
    diag = cur.fetchone()
    if not diag:
        cur.close()
        conn.close()
        raise ValueError("Diagnosis ID not found in DB.")

    disease_name = diag.get("disease_name", "unknown")
    crop_type = diag.get("crop_type")
    location = diag.get("location")

    # 2. Fetch up to 5 latest weather rows for location (ASC so day_index 0..4)
    if location:
        cur.execute("""
            SELECT * FROM weather_data 
            WHERE location=%s
            ORDER BY _date ASC, slot ASC
            LIMIT 5
        """, (location,))
    else:
        cur.execute("SELECT * FROM weather_data ORDER BY _date ASC, slot ASC LIMIT 5")

    weather_rows = cur.fetchall()
    if not weather_rows:
        cur.close()
        conn.close()
        raise ValueError("No weather data found.")

    # 3. Load model wrapper (or use provided)
    wrapper = wrapper or MLModelWrapper(ML_MODEL_PATH)

    inserted_ids = []

    # 4. For each day row, prepare features, run model and insert result
    for day_index, w in enumerate(weather_rows):
        # w contains keys like _date, slot, temp, humidity, wind, rain, location
        # prepare a single-row dataframe compatible with prepare_features
        w_copy = w.copy()

        # Keep keys matching rename_map in prepare_features: temp, wind, rain, humidity
        # if column names are numeric strings or bytes, convert as necessary (guard)
        df = pd.DataFrame([w_copy])

        # prepare features using your function
        X = prepare_features(df, disease_name, crop_type, wrapper)

        # Predict
        probs = wrapper.predict_proba(X)
        best = int(np.argmax(probs[0]))
        confidence = float(np.max(probs[0]))

        if hasattr(wrapper.model, "classes_"):
            predicted_label = str(wrapper.model.classes_[best])
        else:
            predicted_label = str(best)

        features_json = X.iloc[0].to_json()

        # Insert one row per day with day_index
        cur.execute(f"""
            INSERT INTO {RESULTS_TABLE}
            (diagnosis_id, day_index, prediction, confidence, features_json, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (diagnosis_id, day_index, predicted_label, confidence, features_json, datetime.now()))

        conn.commit()
        inserted_ids.append(cur.lastrowid)

    cur.close()
    conn.close()

    # Build return structure
    results = []
    for idx, rid in enumerate(inserted_ids):
        results.append({
            "result_id": rid,
            "diagnosis_id": diagnosis_id,
            "day_index": idx,
            "prediction": None,  # prediction available in DB if needed
            "confidence": None
        })

    return results
