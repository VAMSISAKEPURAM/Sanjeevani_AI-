from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import requests
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import logging
import json

# ----------------- IMPORT WEATHER ML MODULE -----------------
from ml_integration import (
    run_weather_prediction,
    MLModelWrapper,
)

# ------------------ CONFIG ------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "my_cnn_model_New.keras")

CLASS_NAMES = ['Corn_Common_Rust', 'Corn_Corn_Blight', 'Corn_Gray_Leaf_Spot', 'Tomato_healthy',
               'Ground_Nut_early_leaf_spot_1', 'Ground_Nut_early_rust_1', 'Ground_Nut_healthy_leaf_1',
               'Ground_Nut_late_leaf_spot_1', 'Ground_Nut_nutrition_deficiency_1', 'Ground_Nut_rust_1',
               'Paddy_BLAST', 'Paddy_BLIGHT', 'Paddy_BROWNSPOT', 'Paddy_HEALTHY',
               'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
               'Potato___Late_blight', 'Potato___healthy', 'Tomato_Early_blight', 'Tomato_Late_blight',
               'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato__Target_Spot',
               'Tomato__Tomato_mosaic_virus', 'Corn_Healthy']

INPUT_SIZE = (150, 150)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024
WEATHER_API_KEY = "0e353b69178bef3dcaa9a2349e7ef65a"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sanjeevani_app")

# ------------------ APP ------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# ------------------ DATABASE ------------------
def get_db():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="flask_user",
            password="Flask@232",
            database="sanjeevani_ai",
            auth_plugin="mysql_native_password"
        )
        return conn
    except Error as e:
        logger.error(f"DB connection failed: {e}")
        return None

db = get_db()
cursor = db.cursor(dictionary=True) if db else None

# ------------------ LOAD CNN MODEL ------------------
model = None
_model_has_rescaling = False

def load_dl_model():
    global model, _model_has_rescaling

    if not os.path.exists(MODEL_PATH):
        logger.error(f"CNN model missing: {MODEL_PATH}")
        return

    try:
        model = load_model(MODEL_PATH)
        logger.info("CNN model loaded successfully.")

        # Detect rescaling layer
        for layer in model.layers:
            if layer.__class__.__name__.lower() == "rescaling":
                _model_has_rescaling = True

    except Exception as e:
        logger.exception(f"Unable to load CNN model: {e}")
        model = None

# ------------------ HELPERS ------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Invalid or unreadable image file")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE)
    img = img.astype(np.float32)

    if not _model_has_rescaling:
        img = img / 255.0

    return np.expand_dims(img, axis=0)

def get_pesticide_recommendations(disease_name):
    """
    Fetch pesticide recommendations based on disease name
    Returns list of recommendations or empty list if none found
    """
    if not cursor:
        logger.error("Database connection not available")
        return []

    try:
        # DEBUG: Log what we're searching for
        logger.info(f"🔍 Searching for pesticide recommendations for disease: '{disease_name}'")
        
        # First, let's see what diseases are available in pesticides_recomendation_3
        cursor.execute("SELECT DISTINCT Disease_Name FROM pesticides_recomendation_3")
        available_diseases = [row['Disease_Name'] for row in cursor.fetchall()]
        logger.info(f"📋 Available diseases in pesticides_recomendation_3: {available_diseases}")
        
        # Try exact match first
        cursor.execute(
            "SELECT * FROM pesticides_recomendation_3 WHERE Disease_Name = %s",
            (disease_name,)
        )
        exact_matches = cursor.fetchall()
        
        if exact_matches:
            logger.info(f"✅ Found {len(exact_matches)} exact matches for '{disease_name}'")
            return exact_matches
        
        # If no exact match, try case-insensitive search
        cursor.execute(
            "SELECT * FROM pesticides_recomendation_3 WHERE LOWER(Disease_Name) = LOWER(%s)",
            (disease_name,)
        )
        case_insensitive_matches = cursor.fetchall()
        
        if case_insensitive_matches:
            logger.info(f"✅ Found {len(case_insensitive_matches)} case-insensitive matches for '{disease_name}'")
            return case_insensitive_matches
        
        # If still no matches, try partial matching
        cursor.execute(
            "SELECT * FROM pesticides_recomendation_3 WHERE Disease_Name LIKE %s",
            (f"%{disease_name}%",)
        )
        partial_matches = cursor.fetchall()
        
        if partial_matches:
            logger.info(f"✅ Found {len(partial_matches)} partial matches for '{disease_name}'")
            return partial_matches
            
        logger.warning(f"❌ No pesticide recommendations found for disease: '{disease_name}'")
        return []
        
    except Error as e:
        logger.error(f"Error fetching pesticide recommendations: {e}")
        return []

# ------------------ ERROR HANDLER ------------------
@app.errorhandler(413)
def too_large(e):
    return "File too large. Max 5MB allowed.", 413

# ------------------ ROUTES ------------------
@app.route("/")
def login_page():
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if not cursor:
            return "Database unavailable", 500

        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        if cursor.fetchone():
            return render_template("signup.html", error="Username already exists")

        cursor.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            (username, password)
        )
        db.commit()
        return redirect(url_for("login_page"))

    return render_template("signup.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form["username"]
    password = request.form["password"]

    if not cursor:
        return "Database unavailable", 500

    cursor.execute(
        "SELECT * FROM users WHERE username=%s AND password=%s",
        (username, password)
    )
    user = cursor.fetchone()

    if user:
        return redirect(url_for("index"))
    return render_template("login.html", error="Invalid credentials")

@app.route("/index")
def index():
    # default render without results context
    return render_template("index.html")

# ------------------ RESULTS PAGE ------------------
@app.route("/results/<int:diagnosis_id>")
def results(diagnosis_id):
    """
    Fetch disease_name from plant_diagnosis and latest weather_prediction_results for this diagnosis_id.
    Modified: fetch all 5-day predictions and send GOOD days to template.
    """
    if not cursor:
        return "Database unavailable", 500

    # Fetch disease_name
    cursor.execute("SELECT disease_name FROM plant_diagnosis WHERE id=%s", (diagnosis_id,))
    diag_row = cursor.fetchone()
    disease_name = diag_row['disease_name'] if diag_row and 'disease_name' in diag_row else None

    # DEBUG: Log the detected disease
    logger.info(f"🎯 Detected disease from plant_diagnosis: '{disease_name}'")

    # Fetch pesticide recommendations based on disease_name
    pesticide_recommendations = []
    if disease_name and disease_name != "Inference Error" and disease_name != "Pending":
        pesticide_recommendations = get_pesticide_recommendations(disease_name)

    # Fetch all prediction rows for this diagnosis_id (we added day_index in ml_integration)
    cursor.execute(
        "SELECT id, diagnosis_id, day_index, prediction, confidence, features_json, created_at "
        "FROM weather_prediction_results WHERE diagnosis_id=%s ORDER BY day_index ASC",
        (diagnosis_id,)
    )
    rows = cursor.fetchall() or []

    # Build day entries and filter GOOD days
    all_days = []
    good_days = []
    for r in rows:
        # parse features_json (string or dict)
        features = {}
        if r.get('features_json'):
            try:
                if isinstance(r['features_json'], str):
                    features = json.loads(r['features_json'])
                else:
                    features = r['features_json']
            except Exception:
                features = {}

        day_obj = {
            "id": r.get("id"),
            "diagnosis_id": r.get("diagnosis_id"),
            "day_index": r.get("day_index"),
            "prediction": r.get("prediction"),
            "confidence": r.get("confidence"),
            "features": features,
            "created_at": r.get("created_at")
        }
        all_days.append(day_obj)
        if str(day_obj["prediction"]).upper() == "GOOD":
            good_days.append(day_obj)

    # Keep old single-latest fields for backward compatibility (optional)
    # We'll set them from the first good day if exists else None
    disease_confidence = None
    temperature = humidity = rainfall = wind_speed = None
    if all_days:
        first = all_days[0]
        disease_confidence = first.get("confidence")
        temperature = first["features"].get("temperature") if first.get("features") else None
        humidity = first["features"].get("humidity") if first.get("features") else None
        rainfall = first["features"].get("rainfall") if first.get("features") else None
        wind_speed = first["features"].get("wind_speed") if first.get("features") else None

    # DEBUG: Log what we're sending to template
    logger.info(f"📤 Sending to template - Disease: '{disease_name}', Pesticide recommendations: {len(pesticide_recommendations)}")

    # Render index.html with extra context (front-end will use these)
    return render_template(
        "index.html",
        diagnosis_id=diagnosis_id,
        disease_name=disease_name,
        disease_confidence=disease_confidence,
        temperature=temperature,
        humidity=humidity,
        rainfall=rainfall,
        wind_speed=wind_speed,
        all_days=all_days,
        good_days=good_days,
        pesticide_recommendations=pesticide_recommendations
    )

# ------------------ MAIN IMAGE ANALYSIS ------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image received"}), 400

    image = request.files["image"]

    if image.filename == "":
        return jsonify({"error": "Image not selected"}), 400

    if not allowed_file(image.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image.save(filepath)

    if not cursor:
        # clean up saved file on DB failure
        try:
            os.remove(filepath)
        except Exception:
            pass
        return "Database unavailable", 500

    # Save in DB
    cursor.execute(
        "INSERT INTO plant_diagnosis (username, image_path, created_at, disease_name) "
        "VALUES (%s,%s,%s,%s)",
        ("test_user", filepath, datetime.now(), "Pending")
    )
    db.commit()
    diagnosis_id = cursor.lastrowid

    # CNN Prediction
    try:
        processed = preprocess_image(filepath)
        preds = model.predict(processed)
        index = int(np.argmax(preds[0]))
        disease = CLASS_NAMES[index]
    except Exception as e:
        logger.exception("CNN inference failed")
        disease = "Inference Error"

    cursor.execute(
        "UPDATE plant_diagnosis SET disease_name=%s WHERE id=%s",
        (disease, diagnosis_id)
    )
    db.commit()

    # WEATHER ML PREDICTION (this will create rows in weather_prediction_results for 5 days)
    try:
        # run_weather_prediction will internally connect to DB and store results table (5 rows)
        run_weather_prediction(diagnosis_id)
    except Exception as e:
        logger.exception(f"Weather ML failed for diagnosis_id={diagnosis_id}: {e}")

    # Redirect to results page so user can see disease + weather details
    return redirect(url_for("results", diagnosis_id=diagnosis_id))

# ------------------ WEATHER API (stores daily rows) ------------------
@app.route("/get_weather")
def get_weather():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not lat or not lon:
        return jsonify({"error": "Location missing"}), 400

    url = (
        f"https://api.openweathermap.org/data/2.5/forecast?"
        f"lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    )

    resp = requests.get(url)
    if resp.status_code != 200:
        return jsonify({"error": "Weather API failed", "status_code": resp.status_code}), 500

    resp_json = resp.json()
    if "list" not in resp_json:
        return jsonify({"error": "Weather API failed"}), 500

    # We will pick one representative daily entry per day (12:00) — up to 5 days
    forecast_list = resp_json["list"]
    daily_entries = []
    added_dates = set()

    for item in forecast_list:
        dt_txt = item.get("dt_txt")
        if not dt_txt:
            continue
        date_str, time_str = dt_txt.split(" ")
        # choose 12:00:00 entries (midday) to represent the day
        if time_str == "12:00:00" and date_str not in added_dates:
            daily_entries.append(item)
            added_dates.add(date_str)
        if len(daily_entries) == 5:
            break

    if not daily_entries:
        # fallback: take first 5 entries if 12:00 entries not found
        for item in forecast_list[:5]:
            daily_entries.append(item)

    # ALWAYS INSERT 5 (or available) rows with consistent columns used elsewhere (_date, slot, temp, humidity, wind, rain, location)
    try:
        for item in daily_entries:
            dt = item.get("dt_txt", "")
            date, time = (dt.split(" ") + [""])[:2]
            hour = 0
            try:
                hour = int(time.split(":")[0]) if time else 0
            except:
                hour = 0
            slot = f"{hour:02d}:00-{(hour+3):02d}:00"

            temp = item.get("main", {}).get("temp")
            humidity = item.get("main", {}).get("humidity")
            wind_speed = item.get("wind", {}).get("speed")
            rain_val = item.get("rain", {}).get("3h", 0.0) if item.get("rain") else 0.0

            cursor.execute(
                """
                INSERT INTO weather_data (_date, slot, temp, humidity, wind, rain, location)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    date,
                    slot,
                    temp,
                    humidity,
                    wind_speed,
                    rain_val,
                    round(float(lat), 6),
                )
            )

        db.commit()
    except Error as e:
        db.rollback()
        logger.exception(f"Failed storing weather rows: {e}")
        return jsonify({"error": "DB error storing weather"}), 500

    return jsonify({"message": "Weather stored", "days_inserted": len(daily_entries)})

# ------------------ DEBUG ROUTE ------------------
@app.route("/debug_pesticides")
def debug_pesticides():
    """
    Debug route to check what's in the pesticides_recomendation_3 table
    """
    if not cursor:
        return "Database unavailable", 500
    
    try:
        # Get all diseases from pesticides_recomendation_3
        cursor.execute("SELECT DISTINCT Disease_Name FROM pesticides_recomendation_3")
        diseases = [row['Disease_Name'] for row in cursor.fetchall()]
        
        # Get sample data
        cursor.execute("SELECT * FROM pesticides_recomendation_3 LIMIT 5")
        sample_data = cursor.fetchall()
        
        return jsonify({
            "available_diseases": diseases,
            "sample_data": sample_data,
            "total_diseases": len(diseases)
        })
        
    except Error as e:
        return jsonify({"error": str(e)}), 500

# ------------------ MAIN ------------------
if __name__ == "__main__":
    load_dl_model()
    app.run(debug=True)
