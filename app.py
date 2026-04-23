import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify, render_template
from torchvision import transforms, models
from groq import Groq
import hashlib #db part
import secrets
import mysql.connector
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from functools import wraps


GROQ_API_KEY = "ADD YOUR GROQ_API_KEY HERE"
groq_client = Groq(api_key=GROQ_API_KEY)

ADMIN_EMAIL = "admin@cocoaguard.com" #admin
ADMIN_PASSWORD = "admin123"
ADMIN_NAME = "Admin"

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "cocoa_app"
}


BUNDLE_PATH = "cocoa_model_bundle.pt"

bundle = torch.load(BUNDLE_PATH, map_location="cpu")
class_names = bundle["class_names"]
thresholds = bundle.get("severity_thresholds", {"severe": 0.85, "moderate": 0.65})


model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(class_names))
model.load_state_dict(bundle["model_state"])
model.eval()


infer_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def severity_from_pred_and_conf(pred_class: str, conf: float) -> str:
    if pred_class.upper() == "HEALTHY":
        return "NONE"
    if conf >= thresholds["severe"]:
        return "SEVERE"
    if conf >= thresholds["moderate"]:
        return "MODERATE"
    return "MILD"


def get_rule_based_advice(prediction: str, severity: str) -> str:
    pred = prediction.upper()

    if pred == "HEALTHY":
        return "Pod looks healthy. Keep monitoring weekly, maintain good pruning, and avoid waterlogging."

    if pred == "BLACKPOD":
        base = "Black Pod: Remove infected pods immediately and dispose/bury them. Improve drainage and airflow."
        if severity == "SEVERE":
            return base + " Apply a copper-based fungicide promptly and repeat as recommended."
        if severity == "MODERATE":
            return base + " Consider preventive fungicide spray if weather is wet/humid."
        return base + " Monitor closely and remove any new infected pods early."

    if pred == "FROSTYPOD":
        base = "Frosty Pod: Remove and destroy infected pods early to stop spore spread. Keep the area clean and reduce humidity."
        if severity == "SEVERE":
            return base + " Increase sanitation frequency and consult extension guidance for local control measures."
        if severity == "MODERATE":
            return base + " Focus on sanitation and avoid leaving pods on trees once infected."
        return base + " Monitor nearby pods and remove any that show early symptoms."

    if pred == "MIRID":
        base = "Mirid (capsid) damage: Inspect pods and stems for pests. Remove heavily damaged pods."
        if severity == "SEVERE":
            return base + " Use approved insect control methods per local agricultural guidance."
        if severity == "MODERATE":
            return base + " Consider targeted control and increase monitoring frequency."
        return base + " Monitor and manage shade/pruning to reduce pest habitat."

    return "No advice available for this class."


def get_llm_advice(prediction: str, severity: str, confidence: float) -> str:
    try:
        prompt = f"""You are an agricultural expert advising cocoa farmers.

A cocoa pod has been analysed by an AI disease detection system with the following result:
- Disease detected: {prediction}
- Severity level: {severity}
- Model confidence: {round(confidence * 100, 1)}%

Please provide clear, simple, and practical advice for a cocoa farmer with no technical background.
Your response should include:
1. A brief explanation of what this disease/condition means
2. Immediate actions the farmer should take
3. Preventive measures to avoid further spread

Keep your response concise (under 150 words), friendly, and easy to understand.
Do not use technical jargon. Write as if speaking directly to the farmer."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Groq API error: {type(e).__name__}: {e}")
        return get_rule_based_advice(prediction, severity)



app = Flask(__name__)

app.secret_key = secrets.token_hex(24) #db part


def get_db():
    return mysql.connector.connect(**DB_CONFIG)

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(150) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    cursor.close()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_user_by_email(email):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

def create_user(name, email, password):
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s)',
            (name, email, hash_password(password))
        )
        conn.commit()
        return True
    except mysql.connector.IntegrityError:
        return False
    finally:
        cursor.close()
        conn.close()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('home') + '?popup=login')  # ← fixed
        return f(*args, **kwargs)
    return decorated

def admin_required(f): 
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('is_admin'):
            flash('Access denied.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated



@app.get("/health")
def health():
    return {"status": "ok", "classes": class_names}

@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file field named 'image'"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400


    x = infer_tfms(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)
        conf, idx = torch.max(probs, dim=0)

    pred = class_names[int(idx)]
    conf = float(conf)
    sev = severity_from_pred_and_conf(pred, conf)

    advice = get_llm_advice(pred, sev, conf)

    try:
        log_conn = get_db()
        log_cursor = log_conn.cursor()
        log_cursor.execute(
            'INSERT INTO predictions (user_id, prediction, confidence, severity) VALUES (%s, %s, %s, %s)',
            (session.get('user_id', 0), pred, round(conf, 4), sev)
        )
        log_conn.commit()
        log_cursor.close()
        log_conn.close()
    except Exception as e:
        print(f"Logging error: {e}") #finish log

    

    return jsonify({
        "prediction": pred,
        "confidence": round(conf, 4),
        "severity": sev,
        "advice": advice
    })

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/detect")
@login_required
def detect():
    return render_template("detect.html")



@app.post("/register")
def register_post():
    name     = request.form.get("name", "").strip()
    email    = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    confirm  = request.form.get("confirm_password", "")

    if not name or not email or not password:
        flash("All fields are required.", "register_error")
        return redirect(url_for("home") + "?popup=register")
    if len(password) < 6:
        flash("Password must be at least 6 characters.", "register_error")
        return redirect(url_for("home") + "?popup=register")
    if password != confirm:
        flash("Passwords do not match.", "register_error")
        return redirect(url_for("home") + "?popup=register")
    if not create_user(name, email, password):
        flash("Email already exists.", "register_error")
        return redirect(url_for("home") + "?popup=register")

    flash("Account created! Please log in.", "login_success")
    return redirect(url_for("home") + "?popup=login")



@app.post("/login")
def login_post():
    email    = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")


    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        session["user_id"]   = 0
        session["user_name"] = ADMIN_NAME
        session["is_admin"]  = True
        return redirect(url_for("admin"))

    user = get_user_by_email(email)
    if not user or user["password_hash"] != hash_password(password):
        flash("Invalid email or password.", "login_error")
        return redirect(url_for("home") + "?popup=login")
    
  
    session["user_id"]   = user["id"]
    session["user_name"] = user["name"]
    session["user_email"] = user["email"]
    session["is_admin"]  = False 
    return redirect(url_for("dashboard"))


@app.get("/admin")
@admin_required
def admin():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute('SELECT id, name, email, created_at FROM users')
    users = cursor.fetchall()

    cursor.execute('SELECT COUNT(*) as total FROM predictions')
    total_predictions = cursor.fetchone()['total']

    cursor.execute('SELECT prediction, COUNT(*) as count FROM predictions GROUP BY prediction')
    disease_stats = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template("admin.html",
        users=users,
        total_predictions=total_predictions,
        disease_stats=disease_stats,
        total_users=len(users)
    )

@app.post("/admin/delete_user/<int:user_id>")
@admin_required
def delete_user(user_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE id = %s', (user_id,))
    conn.commit()
    cursor.close()
    conn.close()
    flash('User deleted.', 'success')
    return redirect(url_for('admin'))

@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("home") + "?popup=login")  # ← was url_for("login")

@app.get("/dashboard")
@login_required
def dashboard():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute('SELECT COUNT(*) as total FROM predictions WHERE user_id = %s', (session['user_id'],))
    total_scans = cursor.fetchone()['total']

    cursor.execute('''
        SELECT id, prediction, confidence, severity, created_at 
        FROM predictions 
        WHERE user_id = %s 
        ORDER BY created_at DESC 
        LIMIT 10
    ''', (session['user_id'],))
    history = cursor.fetchall()

    # Convert datetime and Decimal for JSON serialization
    for record in history:
        if record.get('created_at'):
            record['created_at'] = record['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        if record.get('confidence') is not None:
            record['confidence'] = float(record['confidence'])

    cursor.close()
    conn.close()

    return render_template("dashboard.html",
        total_scans=total_scans,
        history=history,
        user_name=session['user_name'],
        user_email=session['user_email']
    )
@app.post("/delete_prediction/<int:pred_id>")
@login_required
def delete_prediction(pred_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions WHERE id = %s AND user_id = %s', (pred_id, session['user_id']))
    conn.commit()
    cursor.close()
    conn.close()
    flash('Prediction deleted.', 'success')
    return redirect(url_for('dashboard'))

@app.post("/clear_history")
@login_required
def clear_history():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions WHERE user_id = %s', (session['user_id'],))
    conn.commit()
    cursor.close()
    conn.close()
    flash('All history cleared.', 'success')
    return redirect(url_for('dashboard'))

if __name__ == "__main__":
    init_db() #db part
    print("Database ready") #db finish
    app.run(host="127.0.0.1", port=5000, debug=True)