from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import numpy as np
from sqlalchemy import create_engine, text

# -------------------------------------------------
# FLASK SETUP
# -------------------------------------------------

app = Flask(__name__)
CORS(app)
app.config["DEBUG"] = True

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

MODEL_FEATURES = [
    "st_teff",
    "st_rad",
    "st_mass",
    "st_met",
    "st_luminosity",
    "pl_orbper",
    "pl_orbeccen",
    "pl_insol"
]

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("❌ DATABASE_URL not set")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "model")

reg_model = joblib.load(os.path.join(MODELS_DIR, "xgboost_reg.pkl"))
cls_model = joblib.load(os.path.join(MODELS_DIR, "xgboost_classifier.pkl"))

# -------------------------------------------------
# INIT DB (SUPABASE)
# -------------------------------------------------

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS planets (
            id SERIAL PRIMARY KEY,
            planet_name TEXT UNIQUE,
            st_teff FLOAT,
            st_rad FLOAT,
            st_mass FLOAT,
            st_met FLOAT,
            st_luminosity FLOAT,
            pl_orbper FLOAT,
            pl_orbeccen FLOAT,
            pl_insol FLOAT,
            source TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """))

init_db()

# -------------------------------------------------
# RESPONSE HELPER
# -------------------------------------------------

def response(status, message, data=None):
    return jsonify({
        "status": status,
        "message": message,
        "data": data
    })

# -------------------------------------------------
# ROUTES
# -------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/", methods=["GET"])
def home():
    return response(
        "success",
        "Exoplanet Habitability Prediction API",
        {
            "required_features": MODEL_FEATURES,
            "endpoints": ["/add_planet", "/predict", "/rank"]
        }
    )

# ---------------- ADD PLANET ----------------

@app.route("/add_planet", methods=["POST"])
def add_planet():
    data = request.get_json()
    planet_name = data.get("planet_name", "Unknown")

    try:
        row = {
            "planet_name": planet_name,
            **{f: data[f] for f in MODEL_FEATURES},
            "source": "user"
        }

        df = pd.DataFrame([row])

        df.to_sql(
            "planets",
            engine,
            if_exists="append",
            index=False
        )

        return response(
            "success",
            "Planet added successfully",
            {"planet_saved": True}
        )

    except Exception as e:
        return response("error", str(e)), 400

# ---------------- PREDICT ----------------

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    planet_name = data.get("planet_name", "Unknown")

    try:
        input_df = pd.DataFrame([data])[MODEL_FEATURES]

        proba = float(cls_model.predict_proba(input_df)[0][1])
        probax = np.clip(proba - 0.1225, 0.0, 1.0)
        habitability = int(proba >= 0.5)

        # Insert (idempotent)
        row = {
            "planet_name": planet_name,
            **{f: data[f] for f in MODEL_FEATURES},
            "source": "prediction"
        }

        pd.DataFrame([row]).to_sql(
            "planets",
            engine,
            if_exists="append",
            index=False,
            method="multi"
        )

        return response(
            "success",
            "Prediction generated and saved",
            {
                "habitability": habitability,
                "habitability_score": round(probax, 4),
                "confidence": round(proba, 4),
                "planet_saved": True
            }
        )

    except Exception as e:
        return response("error", str(e)), 400

# ---------------- RANK ----------------

@app.route("/rank", methods=["GET"])
def rank():
    top_n = request.args.get("top", type=int)

    df = pd.read_sql("SELECT * FROM planets", engine)

    if df.empty:
        return response(
            "success",
            "No planets available",
            {
                "total_count": 0,
                "habitable_count": 0,
                "average_score": 0,
                "data": []
            }
        )

    X = df[MODEL_FEATURES]

    proba = cls_model.predict_proba(X)[:, 1]
    probax = np.clip(proba - 0.1225, 0.0, 1.0)

    df["habitability_score"] = probax.round(4)
    df["confidence"] = proba.round(4)
    df["habitability"] = (proba >= 0.5).astype(int)

    ranked = (
        df[["planet_name", "habitability", "habitability_score", "confidence"]]
        .sort_values("habitability_score", ascending=False)
        .drop_duplicates(subset=["planet_name"])
        .reset_index(drop=True)
    )

    ranked["rank"] = ranked.index + 1

    if top_n:
        ranked = ranked.head(top_n)

    return response(
        "success",
        "Ranking generated",
        {
            "total_count": len(df),
            "habitable_count": int(df["habitability"].sum()),
            "average_score": float(df["habitability_score"].mean()),
            "data": ranked.to_dict("records")
        }
    )

# -------------------------------------------------
# RUN
# -------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
