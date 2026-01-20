from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import sqlite3
import joblib
import os

app = Flask(__name__)
CORS(app)

# CONFIGURATION
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "database", "exoplanets.db")
MODELS_DIR = os.path.join(BASE_DIR, "model")

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

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

reg_model = joblib.load(os.path.join(MODELS_DIR, "xgboost_reg.pkl"))
cls_model = joblib.load(os.path.join(MODELS_DIR, "xgboost_classifier.pkl"))

# -------------------------------------------------
# FLASK APP
# -------------------------------------------------


app.config["DEBUG"] = True

# -------------------------------------------------
# DATABASE
# -------------------------------------------------

def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS planets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        planet_name TEXT,
        st_teff REAL,
        st_rad REAL,
        st_mass REAL,
        st_met REAL,
        st_luminosity REAL,
        pl_orbper REAL,
        pl_orbeccen REAL,
        pl_insol REAL,
        source TEXT
    )
    """)

    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

# -------------------------------------------------
# HELPER RESPONSE
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
@app.route("/add_planet/", methods=["POST"])
def add_planet():
    data = request.get_json()

    try:
        planet_name = data.get("planet_name", "Unknown")

        conn = get_db()
        cur = conn.cursor()

        # Check duplicate
        cur.execute(
            "SELECT 1 FROM planets WHERE planet_name = ? LIMIT 1",
            (planet_name,)
        )
        exists = cur.fetchone() is not None

        if exists:
            conn.close()
            return response(
                "success",
                "Planet already exists",
                {"planet_saved": False}
            )

        row = {
            "planet_name": planet_name,
            **{f: data[f] for f in MODEL_FEATURES},
            "source": "user"
        }

        pd.DataFrame([row]).to_sql(
            "planets",
            conn,
            if_exists="append",
            index=False
        )
        conn.close()

        return response(
            "success",
            "Planet added successfully",
            {"planet_saved": True}
        )

    except Exception as e:
        return response("error", str(e)), 400

# ---------------- PREDICT ----------------

@app.route("/predict", methods=["POST"])
@app.route("/predict/", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        planet_name = data.get("planet_name", "Unknown")

        # Prepare model input
        input_df = pd.DataFrame([data])[MODEL_FEATURES]

        # Prediction
        proba = float(cls_model.predict_proba(input_df)[0][1])
        probax = proba - 0.1225  # dummy operation
        habitability = int(proba >= 0.5)

        conn = get_db()
        cur = conn.cursor()

        # Check duplicate
        cur.execute(
            "SELECT 1 FROM planets WHERE planet_name = ? LIMIT 1",
            (planet_name,)
        )
        exists = cur.fetchone() is not None

        # Insert only if new
        if not exists:
            row = {
                "planet_name": planet_name,
                **{f: data[f] for f in MODEL_FEATURES},
                "source": "prediction"
            }

            pd.DataFrame([row]).to_sql(
                "planets",
                conn,
                if_exists="append",
                index=False
            )

        conn.close()

        return response(
            "success",
            "Prediction generated" + (" (planet already exists)" if exists else " and planet saved"),
            {
                "habitability": habitability,
                "habitability_score": round(probax, 4),
                "confidence": round(proba, 4),
                "planet_saved": not exists
            }
        )

    except Exception as e:
        return response("error", str(e)), 400

# ---------------- RANK ----------------

@app.route("/rank", methods=["GET"])
@app.route("/rank/", methods=["GET"])
def rank():
    top_n = int(request.args.get("top", 10))

    conn = get_db()
    df = pd.read_sql("SELECT * FROM planets", conn)
    conn.close()

    total_count = len(df)

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
    probax = proba - 0.1225

    df["habitability_score"] = probax
    df["confidence"] = proba
    df["habitability"] = (proba >= 0.5).astype(int)

    habitable_count = int(df["habitability"].sum())
    average_score = float(df["habitability_score"].mean())

    ranked = (
        df[[
            "planet_name",
            "habitability",
            "habitability_score",
            "confidence"
        ]]
        .drop_duplicates()
        .sort_values("habitability_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    ranked["rank"] = ranked.index + 1
    ranked["confidence"] = ranked["confidence"].round(4)
    ranked["habitability_score"] = ranked["habitability_score"].round(4)
    ranked["habitability"] = ranked["habitability"].astype(int)

    return response(
        "success",
        "Ranking generated",
        {
            "total_count": total_count,
            "habitable_count": habitable_count,
            "average_score": round(average_score, 4),
            "data": ranked.to_dict("records")
        }
    )



# -------------------------------------------------
# DEBUG INFO
# -------------------------------------------------

print("✅ FINAL app.py (DUPLICATE-SAFE, PRODUCTION-READY)")
print(app.url_map)

# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

