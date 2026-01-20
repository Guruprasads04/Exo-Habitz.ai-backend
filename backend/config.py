import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "database", "exoplanets.db")
MODELS_DIR = os.path.join(BASE_DIR, "model")

DEBUG = True

# 🔒 Fixed model input schema
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
