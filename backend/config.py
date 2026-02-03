import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEBUG = True

# 🔐 Supabase PostgreSQL URL
DATABASE_URL = os.getenv("DATABASE_URL")

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
