"""
Centralized, absolute paths and config helpers for the CLI app.

Everything is resolved relative to this file so the app works regardless of the
current working directory (the older scripts used fragile "../" relative paths).
"""

import os
import json

# Directory that contains this file (main_prediction_system/).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Master training dataset (44-column enriched fight-level CSV).
DATA_PATH = os.path.join(BASE_DIR, "mayberealdata_enriched.csv")

# Trained-model artifacts (model weights, preprocessor, feature cols, config).
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

# Scratch space for cached scrapes and pre-retrain backups.
CACHE_DIR = os.path.join(BASE_DIR, "cache")
BACKUP_DIR = os.path.join(BASE_DIR, "backups")

# Artifact file paths.
MODEL_PATH = os.path.join(ARTIFACT_DIR, "ufc_nn_model.pt")
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.joblib")
FEATURE_COLS_PATH = os.path.join(ARTIFACT_DIR, "feature_cols.json")
CONFIG_PATH = os.path.join(ARTIFACT_DIR, "config.json")

# Cached upcoming-card scrape.
UPCOMING_CACHE_CSV = os.path.join(CACHE_DIR, "upcoming_cards.csv")
UPCOMING_CACHE_META = os.path.join(CACHE_DIR, "upcoming_meta.json")

# Default freshness window for the upcoming-card cache (hours).
UPCOMING_CACHE_TTL_HOURS = 12

# Defaults used when artifacts/config.json is unavailable (matches train.py).
DEFAULT_PREPROCESSING = {
    "base_elo": 1500,
    "elo_k": 16,
    "rolling_window": 5,
}


def ensure_dirs():
    """Create the cache/backup directories if they do not exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)


def load_training_config():
    """
    Return the preprocessing config saved at train time, falling back to
    DEFAULT_PREPROCESSING when artifacts/config.json is missing.
    """
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        return {
            "base_elo": cfg.get("base_elo", DEFAULT_PREPROCESSING["base_elo"]),
            "elo_k": cfg.get("elo_k", DEFAULT_PREPROCESSING["elo_k"]),
            "rolling_window": cfg.get(
                "rolling_window", DEFAULT_PREPROCESSING["rolling_window"]
            ),
        }
    return dict(DEFAULT_PREPROCESSING)
