"""Shared paths and feature definitions for the training and inference code."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "real_estate_raw.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "real_estate_clean.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "price"

NUMERIC_FEATURES = (
    "area",
    "kitchen_area",
    "kitchen_area_share",
    "rooms_count",
    "area_per_room",
    "floor",
    "floors",
    "floor_ratio",
    "is_first_floor",
    "is_last_floor",
    "is_studio",
    "kitchen_area_missing",
    "geo_lat",
    "geo_lon",
    "publication_year",
    "publication_month_sin",
    "publication_month_cos",
)

CATEGORICAL_FEATURES = ("region", "building_type", "object_type")
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
VALIDATION_FRACTION = 0.15
HOLDOUT_FRACTION = 0.15
ADVISOR_THRESHOLD_PERCENT = 15.0
