"""Leakage-safe feature engineering shared by training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from modeling.config import CATEGORICAL_FEATURES, FEATURES


def make_feature_frame(listings: pd.DataFrame) -> pd.DataFrame:
    """Return model-ready features without reading the target column."""
    required = {
        "area",
        "kitchen_area",
        "rooms",
        "floor",
        "floors",
        "geo_lat",
        "geo_lon",
        "region",
        "building_type",
        "object_type",
    }
    missing = sorted(required.difference(listings.columns))
    if missing:
        raise ValueError(f"Listings are missing model input columns: {missing}")

    features = pd.DataFrame(index=listings.index)
    for column in ("area", "kitchen_area", "rooms", "floor", "floors", "geo_lat", "geo_lon"):
        features[column] = pd.to_numeric(listings[column], errors="coerce")

    if "published_at" in listings:
        published_at = pd.to_datetime(listings["published_at"], errors="coerce")
    elif {"date", "time"}.issubset(listings.columns):
        published_at = pd.to_datetime(
            listings["date"].astype("string") + " " + listings["time"].astype("string"),
            errors="coerce",
        )
    else:
        published_at = pd.Series(pd.NaT, index=features.index, dtype="datetime64[ns]")

    month = published_at.dt.month.astype(float)
    features["publication_year"] = published_at.dt.year.astype(float)
    features["publication_month_sin"] = np.sin(2 * np.pi * month / 12)
    features["publication_month_cos"] = np.cos(2 * np.pi * month / 12)

    rooms = features["rooms"]
    area = features["area"]
    kitchen_area = features["kitchen_area"]
    floor = features["floor"]
    floors = features["floors"]

    features["is_studio"] = rooms.eq(-1).astype("int8")
    features["rooms_count"] = rooms.clip(lower=0)
    features["area_per_room"] = area.div(features["rooms_count"].replace(0, 1))
    features["kitchen_area_share"] = kitchen_area.div(area.where(area.gt(0)))
    features["kitchen_area_missing"] = kitchen_area.isna().astype("int8")
    features["floor_ratio"] = floor.div(floors.where(floors.gt(0)))
    features["is_first_floor"] = floor.eq(1).astype("int8")
    features["is_last_floor"] = floor.eq(floors).astype("int8")

    for column in CATEGORICAL_FEATURES:
        features[column] = (
            listings[column]
            .astype("string")
            .fillna("__MISSING__")
            .replace("", "__MISSING__")
            .astype(str)
        )

    return features.loc[:, list(FEATURES)]
