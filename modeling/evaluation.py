"""Chronological partitioning and regression diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from modeling.config import HOLDOUT_FRACTION, TARGET, VALIDATION_FRACTION


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    holdout: pd.DataFrame
    summary: dict[str, object]


def chronological_split(
    listings: pd.DataFrame,
    validation_fraction: float = VALIDATION_FRACTION,
    holdout_fraction: float = HOLDOUT_FRACTION,
) -> TemporalSplit:
    """Split at calendar-month boundaries so months cannot straddle partitions."""
    if validation_fraction <= 0 or holdout_fraction <= 0:
        raise ValueError("Validation and holdout fractions must both be positive")
    if validation_fraction + holdout_fraction >= 0.8:
        raise ValueError("Validation and holdout fractions leave too little training data")
    if "published_at" not in listings:
        raise ValueError("A published_at timestamp is required for chronological validation")

    columns = ["published_at", TARGET]
    columns.extend(
        column
        for column in ("region", "object_type", "building_type")
        if column in listings.columns
    )
    frame = listings.loc[:, columns].reset_index(drop=True)
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce")
    rows_without_timestamp = int(frame["published_at"].isna().sum())
    frame = frame.dropna(subset=["published_at", TARGET])
    frame = frame.loc[frame[TARGET].gt(0)].sort_values("published_at")
    if frame.empty:
        raise ValueError("No positive-price listings with a valid publication timestamp remain")

    frame["_split_month"] = frame["published_at"].dt.to_period("M")
    months = pd.PeriodIndex(frame["_split_month"].unique()).sort_values()
    if len(months) < 6:
        raise ValueError(
            "At least six distinct listing months are required for a train/validation/holdout split"
        )

    holdout_month_count = max(1, int(np.ceil(len(months) * holdout_fraction)))
    validation_month_count = max(1, int(np.ceil(len(months) * validation_fraction)))
    train_month_count = len(months) - holdout_month_count - validation_month_count
    if train_month_count < 1:
        raise ValueError("Not enough distinct listing months for the requested split fractions")

    validation_start = months[train_month_count]
    holdout_start = months[train_month_count + validation_month_count]
    train = frame.loc[frame["_split_month"] < validation_start].copy()
    validation = frame.loc[
        frame["_split_month"].ge(validation_start)
        & frame["_split_month"].lt(holdout_start)
    ].copy()
    holdout = frame.loc[frame["_split_month"].ge(holdout_start)].copy()

    for partition in (train, validation, holdout):
        partition.drop(columns="_split_month", inplace=True)

    if min(len(train), len(validation), len(holdout)) == 0:
        raise ValueError("The chronological split produced an empty partition")

    summary = {
        "split_type": "calendar_month_chronological",
        "rows_without_timestamp": rows_without_timestamp,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "holdout_rows": len(holdout),
        "train_start": train["published_at"].min().isoformat(),
        "train_end": train["published_at"].max().isoformat(),
        "validation_start": validation["published_at"].min().isoformat(),
        "validation_end": validation["published_at"].max().isoformat(),
        "holdout_start": holdout["published_at"].min().isoformat(),
        "holdout_end": holdout["published_at"].max().isoformat(),
    }
    return TemporalSplit(train=train, validation=validation, holdout=holdout, summary=summary)


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted arrays must have the same shape")
    if actual.size == 0 or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require non-empty, finite actual and predicted prices")
    if np.any(actual <= 0):
        raise ValueError("Percentage metrics require strictly positive actual prices")

    absolute_error = np.abs(actual - predicted)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "MAPE": float(np.mean(absolute_error / actual) * 100),
        "MdAPE": float(np.median(absolute_error / actual) * 100),
        "WAPE": float(absolute_error.sum() / actual.sum() * 100),
    }


def segment_metrics(
    holdout: pd.DataFrame,
    predictions: np.ndarray,
    minimum_rows: int = 30,
    top_groups: int = 25,
) -> pd.DataFrame:
    """Summarize holdout error by major categorical segments."""
    rows: list[dict[str, object]] = []
    for column in ("region", "object_type", "building_type"):
        if column not in holdout:
            continue
        values = holdout[column].astype("string").fillna("__MISSING__")
        counts = values.value_counts().head(top_groups)
        for value, count in counts.items():
            if count < minimum_rows:
                continue
            positions = values.eq(value).to_numpy()
            metrics = regression_metrics(holdout.loc[positions, TARGET], predictions[positions])
            rows.append(
                {
                    "segment_type": column,
                    "segment": str(value),
                    "rows": int(count),
                    **metrics,
                }
            )
    return pd.DataFrame(rows, columns=["segment_type", "segment", "rows", "MAE", "RMSE", "R2", "MAPE", "MdAPE", "WAPE"])
