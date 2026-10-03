"""Prediction wrapper and listing price-position labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from modeling.config import ADVISOR_THRESHOLD_PERCENT
from modeling.features import make_feature_frame


def market_price_status(
    asking_price: float,
    predicted_market_price: float,
    threshold_percent: float = ADVISOR_THRESHOLD_PERCENT,
) -> str:
    """Label a listing by its asking-price gap from the estimated market price."""
    try:
        asking_price = float(asking_price)
        predicted_market_price = float(predicted_market_price)
        threshold_percent = float(threshold_percent)
    except (TypeError, ValueError):
        return "unknown"

    if not np.isfinite(threshold_percent) or threshold_percent < 0:
        raise ValueError("threshold_percent must be a finite, non-negative number")
    if not np.isfinite(asking_price) or not np.isfinite(predicted_market_price):
        return "unknown"
    if asking_price <= 0 or predicted_market_price <= 0:
        return "unknown"

    gap_percent = (asking_price / predicted_market_price - 1) * 100
    if gap_percent > threshold_percent:
        return "above_market"
    if gap_percent < -threshold_percent:
        return "below_market"
    return "market_price"


@dataclass
class PriceAdvisor:
    """Model bundle that keeps its inference schema beside the estimator."""

    estimator: Any
    feature_columns: tuple[str, ...]
    model_name: str
    target_transform: str = "log1p"

    def predict_expected_price(self, listings: pd.DataFrame) -> np.ndarray:
        features = make_feature_frame(listings)
        model_input = features.loc[:, list(self.feature_columns)]
        predicted_log_price = np.asarray(self.estimator.predict(model_input), dtype=float)
        predicted_log_price = np.nan_to_num(
            predicted_log_price,
            nan=0.0,
            posinf=30.0,
            neginf=0.0,
        )
        # The wide upper bound is a numerical guard for extrapolating linear baselines.
        return np.maximum(np.expm1(np.clip(predicted_log_price, 0.0, 30.0)), 1.0)

    def advise(
        self,
        listings: pd.DataFrame,
        threshold_percent: float = ADVISOR_THRESHOLD_PERCENT,
    ) -> pd.DataFrame:
        """Return a copy with market-price estimates and optional asking-price labels."""
        result = listings.copy()
        expected_price = self.predict_expected_price(listings)
        result["predicted_market_price"] = expected_price

        if "price" in listings.columns:
            asking_price = pd.to_numeric(listings["price"], errors="coerce")
            result["asking_price"] = asking_price
            result["price_gap"] = asking_price - expected_price
            result["price_gap_pct"] = (asking_price / expected_price - 1) * 100
            result["absolute_error"] = result["price_gap"].abs()
            result["absolute_percentage_error"] = (
                result["absolute_error"] / asking_price.where(asking_price.gt(0)) * 100
            )
            result["market_price_status"] = [
                market_price_status(asking, predicted, threshold_percent)
                for asking, predicted in zip(asking_price, expected_price, strict=True)
            ]

        return result
