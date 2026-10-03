"""Baseline, linear and gradient-boosted model candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modeling.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


@dataclass
class Experiment:
    name: str
    estimator: Any
    feature_columns: tuple[str, ...]
    stage: str = "baseline"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlendComponent:
    name: str
    estimator: Any
    feature_columns: tuple[str, ...]
    best_iteration: int | None


@dataclass
class WeightedBlendRegressor:
    """Blend component predictions in price space while exposing a log target."""

    components: tuple[BlendComponent, ...]
    weights: tuple[float, ...]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if len(self.components) != len(self.weights):
            raise ValueError("Each blend component must have exactly one weight")
        if not np.isclose(sum(self.weights), 1.0) or any(weight < 0 for weight in self.weights):
            raise ValueError("Blend weights must be non-negative and sum to one")

        blended_price = np.zeros(len(features), dtype=float)
        for component, weight in zip(self.components, self.weights, strict=True):
            model_input = features.loc[:, list(component.feature_columns)]
            log_prediction = np.asarray(component.estimator.predict(model_input), dtype=float)
            log_prediction = np.nan_to_num(log_prediction, nan=0.0, posinf=30.0, neginf=0.0)
            component_price = np.expm1(np.clip(log_prediction, 0.0, 30.0))
            blended_price += weight * component_price
        return np.log1p(np.maximum(blended_price, 1.0))


def _ridge_pipeline() -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=10.0, solver="lsqr")),
        ]
    )


def _catboost(
    depth: int,
    random_state: int,
    *,
    learning_rate: float = 0.05,
    l2_leaf_reg: float = 5.0,
    random_strength: float = 1.0,
    iterations: int = 1200,
) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_strength=random_strength,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=random_state,
        allow_writing_files=False,
        verbose=False,
    )


def catboost_parameter_key(
    parameters: dict[str, Any],
) -> tuple[int, float, float, float, int]:
    return (
        int(parameters["depth"]),
        round(float(parameters["learning_rate"]), 5),
        round(float(parameters["l2_leaf_reg"]), 5),
        round(float(parameters["random_strength"]), 5),
        int(parameters["iterations"]),
    )


def _catboost_name(parameters: dict[str, Any]) -> str:
    return (
        f"catboost_d{int(parameters['depth'])}"
        f"_lr{float(parameters['learning_rate']):.3f}"
        f"_l2{float(parameters['l2_leaf_reg']):g}"
        f"_rs{float(parameters['random_strength']):g}"
    )


def _catboost_experiment(
    parameters: dict[str, Any],
    random_state: int,
    stage: str,
) -> Experiment:
    return Experiment(
        name="catboost_baseline" if stage == "baseline" else _catboost_name(parameters),
        estimator=_catboost(random_state=random_state, **parameters),
        feature_columns=FEATURES,
        stage=stage,
        parameters=parameters.copy(),
    )


def build_experiments(
    random_state: int = 42,
    tune: bool = False,
    ablation: bool = False,
) -> list[Experiment]:
    """Build a small, reproducible model comparison and optional experiments."""
    baseline_parameters = {
        "depth": 8,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
        "iterations": 1200,
    }
    experiments = [
        Experiment(
            name="dummy_median",
            estimator=DummyRegressor(strategy="median"),
            feature_columns=FEATURES,
        ),
        Experiment(
            name="ridge",
            estimator=_ridge_pipeline(),
            feature_columns=FEATURES,
        ),
        _catboost_experiment(baseline_parameters, random_state, "baseline"),
    ]

    if tune:
        # Coarse search: vary tree depth and learning rate while keeping the
        # regularization fixed. The baseline configuration is already present.
        seen = {catboost_parameter_key(baseline_parameters)}
        for depth in (6, 8, 10):
            for learning_rate in (0.03, 0.07):
                parameters = {
                    **baseline_parameters,
                    "depth": depth,
                    "learning_rate": learning_rate,
                }
                key = catboost_parameter_key(parameters)
                if key in seen:
                    continue
                seen.add(key)
                experiments.append(
                    _catboost_experiment(parameters, random_state, "coarse")
                )

    if ablation:
        no_location = tuple(
            feature
            for feature in FEATURES
            if feature not in {"geo_lat", "geo_lon", "region"}
        )
        no_time = tuple(
            feature
            for feature in FEATURES
            if feature not in {"publication_year", "publication_month_sin", "publication_month_cos"}
        )
        experiments.extend(
            [
                Experiment(
                    name="catboost_without_location",
                    estimator=_catboost(random_state=random_state, **baseline_parameters),
                    feature_columns=no_location,
                    stage="ablation",
                    parameters=baseline_parameters.copy(),
                ),
                Experiment(
                    name="catboost_without_time",
                    estimator=_catboost(random_state=random_state, **baseline_parameters),
                    feature_columns=no_time,
                    stage="ablation",
                    parameters=baseline_parameters.copy(),
                ),
            ]
        )

    return experiments


def build_fine_tuning_experiments(
    anchor: Experiment,
    random_state: int,
    seen_configurations: set[tuple[int, float, float, float, int]],
) -> list[Experiment]:
    """Create a small local search around the best coarse CatBoost candidate."""
    base = anchor.parameters
    if not base:
        raise ValueError("The fine-tuning anchor must include CatBoost parameters")

    proposals = [
        {**base, "depth": int(base["depth"]) - 1},
        {**base, "depth": int(base["depth"]) + 1},
        {**base, "learning_rate": round(float(base["learning_rate"]) * 0.8, 4)},
        {**base, "learning_rate": round(float(base["learning_rate"]) * 1.2, 4)},
        {**base, "l2_leaf_reg": 3.0},
        {**base, "l2_leaf_reg": 8.0},
        {**base, "random_strength": 0.5},
        {**base, "random_strength": 1.5},
    ]

    experiments: list[Experiment] = []
    seen = set(seen_configurations)
    for parameters in proposals:
        if int(parameters["depth"]) < 4 or int(parameters["depth"]) > 12:
            continue
        key = catboost_parameter_key(parameters)
        if key in seen:
            continue
        seen.add(key)
        experiments.append(_catboost_experiment(parameters, random_state, "fine"))
    return experiments


def categorical_columns(feature_columns: tuple[str, ...]) -> list[str]:
    return [column for column in CATEGORICAL_FEATURES if column in feature_columns]
