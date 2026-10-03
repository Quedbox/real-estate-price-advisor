"""Training orchestration, model selection and artifact persistence."""

from __future__ import annotations

import json
import logging
import os
import platform
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modeling.advisor import PriceAdvisor
from modeling.config import (
    ADVISOR_THRESHOLD_PERCENT,
    CATEGORICAL_FEATURES,
    FEATURES,
    FIGURES_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
)
from modeling.evaluation import chronological_split, regression_metrics, segment_metrics
from modeling.features import make_feature_frame
from modeling.models import (
    BlendComponent,
    Experiment,
    WeightedBlendRegressor,
    build_experiments,
    build_fine_tuning_experiments,
    catboost_parameter_key,
    categorical_columns,
)

LOGGER = logging.getLogger(__name__)


def _predict_prices(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    log_prediction = np.asarray(estimator.predict(features), dtype=float).reshape(-1)
    log_prediction = np.nan_to_num(log_prediction, nan=0.0, posinf=30.0, neginf=0.0)
    return np.expm1(np.clip(log_prediction, 0.0, 30.0))


def _fit_experiment(
    experiment: Experiment,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    engineered: pd.DataFrame,
) -> tuple[Any, np.ndarray, int | None]:
    columns = list(experiment.feature_columns)
    x_train = engineered.loc[train.index, columns]
    x_validation = engineered.loc[validation.index, columns]
    y_train_log = np.log1p(train[TARGET])
    y_validation_log = np.log1p(validation[TARGET])

    if experiment.name.startswith("catboost"):
        experiment.estimator.fit(
            x_train,
            y_train_log,
            cat_features=categorical_columns(experiment.feature_columns),
            eval_set=(x_validation, y_validation_log),
            early_stopping_rounds=100,
            verbose=False,
        )
        best_iteration = experiment.estimator.get_best_iteration()
        best_iteration = int(best_iteration) if best_iteration >= 0 else None
    else:
        experiment.estimator.fit(x_train, y_train_log)
        best_iteration = None

    prediction = _predict_prices(experiment.estimator, x_validation)
    return experiment.estimator, prediction, best_iteration


def _refit_selected(
    experiment: Experiment,
    estimator: Any,
    development: pd.DataFrame,
    engineered: pd.DataFrame,
    best_iteration: int | None,
) -> Any:
    if isinstance(estimator, WeightedBlendRegressor):
        refit_components = []
        for component in estimator.components:
            component_experiment = Experiment(
                name=component.name,
                estimator=component.estimator,
                feature_columns=component.feature_columns,
            )
            refit_estimator = _refit_selected(
                component_experiment,
                component.estimator,
                development,
                engineered,
                component.best_iteration,
            )
            refit_components.append(
                BlendComponent(
                    name=component.name,
                    estimator=refit_estimator,
                    feature_columns=component.feature_columns,
                    best_iteration=component.best_iteration,
                )
            )
        return WeightedBlendRegressor(tuple(refit_components), estimator.weights)

    columns = list(experiment.feature_columns)
    x_development = engineered.loc[development.index, columns]
    y_development_log = np.log1p(development[TARGET])

    if experiment.name.startswith("catboost"):
        parameters = estimator.get_params()
        if best_iteration is not None:
            parameters["iterations"] = max(1, best_iteration + 1)
        final_estimator = type(estimator)(**parameters)
        final_estimator.fit(
            x_development,
            y_development_log,
            cat_features=categorical_columns(experiment.feature_columns),
            verbose=False,
        )
    else:
        final_estimator = estimator
        final_estimator.fit(x_development, y_development_log)
    return final_estimator


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        handle.write("\n")


def _candidate_parameters(experiment: Experiment, estimator: Any) -> dict[str, object]:
    if experiment.parameters:
        return experiment.parameters.copy()
    if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
        model = estimator.named_steps["model"]
        return {"alpha": model.alpha} if hasattr(model, "alpha") else {}
    return {}


def _candidate_leaderboard(
    records: list[dict[str, object]], selected_model: str
) -> pd.DataFrame:
    leaderboard = pd.DataFrame(records).sort_values("MAE").reset_index(drop=True)
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))
    leaderboard["selected"] = leaderboard["model"].eq(selected_model)

    baseline = leaderboard.loc[leaderboard["model"].eq("catboost_baseline"), "MAE"]
    if not baseline.empty and baseline.iloc[0] > 0:
        baseline_mae = float(baseline.iloc[0])
        leaderboard["MAE_gain_vs_catboost_baseline"] = baseline_mae - leaderboard["MAE"]
        leaderboard["MAE_gain_percent_vs_catboost_baseline"] = (
            leaderboard["MAE_gain_vs_catboost_baseline"] / baseline_mae * 100
        )
    else:
        leaderboard["MAE_gain_vs_catboost_baseline"] = np.nan
        leaderboard["MAE_gain_percent_vs_catboost_baseline"] = np.nan
    return leaderboard


def _selection_summary(
    leaderboard: pd.DataFrame,
    selected_model: str,
    holdout_metrics: dict[str, float],
    split_summary: dict[str, object],
    tune: bool,
) -> str:
    selected = leaderboard.loc[leaderboard["model"].eq(selected_model)].iloc[0]
    catboost_candidates = leaderboard.loc[
        leaderboard["stage"].isin(["baseline", "coarse", "fine"])
        & leaderboard["model"].astype(str).str.startswith("catboost_")
    ].sort_values("MAE")
    baseline = leaderboard.loc[leaderboard["model"].eq("catboost_baseline")]
    coarse_count = int(leaderboard["stage"].eq("coarse").sum())
    fine_count = int(leaderboard["stage"].eq("fine").sum())
    top = leaderboard.head(5)
    selected_parameters = json.loads(str(selected["parameters"]))

    lines = [
        "# Model selection and tuning results",
        "",
        f"Selected candidate: **`{selected_model}`**",
        f"Selected configuration: `{json.dumps(selected_parameters, sort_keys=True)}`",
        f"Validation MAE: **{selected['MAE']:,.0f} RUB**",
        f"Final temporal holdout MAE after refitting: **{holdout_metrics['MAE']:,.0f} RUB**",
        "",
        "## Search protocol",
        "",
        "Candidates were ranked by MAE on the same chronological validation window. "
        "The final temporal holdout was evaluated only after candidate selection.",
        f"Split sizes: train {int(split_summary['train_rows']):,}, "
        f"validation {int(split_summary['validation_rows']):,}, "
        f"holdout {int(split_summary['holdout_rows']):,} listings.",
    ]

    if tune:
        lines.extend(
            [
                "",
                f"Fine tuning was enabled: {coarse_count} coarse-search and "
                f"{fine_count} local-search CatBoost configurations were evaluated, "
                "alongside the baseline models.",
            ]
        )
        if not baseline.empty and not catboost_candidates.empty:
            baseline_mae = float(baseline.iloc[0]["MAE"])
            best_catboost = catboost_candidates.iloc[0]
            gain = baseline_mae - float(best_catboost["MAE"])
            gain_percent = gain / baseline_mae * 100 if baseline_mae else 0.0
            parameters = json.loads(str(best_catboost["parameters"]))
            iteration = best_catboost["best_iteration"]
            iteration_text = (
                f"; best iteration {int(iteration)}" if pd.notna(iteration) else ""
            )
            lines.append(
                "Best full-feature CatBoost configuration: "
                f"`{best_catboost['model']}` with `{json.dumps(parameters, sort_keys=True)}`"
                f"{iteration_text}."
            )
            if gain_percent >= 1.0:
                lines.append(
                    f"It reduced validation MAE by **{gain:,.0f} RUB ({gain_percent:.2f}%)** "
                    "against the default CatBoost configuration."
                )
            elif gain_percent > 0:
                lines.append(
                    f"It reduced validation MAE by **{gain:,.0f} RUB ({gain_percent:.2f}%)** "
                    "against the default configuration; this is a small validation gain."
                )
            elif gain_percent == 0:
                lines.append(
                    "The search tied the default CatBoost configuration on validation MAE."
                )
            else:
                lines.append(
                    f"The search did not beat default CatBoost: the best searched "
                    f"configuration was {-gain:,.0f} RUB ({-gain_percent:.2f}%) worse "
                    "on validation MAE."
                )
    else:
        lines.extend(
            [
                "",
                "Fine tuning was not enabled for this run. Use `--tune` to run the "
                "coarse CatBoost search followed by a local search around its leader.",
            ]
        )

    lines.extend(
        [
            "",
            "## Validation leaderboard",
            "",
            "| Rank | Candidate | Stage | MAE (RUB) | RMSE (RUB) | Selected |",
            "|---:|---|---|---:|---:|:---:|",
        ]
    )
    for _, row in top.iterrows():
        selected_mark = "yes" if bool(row["selected"]) else ""
        lines.append(
            f"| {int(row['rank'])} | `{row['model']}` | {row['stage']} | "
            f"{row['MAE']:,.0f} | {row['RMSE']:,.0f} | {selected_mark} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The selected candidate minimizes validation MAE across {len(leaderboard)} "
            "model and blend candidates. Its holdout metrics are a later-period estimate "
            "of generalization, not part of the tuning decision.",
            "",
            f"Holdout RMSE: {holdout_metrics['RMSE']:,.0f} RUB; "
            f"MAPE: {holdout_metrics['MAPE']:.2f}%; "
            f"R²: {holdout_metrics['R2']:.4f}.",
            "",
            "These conclusions describe this dataset and these temporal windows. "
            "A fresh, later time split is needed before treating a small validation "
            "difference as a stable improvement.",
            "",
        ]
    )
    return "\n".join(lines)


def _feature_importance(advisor: PriceAdvisor) -> pd.DataFrame:
    estimator = advisor.estimator
    if isinstance(estimator, WeightedBlendRegressor):
        component_index = next(
            (
                index
                for index, candidate in enumerate(estimator.components)
                if hasattr(candidate.estimator, "get_feature_importance")
                or (
                    hasattr(candidate.estimator, "named_steps")
                    and hasattr(candidate.estimator.named_steps.get("model"), "coef_")
                )
            ),
            0,
        )
        component = estimator.components[component_index]
        component_advisor = PriceAdvisor(
            estimator=component.estimator,
            feature_columns=component.feature_columns,
            model_name=component.name,
        )
        importance = _feature_importance(component_advisor)
        if not importance.empty:
            importance["source_model"] = component.name
            importance["blend_weight"] = estimator.weights[component_index]
        return importance

    if hasattr(estimator, "get_feature_importance"):
        values = estimator.get_feature_importance()
        return pd.DataFrame(
            {
                "feature": advisor.feature_columns,
                "importance": values,
                "source_model": advisor.model_name,
                "blend_weight": 1.0,
            }
        ).sort_values("importance", ascending=False)

    if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
        model = estimator.named_steps["model"]
        if hasattr(model, "coef_"):
            preprocessor = estimator.named_steps["preprocessor"]
            names = preprocessor.get_feature_names_out()
            return pd.DataFrame(
                {
                    "feature": names,
                    "importance": np.abs(model.coef_),
                    "source_model": advisor.model_name,
                    "blend_weight": 1.0,
                }
            ).sort_values("importance", ascending=False)

    return pd.DataFrame(columns=["feature", "importance", "source_model", "blend_weight"])


def _model_parameters(estimator: Any) -> dict[str, object]:
    if isinstance(estimator, WeightedBlendRegressor):
        return {
            "estimator": type(estimator).__name__,
            "weights": list(estimator.weights),
            "components": [
                {
                    "model": component.name,
                    "best_iteration": component.best_iteration,
                    "parameters": _model_parameters(component.estimator),
                }
                for component in estimator.components
            ],
        }
    if hasattr(estimator, "named_steps"):
        model = estimator.named_steps["model"]
        return {
            "estimator": type(model).__name__,
            "parameters": model.get_params(),
            "preprocessing": {
                "numeric_imputation": "median",
                "numeric_scaling": "standard",
                "categorical_imputation": "most_frequent",
                "one_hot_min_frequency": 20,
                "unknown_categories": "ignored",
            },
        }
    return {
        "estimator": type(estimator).__name__,
        "parameters": estimator.get_params(),
    }


def _save_figures(metrics: pd.DataFrame, predictions: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    validation = metrics.loc[metrics["split"].eq("validation")].sort_values("MAE")

    labels = (
        validation["model"]
        .str.replace("catboost_depth", "CatBoost d", regex=False)
        .str.replace("dummy_median", "Median baseline", regex=False)
        .str.replace("blend_", "blend ", regex=False)
        .str.replace("_25_75", " (25/75)", regex=False)
        .str.replace("_50_50", " (50/50)", regex=False)
        .str.replace("_75_25", " (75/25)", regex=False)
        .str.replace("_", " ", regex=False)
    )
    fig, ax = plt.subplots(figsize=(10, max(4, len(validation) * 0.45)))
    ax.barh(labels.iloc[::-1], validation["MAE"].iloc[::-1], color="#2f6690")
    ax.set_title("Chronological validation: model comparison")
    ax.set_xlabel("MAE (RUB)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "model_comparison_mae.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    sample = predictions
    if len(sample) > 20_000:
        sample = sample.sample(20_000, random_state=RANDOM_STATE)
    upper = float(
        np.nanquantile(
            np.concatenate([sample[TARGET].to_numpy(), sample["predicted_market_price"].to_numpy()]),
            0.995,
        )
    )
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        sample[TARGET],
        sample["predicted_market_price"],
        alpha=0.18,
        s=10,
        color="#3a7d44",
        edgecolors="none",
    )
    ax.plot([1, upper], [1, upper], color="#c44536", linestyle="--", linewidth=1.5)
    ax.set_xlim(1, upper)
    ax.set_ylim(1, upper)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Final temporal holdout: actual vs predicted price")
    ax.set_xlabel("Actual listing price (RUB, log scale)")
    ax.set_ylabel("Predicted market price (RUB, log scale)")
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(figures_dir / "actual_vs_predicted.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    error_sample = predictions["absolute_percentage_error"].dropna()
    if len(error_sample) > 100_000:
        error_sample = error_sample.sample(100_000, random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(error_sample.clip(upper=100), bins=50, color="#bc6c25", alpha=0.9)
    ax.set_title("Final temporal holdout: absolute percentage error")
    ax.set_xlabel("Absolute percentage error, clipped at 100%")
    ax.set_ylabel("Listings")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_dir / "absolute_percentage_error.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_artifacts(
    advisor: PriceAdvisor,
    experiment: Experiment,
    metrics: pd.DataFrame,
    candidate_leaderboard: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    holdout_predictions: pd.DataFrame,
    segments: pd.DataFrame,
    holdout_metrics: dict[str, float],
    split_summary: dict[str, object],
    tune: bool,
    ablation: bool,
    random_state: int,
    models_dir: Path,
    reports_dir: Path,
    figures_dir: Path,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "best_model.joblib"
    descriptor, temporary_path = tempfile.mkstemp(
        prefix="best_model_", suffix=".joblib", dir=models_dir
    )
    os.close(descriptor)
    try:
        joblib.dump(advisor, temporary_path)
        os.replace(temporary_path, model_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)

    metrics.to_csv(reports_dir / "model_metrics.csv", index=False)
    candidate_leaderboard.to_csv(reports_dir / "candidate_leaderboard.csv", index=False)
    holdout_predictions.to_csv(reports_dir / "holdout_predictions.csv", index=False)
    validation_predictions.to_csv(reports_dir / "validation_predictions.csv", index=False)
    segments.to_csv(reports_dir / "segment_metrics.csv", index=False)

    importance = _feature_importance(advisor)
    importance.to_csv(reports_dir / "feature_importance.csv", index=False)
    if not importance.empty:
        top_features = importance.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(top_features["feature"], top_features["importance"], color="#52796f")
        source_model = str(importance.iloc[0].get("source_model", experiment.name))
        ax.set_title(f"Top feature importance: {source_model}")
        ax.set_xlabel("Absolute coefficient" if "ridge" in source_model else "Importance")
        fig.tight_layout()
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures_dir / "feature_importance.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    _save_figures(metrics, holdout_predictions, figures_dir)
    resolved_model_path = model_path.resolve()
    try:
        model_artifact = resolved_model_path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        model_artifact = resolved_model_path.as_posix()
    metadata = {
        "project": "Real Estate Price Advisor",
        "model_artifact": model_artifact,
        "selected_model": experiment.name,
        "selected_model_parameters": _model_parameters(advisor.estimator),
        "random_state": random_state,
        "target": TARGET,
        "target_transformation": "log1p(price)",
        "selection_metric": "validation MAE in RUB",
        "catboost_early_stopping_metric": "RMSE on log1p(price)",
        "selection_policy": "choose on chronological validation; evaluate once on later holdout",
        "numeric_features": [
            feature for feature in experiment.feature_columns if feature not in CATEGORICAL_FEATURES
        ],
        "categorical_features": [
            feature for feature in experiment.feature_columns if feature in CATEGORICAL_FEATURES
        ],
        "excluded_from_features": [
            "price",
            "published_at",
            "date",
            "time",
            "price_per_m2",
        ],
        "hyperparameter_search_enabled": tune,
        "hyperparameter_search_strategy": (
            "coarse CatBoost depth/learning-rate grid followed by a local search over "
            "depth, learning rate, L2 regularization, and random strength"
            if tune
            else "disabled"
        ),
        "validation_candidate_count": len(candidate_leaderboard),
        "feature_ablation_enabled": ablation,
        "blend_weight_grid": [0.25, 0.5, 0.75],
        "advisor_threshold_percent": ADVISOR_THRESHOLD_PERCENT,
        "split": split_summary,
        "metrics": json.loads(metrics.to_json(orient="records")),
        "versions": {
            "python": platform.python_version(),
            "pandas": _package_version("pandas"),
            "numpy": _package_version("numpy"),
            "scikit_learn": _package_version("scikit-learn"),
            "catboost": _package_version("catboost"),
        },
    }
    _save_json(reports_dir / "model_metadata.json", metadata)
    summary = _selection_summary(
        candidate_leaderboard,
        experiment.name,
        holdout_metrics,
        split_summary,
        tune,
    )
    (reports_dir / "model_selection_summary.md").write_text(summary, encoding="utf-8")


def train_and_evaluate(
    listings: pd.DataFrame,
    *,
    tune: bool = False,
    ablation: bool = False,
    random_state: int = RANDOM_STATE,
    models_dir: str | Path = MODELS_DIR,
    reports_dir: str | Path = REPORTS_DIR,
    figures_dir: str | Path = FIGURES_DIR,
) -> tuple[pd.DataFrame, PriceAdvisor]:
    """Compare candidates, select on validation and score a final time holdout."""
    if TARGET not in listings:
        raise ValueError(f"Training data must contain the target column {TARGET!r}")

    split = chronological_split(listings)
    engineered = make_feature_frame(listings).reset_index(drop=True)
    records: list[dict[str, object]] = []
    candidate_records: list[dict[str, object]] = []
    selected_experiment: Experiment | None = None
    selected_estimator: Any = None
    selected_iteration: int | None = None
    selected_validation_prediction: np.ndarray | None = None
    best_validation_mae = float("inf")
    best_tuning_anchor: tuple[float, Experiment] | None = None
    seen_catboost_configurations: set[tuple[int, float, float, float, int]] = set()
    top_candidates: list[
        tuple[float, Experiment, Any, int | None, np.ndarray]
    ] = []

    def evaluate_candidate(experiment: Experiment) -> None:
        nonlocal selected_experiment
        nonlocal selected_estimator
        nonlocal selected_iteration
        nonlocal selected_validation_prediction
        nonlocal best_validation_mae
        nonlocal best_tuning_anchor

        LOGGER.info("Fitting %s", experiment.name)
        estimator, validation_prediction, best_iteration = _fit_experiment(
            experiment, split.train, split.validation, engineered
        )
        metrics = regression_metrics(split.validation[TARGET], validation_prediction)
        parameters = _candidate_parameters(experiment, estimator)
        candidate_record = {
            "model": experiment.name,
            "split": "validation",
            "stage": experiment.stage,
            "parameters": json.dumps(parameters, sort_keys=True, default=str),
            "best_iteration": best_iteration,
            **metrics,
        }
        records.append(candidate_record)
        candidate_records.append(candidate_record.copy())

        if experiment.name.startswith("catboost_") and experiment.parameters:
            seen_catboost_configurations.add(catboost_parameter_key(experiment.parameters))
            if (
                experiment.stage in {"baseline", "coarse"}
                and experiment.feature_columns == FEATURES
                and (best_tuning_anchor is None or metrics["MAE"] < best_tuning_anchor[0])
            ):
                best_tuning_anchor = (metrics["MAE"], experiment)

        # Keep fitted estimators only for the two validation leaders.
        experiment.estimator = None
        top_candidates.append(
            (metrics["MAE"], experiment, estimator, best_iteration, validation_prediction)
        )
        top_candidates.sort(key=lambda candidate: candidate[0])
        del top_candidates[2:]
        if metrics["MAE"] < best_validation_mae:
            selected_experiment = experiment
            selected_estimator = estimator
            selected_iteration = best_iteration
            selected_validation_prediction = validation_prediction
            best_validation_mae = metrics["MAE"]
        del estimator, validation_prediction

    experiments = build_experiments(random_state, tune, ablation)
    for experiment in experiments:
        if experiment.stage != "ablation":
            evaluate_candidate(experiment)

    if tune and best_tuning_anchor is not None:
        fine_experiments = build_fine_tuning_experiments(
            best_tuning_anchor[1],
            random_state,
            seen_catboost_configurations,
        )
        LOGGER.info(
            "Fine-tuning %d CatBoost candidates around %s",
            len(fine_experiments),
            best_tuning_anchor[1].name,
        )
        for experiment in fine_experiments:
            evaluate_candidate(experiment)

    for experiment in experiments:
        if experiment.stage == "ablation":
            evaluate_candidate(experiment)

    if len(top_candidates) >= 2:
        primary, secondary = top_candidates[:2]
        components = (
            BlendComponent(
                primary[1].name,
                primary[2],
                primary[1].feature_columns,
                primary[3],
            ),
            BlendComponent(
                secondary[1].name,
                secondary[2],
                secondary[1].feature_columns,
                secondary[3],
            ),
        )
        for primary_weight in (0.25, 0.5, 0.75):
            blend_prediction = (
                primary_weight * primary[4] + (1 - primary_weight) * secondary[4]
            )
            blend_metrics = regression_metrics(split.validation[TARGET], blend_prediction)
            blend_name = (
                f"blend_{primary[1].name}_{secondary[1].name}_"
                f"{int(primary_weight * 100)}_{int((1 - primary_weight) * 100)}"
            )
            records.append(
                {
                    "model": blend_name,
                    "split": "validation",
                    "stage": "ensemble",
                    "parameters": json.dumps(
                        {
                            "components": [primary[1].name, secondary[1].name],
                            "weights": [primary_weight, 1 - primary_weight],
                        },
                        sort_keys=True,
                    ),
                    "best_iteration": None,
                    **blend_metrics,
                }
            )
            candidate_records.append(records[-1].copy())
            if blend_metrics["MAE"] < best_validation_mae:
                selected_experiment = Experiment(
                    name=blend_name,
                    estimator=None,
                    feature_columns=tuple(
                        dict.fromkeys(primary[1].feature_columns + secondary[1].feature_columns)
                    ),
                )
                selected_estimator = WeightedBlendRegressor(
                    components=components,
                    weights=(primary_weight, 1 - primary_weight),
                )
                selected_iteration = None
                selected_validation_prediction = blend_prediction
                best_validation_mae = blend_metrics["MAE"]

    if selected_experiment is None or selected_validation_prediction is None:
        raise RuntimeError("No model candidate completed successfully")
    selected_name = selected_experiment.name
    candidate_leaderboard = _candidate_leaderboard(candidate_records, selected_name)
    metrics_frame = pd.DataFrame(records).sort_values(
        ["MAE", "RMSE", "model"], kind="stable"
    ).reset_index(drop=True)
    # Keep the indices aligned with the one shared engineered feature frame.
    development = pd.concat([split.train, split.validation]).sort_values("published_at")
    final_estimator = _refit_selected(
        selected_experiment,
        selected_estimator,
        development,
        engineered,
        selected_iteration,
    )
    advisor = PriceAdvisor(
        estimator=final_estimator,
        feature_columns=selected_experiment.feature_columns,
        model_name=selected_name,
    )

    holdout_source = listings.iloc[split.holdout.index]
    holdout_predictions = advisor.advise(holdout_source)
    y_holdout = holdout_source[TARGET].to_numpy()
    y_predicted = holdout_predictions["predicted_market_price"].to_numpy()
    holdout_metrics = regression_metrics(y_holdout, y_predicted)
    metrics_frame = pd.concat(
        [
            metrics_frame,
            pd.DataFrame(
                [{"model": selected_name, "split": "holdout", "best_iteration": selected_iteration, **holdout_metrics}]
            ),
        ],
        ignore_index=True,
    )

    validation_predictions = split.validation.copy()
    validation_predictions["predicted_market_price"] = selected_validation_prediction
    validation_predictions["absolute_error"] = (
        validation_predictions[TARGET] - selected_validation_prediction
    ).abs()
    segments = segment_metrics(split.holdout, y_predicted)
    _write_artifacts(
        advisor,
        selected_experiment,
        metrics_frame,
        candidate_leaderboard,
        validation_predictions,
        holdout_predictions,
        segments,
        holdout_metrics,
        split.summary,
        tune,
        ablation,
        random_state,
        Path(models_dir),
        Path(reports_dir),
        Path(figures_dir),
    )

    LOGGER.info("Selected %s using validation MAE", selected_name)
    LOGGER.info("Holdout MAE: %.0f RUB", holdout_metrics["MAE"])
    LOGGER.info("Wrote model, metrics, predictions and diagnostics")
    return metrics_frame, advisor


def load_advisor(model_path: str | Path) -> PriceAdvisor:
    """Load a persisted advisor bundle and reject incompatible artifacts."""
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact was not found: {path}")
    advisor = joblib.load(path)
    if not isinstance(advisor, PriceAdvisor):
        raise ValueError("The artifact is not a compatible PriceAdvisor model bundle")
    return advisor
