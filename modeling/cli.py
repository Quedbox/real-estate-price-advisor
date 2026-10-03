"""Command-line entry point: prepare data, train models and score listings."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from modeling.config import (
    ADVISOR_THRESHOLD_PERCENT,
    MODELS_DIR,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    REPORTS_DIR,
)
from modeling.data import prepare_dataset, read_listings
from modeling.pipeline import load_advisor, train_and_evaluate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="real-estate-advisor",
        description="Prepare, train and apply the real-estate price advisor model.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="clean raw listing data")
    prepare.add_argument("--input", type=Path, default=RAW_DATA_PATH)
    prepare.add_argument("--output", type=Path, default=PROCESSED_DATA_PATH)
    prepare.add_argument("--report", type=Path, default=REPORTS_DIR / "cleaning_report.csv")

    train = commands.add_parser("train", help="compare models and evaluate a temporal holdout")
    train.add_argument("--data", type=Path, default=PROCESSED_DATA_PATH)
    train.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    train.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    train.add_argument("--figures-dir", type=Path, default=REPORTS_DIR / "figures")
    train.add_argument(
        "--tune",
        action="store_true",
        help="run a coarse and local CatBoost search; write candidate conclusions",
    )
    train.add_argument("--ablation", action="store_true", help="compare models without location/time features")
    train.add_argument("--random-state", type=int, default=42)

    predict = commands.add_parser("predict", help="estimate market prices for new listings")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--model", type=Path, default=MODELS_DIR / "best_model.joblib")
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument(
        "--threshold-percent",
        type=float,
        default=ADVISOR_THRESHOLD_PERCENT,
    )

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parser().parse_args()

    if args.command == "prepare":
        raw_rows, clean_rows = prepare_dataset(args.input, args.output, args.report)
        print(f"Prepared {clean_rows:,} of {raw_rows:,} rows.")
        print(f"Clean data: {args.output}")
        print(f"Cleaning audit: {args.report}")
    elif args.command == "train":
        listings = read_listings(args.data)
        metrics, advisor = train_and_evaluate(
            listings,
            tune=args.tune,
            ablation=args.ablation,
            random_state=args.random_state,
            models_dir=args.models_dir,
            reports_dir=args.reports_dir,
            figures_dir=args.figures_dir,
        )
        holdout = metrics.loc[metrics["split"].eq("holdout")].iloc[0]
        validation = metrics.loc[
            metrics["split"].eq("validation") & metrics["model"].eq(advisor.model_name)
        ].iloc[0]
        print(f"Selected model: {advisor.model_name}")
        print(f"Validation MAE: {validation['MAE']:,.0f} RUB")
        print(f"Final temporal holdout MAE: {holdout['MAE']:,.0f} RUB")
        print(f"Reports: {args.reports_dir}")
        print(f"Selection summary: {args.reports_dir / 'model_selection_summary.md'}")
    elif args.command == "predict":
        advisor = load_advisor(args.model)
        listings = read_listings(args.input)
        predictions = advisor.advise(listings, threshold_percent=args.threshold_percent)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output, index=False)
        print(f"Scored {len(predictions):,} listings with {advisor.model_name}.")
        print(f"Predictions: {args.output}")


if __name__ == "__main__":
    main()
