"""Input validation, deterministic cleaning and dataset preparation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "price",
    "area",
    "rooms",
    "floor",
    "floors",
    "kitchen_area",
    "geo_lat",
    "geo_lon",
    "region",
    "building_type",
    "object_type",
}
NUMERIC_COLUMNS = (
    "price",
    "area",
    "rooms",
    "floor",
    "floors",
    "kitchen_area",
    "geo_lat",
    "geo_lon",
)


def read_listings(path: str | Path) -> pd.DataFrame:
    """Read a CSV and provide a useful error when the input is unavailable."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(
            f"Listing data was not found at {source}. Download the source data "
            "and place it in data/raw, or pass its path with --input."
        )

    try:
        return pd.read_csv(source, low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"The input CSV is empty: {source}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse the input CSV: {source}") from exc


def _append_stage(
    report: list[dict[str, int | float | str]],
    name: str,
    rows_before: int,
    rows_after: int,
) -> None:
    removed = rows_before - rows_after
    report.append(
        {
            "stage": name,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": removed,
            "removed_percent": round(removed / rows_before * 100, 3)
            if rows_before
            else 0.0,
        }
    )


def clean_listings(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean impossible records while retaining plausible market extremes.

    Price-dependent percentile filters are intentionally excluded here. They
    would use the label to decide which validation and holdout rows exist.
    """
    missing = sorted(REQUIRED_COLUMNS.difference(raw.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    cleaned = raw.copy()
    report: list[dict[str, int | float | str]] = []

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "published_at" in cleaned:
        published_at = pd.to_datetime(cleaned["published_at"], errors="coerce")
    elif {"date", "time"}.issubset(cleaned.columns):
        published_at = pd.to_datetime(
            cleaned["date"].astype("string") + " " + cleaned["time"].astype("string"),
            errors="coerce",
        )
    else:
        published_at = pd.Series(pd.NaT, index=cleaned.index, dtype="datetime64[ns]")
    cleaned["published_at"] = published_at

    before = len(cleaned)
    cleaned = cleaned.dropna(subset=["price", "area", "rooms"])
    _append_stage(report, "required target and property fields", before, len(cleaned))

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates().copy()
    _append_stage(report, "exact duplicate rows", before, len(cleaned))

    before = len(cleaned)
    valid_core = cleaned["price"].gt(0) & cleaned["area"].gt(0) & cleaned["rooms"].ge(-1)
    cleaned = cleaned.loc[valid_core].copy()
    _append_stage(report, "positive price and area; valid room count", before, len(cleaned))

    before = len(cleaned)
    valid_floors = (
        cleaned["floor"].ge(1)
        & cleaned["floors"].ge(1)
        & cleaned["floor"].le(cleaned["floors"])
    )
    cleaned = cleaned.loc[valid_floors].copy()
    _append_stage(report, "valid floor numbers", before, len(cleaned))

    before = len(cleaned)
    valid_kitchen = cleaned["kitchen_area"].isna() | (
        cleaned["kitchen_area"].ge(0) & cleaned["kitchen_area"].le(cleaned["area"])
    )
    cleaned = cleaned.loc[valid_kitchen].copy()
    _append_stage(report, "valid kitchen area", before, len(cleaned))

    before = len(cleaned)
    valid_geo = cleaned["geo_lat"].between(41, 82) & cleaned["geo_lon"].between(19, 180)
    cleaned = cleaned.loc[valid_geo].copy()
    _append_stage(report, "valid geographic coordinates", before, len(cleaned))

    cleaned["publication_year"] = cleaned["published_at"].dt.year.astype("Int64")
    cleaned["publication_month"] = cleaned["published_at"].dt.month.astype("Int64")
    cleaned["publication_quarter"] = cleaned["published_at"].dt.quarter.astype("Int64")

    return cleaned.reset_index(drop=True), pd.DataFrame(report)


def prepare_dataset(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
) -> tuple[int, int]:
    """Clean one source CSV and write the processed data and audit report."""
    raw = read_listings(input_path)
    cleaned, report = clean_listings(raw)

    output = Path(output_path)
    audit = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output, index=False)
    report.to_csv(audit, index=False)
    return len(raw), len(cleaned)
