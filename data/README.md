# Data

The source dataset is the Kaggle **Russia Real Estate 2018-2021** listings dataset. The large source CSV is intentionally not tracked by Git.

Download the CSV and save it as:

```text
data/raw/real_estate_raw.csv
```

Then create the cleaned dataset and row-count audit with:

```bash
uv run python -m modeling.cli prepare
```

The default outputs are `data/processed/real_estate_clean.csv` and `reports/cleaning_report.csv`. The original file and generated data are ignored by Git.
