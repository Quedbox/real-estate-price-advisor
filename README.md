# Real Estate Price Advisor

A machine learning project for estimating real estate prices from listing data.

The model predicts the expected market price of a property and then compares it with the actual listing price. Based on this difference, the listing can be marked as:

- `below_market`
- `market_price`
- `above_market`

## Dataset

Dataset: **Russia Real Estate 2018–2021** from Kaggle.

The dataset contains Russian real estate listings with features such as:

- price
- publication date and time
- latitude and longitude
- region
- building type
- floor and total floors
- number of rooms
- total area
- kitchen area
- object type

The dataset contains some input errors and outliers, so basic cleaning was required before modeling.

## Project structure

```text
real-estate-price-advisor/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_loading_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling.ipynb
│
├── models/
│   └── catboost_model.joblib
│
└── reports/
    ├── figures/
    ├── model_metrics.csv
    ├── test_predictions.csv
    └── model_metadata.json
```

## Workflow

1. Load and clean the data
2. Explore the dataset
3. Create additional features
4. Filter clear outliers
5. Split the data by time
6. Train baseline models
7. Train CatBoostRegressor
8. Evaluate model quality
9. Add Price Advisor logic
10. Save model, metrics and predictions

## Data cleaning

The cleaning step included:

- removing exact duplicates
- removing rows with impossible prices or areas
- checking floor values
- checking kitchen area values
- checking geo coordinates
- handling missing values in important columns
- filtering extreme price and area outliers

Outlier filtering was important because the dataset contained clearly wrong examples, such as listings with a price of `1 RUB` or extremely expensive listings with very small area.

## Feature engineering

Created features:

- `publication_year`
- `publication_month`
- `publication_quarter`
- `floor_ratio`
- `is_first_floor`
- `is_last_floor`
- `is_studio`
- `rooms_count`
- `kitchen_area_missing`

The feature `price_per_m2` was used only for EDA and outlier filtering. It was not used for model training because it is calculated from `price` and would cause data leakage.

## Models

I compared three models:

- `DummyRegressor` as a simple baseline
- `Ridge` as a linear baseline
- `CatBoostRegressor` as the main model

The target variable was transformed with:

```python
np.log1p(price)
```

Predictions were converted back to the original price scale before calculating metrics.

## Validation

I used a time-based train/test split.

The model was trained on older listings and tested on newer listings. This is closer to a real use case than a fully random split.

## Results

Best model: **CatBoostRegressor**

| Metric | Value |
|---|---:|
| MAE | 1.29M RUB |
| RMSE | 3.85M RUB |
| R² | 0.76 |
| MAPE | 19.95% |
| MdAPE | 16.75% |
| WAPE | 22.57% |

CatBoost showed the best result and was used as the final model.

## Price Advisor logic

After predicting the expected price, the project compares it with the actual listing price:

- if actual price is more than 15% below predicted price → `below_market`
- if actual price is within ±15% of predicted price → `market_price`
- if actual price is more than 15% above predicted price → `above_market`

This adds a simple product layer on top of the regression model.

## Main takeaways

- Real estate prices are strongly skewed, so log-transforming the target was useful.
- Area, location, region, rooms and building/object type were important features.
- Outlier filtering made the metrics much more stable.
- CatBoost worked better than simple baseline models on this tabular dataset.
- The final model can be used as a basic prototype for detecting underpriced and overpriced listings.

## Tech stack

- Python
- pandas
- NumPy
- matplotlib
- scikit-learn
- CatBoost
- Jupyter Notebook
- joblib
- Git / GitHub

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks in order:

```text
01_data_loading_cleaning.ipynb
02_eda.ipynb
03_modeling.ipynb
```

The original dataset should be placed in:

```text
data/raw/
```

The cleaned dataset, model and reports are saved to:

```text
data/processed/
models/
reports/
```