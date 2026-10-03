# Model selection and tuning results

Selected candidate: **`catboost_baseline`**
Selected configuration: `{"depth": 8, "iterations": 1200, "l2_leaf_reg": 5.0, "learning_rate": 0.05, "random_strength": 1.0}`
Validation MAE: **972,328 RUB**
Final temporal holdout MAE after refitting: **1,450,474 RUB**

## Search protocol

Candidates were ranked by MAE on the same chronological validation window. The final temporal holdout was evaluated only after candidate selection.
Split sizes: train 3,807,801, validation 973,532, holdout 689,208 listings.

Fine tuning was not enabled for this run. Use `--tune` to run the coarse CatBoost search followed by a local search around its leader.

## Validation leaderboard

| Rank | Candidate | Stage | MAE (RUB) | RMSE (RUB) | Selected |
|---:|---|---|---:|---:|:---:|
| 1 | `catboost_baseline` | baseline | 972,328 | 7,491,129 | yes |
| 2 | `blend_catboost_baseline_dummy_median_75_25` | ensemble | 1,260,544 | 7,856,713 |  |
| 3 | `blend_catboost_baseline_dummy_median_50_50` | ensemble | 1,742,100 | 8,515,241 |  |
| 4 | `blend_catboost_baseline_dummy_median_25_75` | ensemble | 2,293,614 | 9,405,380 |  |
| 5 | `dummy_median` | baseline | 2,868,843 | 10,468,214 |  |

## Interpretation

The selected candidate minimizes validation MAE across 6 model and blend candidates. Its holdout metrics are a later-period estimate of generalization, not part of the tuning decision.

Holdout RMSE: 7,609,150 RUB; MAPE: 1212.44%; R²: 0.5209.

These conclusions describe this dataset and these temporal windows. A fresh, later time split is needed before treating a small validation difference as a stable improvement.
