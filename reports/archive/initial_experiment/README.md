# Initial notebook experiment

These artifacts preserve the first modeling run before the pipeline was moved into reusable modules.

That experiment compared Dummy, Ridge, and CatBoost on one chronological split. It selected the winner using test-set MAE and removed listings with target-dependent price and price-per-square-metre thresholds before splitting. The reported values therefore are not directly comparable with the current validation-plus-holdout protocol and should not be presented as fresh holdout results.

| Model | MAE (RUB) | RMSE (RUB) | R2 | MdAPE | WAPE |
|---|---:|---:|---:|---:|---:|
| CatBoost | 1,292,954 | 3,847,274 | 0.764 | 16.75% | 22.57% |
| Dummy median | 3,400,781 | 8,441,422 | -0.137 | 42.27% | 59.37% |
| Ridge | 4,656,774 | 1,361,065,874 | -29,551.055 | 20.74% | 81.30% |

The Ridge RMSE is a useful warning from that first run: linear extrapolation in log-price space produced extreme errors after converting predictions back to RUB. The revised pipeline includes an additional later holdout and records its own validation and holdout rows separately.

The production-style pipeline now uses a validation period for model selection, keeps a later temporal holdout untouched until the final evaluation, and avoids target-dependent row filtering.
