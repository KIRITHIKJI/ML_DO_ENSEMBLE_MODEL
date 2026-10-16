# Step 3: baselines and ML models (validation only)

- Trained on **2023**, scored on **2024**. The 2025 test split was not read by any script in this step.
- Horizons: [6, 12, 24, 48] hours ahead. Classes: Low < 5.0 mg/L, Medium 5.0-7.0, Good > 7.0.
- Seed 42 everywhere. Settings are in config.yaml under `models:`; they are starting values, not tuned.

## Pass criteria (fixed in config.yaml before any results were seen)

- **Regression:** `skill_vs_persistence > 0.0`, i.e. a lower RMSE than the naive "DO stays as it is" forecast.
- **Classification:** macro F1 **higher** than persistence AND Low-class recall **at least as high** as persistence. Catching the low-oxygen hours is the point of the system, so a better overall score with fewer Low hours caught does not count.
- A model goes forward to the ensemble if it passes at **3 of 4** horizons.

## Regression, 24 hours ahead

| model | R2 | RMSE | MAE | NSE | skill vs persistence | pass? |
|---|---|---|---|---|---|---|
| persistence | 0.9370 | 0.7362 | 0.4528 | 0.9370 | - | (baseline) |
| linear | 0.9467 | 0.6774 | 0.4247 | 0.9467 | +0.080 | PASS |
| random_forest | 0.9486 | 0.6653 | 0.4121 | 0.9486 | +0.096 | PASS |
| xgboost | 0.9486 | 0.6654 | 0.4126 | 0.9486 | +0.096 | PASS |
| lightgbm | 0.9497 | 0.6581 | 0.4089 | 0.9497 | +0.106 | PASS |

## Classification, 24 hours ahead

| model | accuracy | balanced acc | macro F1 | ROC-AUC | Low recall | Low precision | pass? |
|---|---|---|---|---|---|---|---|
| persistence | 0.900 | 0.879 | 0.879 | - | 0.893 | 0.893 | (baseline) |
| linear | 0.889 | 0.884 | 0.872 | 0.973 | 0.915 | 0.872 | fail |
| random_forest | 0.907 | 0.896 | 0.891 | 0.981 | 0.916 | 0.899 | PASS |
| xgboost | 0.908 | 0.898 | 0.892 | 0.982 | 0.914 | 0.903 | PASS |
| lightgbm | 0.906 | 0.897 | 0.891 | 0.981 | 0.913 | 0.903 | PASS |

## All horizons, compact

### Regression: RMSE (mg/L) and skill vs persistence

| model | 6 h RMSE | 12 h RMSE | 24 h RMSE | 48 h RMSE | 6 h skill | 12 h skill | 24 h skill | 48 h skill | passed |
|---|---|---|---|---|---|---|---|---|---|
| persistence | 1.462 | 1.671 | 0.736 | 1.022 | - | - | - | - | (baseline) |
| linear | 0.615 | 0.639 | 0.677 | 0.901 | +0.579 | +0.617 | +0.080 | +0.119 | 4/4 |
| random_forest | 0.613 | 0.619 | 0.665 | 0.879 | +0.580 | +0.630 | +0.096 | +0.140 | 4/4 |
| xgboost | 0.540 | 0.595 | 0.665 | 0.873 | +0.630 | +0.644 | +0.096 | +0.146 | 4/4 |
| lightgbm | 0.531 | 0.593 | 0.658 | 0.863 | +0.637 | +0.645 | +0.106 | +0.156 | 4/4 |

### Classification: macro F1 and Low recall

| model | 6 h macroF1 | 12 h macroF1 | 24 h macroF1 | 48 h macroF1 | 6 h Low recall | 12 h Low recall | 24 h Low recall | 48 h Low recall | passed |
|---|---|---|---|---|---|---|---|---|---|
| persistence | 0.749 | 0.739 | 0.879 | 0.826 | 0.735 | 0.745 | 0.893 | 0.841 | (baseline) |
| linear | 0.888 | 0.881 | 0.872 | 0.828 | 0.926 | 0.923 | 0.915 | 0.881 | 3/4 |
| random_forest | 0.896 | 0.898 | 0.891 | 0.849 | 0.923 | 0.918 | 0.916 | 0.884 | 4/4 |
| xgboost | 0.909 | 0.904 | 0.892 | 0.852 | 0.921 | 0.916 | 0.914 | 0.881 | 4/4 |
| lightgbm | 0.909 | 0.903 | 0.891 | 0.849 | 0.918 | 0.916 | 0.913 | 0.878 | 4/4 |

## Who goes forward to the ensemble

| model | task | horizons passed | goes forward? |
|---|---|---|---|
| lightgbm | classification | 4/4 | YES |
| random_forest | classification | 4/4 | YES |
| xgboost | classification | 4/4 | YES |
| linear | classification | 3/4 | YES |
| lightgbm | regression | 4/4 | YES |
| linear | regression | 4/4 | YES |
| random_forest | regression | 4/4 | YES |
| xgboost | regression | 4/4 | YES |

### Where models fail, and why

| model | task | horizon | reason |
|---|---|---|---|
| linear | classification | 24 h | macroF1 0.872 <= persistence 0.879 |

## Training time (seconds, one fit per horizon)

| model | reg 6h | reg 12h | reg 24h | reg 48h | cls 6h | cls 12h | cls 24h | cls 48h | total (min) |
|---|---|---|---|---|---|---|---|---|---|
| lightgbm | 36.8 | 18.1 | 14.1 | 12.0 | 54.4 | 44.2 | 34.4 | 31.6 | 4.1 |
| linear | 11.9 | 11.6 | 11.0 | 11.1 | 105.4 | 75.8 | 87.1 | 97.7 | 6.9 |
| random_forest | 231.1 | 216.3 | 244.5 | 249.8 | 226.1 | 242.2 | 273.6 | 270.3 | 32.6 |
| xgboost | 216.3 | 145.7 | 61.9 | 59.7 | 215.4 | 191.8 | 122.4 | 102.2 | 18.6 |

## Plots

- `results\val\plots\rmse_vs_horizon.png` - RMSE against horizon, all models
- `results\val\plots\confusion_persistence_h24.png`
- `results\val\plots\confusion_linear_h24.png`
- `results\val\plots\confusion_random_forest_h24.png`
- `results\val\plots\confusion_xgboost_h24.png`
- `results\val\plots\confusion_lightgbm_h24.png`

## Honest notes

- **Persistence is strong at 24 h and weak at 6 h.** DO has a daily cycle, so "the same as now" is a good guess exactly one day ahead and a poor one six hours ahead. That is why the skill numbers jump about between horizons, and why beating persistence at 24 h is the hardest test here.
- **XGBoost and LightGBM pick their number of trees by watching the 2024 validation set** (early stopping, 50 rounds). That is a legitimate use of validation, but it makes their validation scores slightly optimistic next to the Random Forest, which never looks at 2024. Only the 2025 test set settles it.
- Feature importances are each library's own measure (impurity decrease for the forest, gain for the boosters), so they are comparable within a model and not across models.
- Nothing here was tuned. These are the starting settings from config.yaml.
