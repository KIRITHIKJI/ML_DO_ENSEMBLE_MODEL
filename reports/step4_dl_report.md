# Step 4: BiLSTM and CNN-LSTM (validation only)

- Trained on **2023**, scored on **2024**. The 2025 test split was not read by any script in this step.
- torch 2.2.2+cu121, CUDA 12.1, device: **NVIDIA GeForce RTX 3050 6GB Laptop GPU**
- Both networks are multi-task: one model produces all 4 horizons for both tasks at once.
- The regression head predicts the **change** in DO in scaled units; DO(t) is added back before scoring, so the numbers below are directly comparable with step 3.
- Input is the raw hourly history (last 48 h of 6 water variables, their missing flags, DO saturation, 8 weather and 8 air variables, time sin/cos), not the 211 engineered features the tree models use.
- Rows are exactly the rows of `data/processed/val.parquet`, so every model in the tables is scored on the same hours.

## How the deep models trained

| model | epochs run | best epoch | min / epoch | total min | peak GPU GB |
|---|---|---|---|---|---|
| bilstm | 7 | 2 | 0.4 | 2.6 | 0.28 |
| cnn_lstm | 7 | 2 | 0.3 | 2.1 | 0.14 |

Training curves: `results/val/plots/bilstm_training_curve.png`, `results/val/plots/cnn_lstm_training_curve.png`

## Regression, 24 hours ahead - all models

| model | R2 | RMSE | MAE | NSE | skill vs persistence | pass? |
|---|---|---|---|---|---|---|
| persistence | 0.9370 | 0.7362 | 0.4528 | 0.9370 | - | (baseline) |
| linear | 0.9467 | 0.6774 | 0.4247 | 0.9467 | +0.080 | PASS |
| random_forest | 0.9486 | 0.6653 | 0.4121 | 0.9486 | +0.096 | PASS |
| xgboost | 0.9486 | 0.6654 | 0.4126 | 0.9486 | +0.096 | PASS |
| lightgbm | 0.9497 | 0.6581 | 0.4089 | 0.9497 | +0.106 | PASS |
| bilstm | 0.9432 | 0.6991 | 0.4396 | 0.9432 | +0.050 | PASS |
| cnn_lstm | 0.9414 | 0.7098 | 0.4452 | 0.9414 | +0.036 | PASS |

## Classification, 24 hours ahead - all models

| model | accuracy | balanced acc | macro F1 | ROC-AUC | Low recall | Low precision | pass? |
|---|---|---|---|---|---|---|---|
| persistence | 0.900 | 0.879 | 0.879 | - | 0.893 | 0.893 | (baseline) |
| linear | 0.889 | 0.884 | 0.872 | 0.973 | 0.915 | 0.872 | fail |
| random_forest | 0.907 | 0.896 | 0.891 | 0.981 | 0.916 | 0.899 | PASS |
| xgboost | 0.908 | 0.898 | 0.892 | 0.982 | 0.914 | 0.903 | PASS |
| lightgbm | 0.906 | 0.897 | 0.891 | 0.981 | 0.913 | 0.903 | PASS |
| bilstm | 0.889 | 0.879 | 0.872 | 0.973 | 0.901 | 0.887 | fail |
| cnn_lstm | 0.894 | 0.883 | 0.876 | 0.976 | 0.910 | 0.883 | fail |

## All horizons, all models

### Regression: RMSE (mg/L) and skill vs persistence

| model | 6 h RMSE | 12 h RMSE | 24 h RMSE | 48 h RMSE | 6 h skill | 12 h skill | 24 h skill | 48 h skill | passed |
|---|---|---|---|---|---|---|---|---|---|
| persistence | 1.462 | 1.671 | 0.736 | 1.022 | - | - | - | - | (baseline) |
| linear | 0.615 | 0.639 | 0.677 | 0.901 | +0.579 | +0.617 | +0.080 | +0.119 | 4/4 |
| random_forest | 0.613 | 0.619 | 0.665 | 0.879 | +0.580 | +0.630 | +0.096 | +0.140 | 4/4 |
| xgboost | 0.540 | 0.595 | 0.665 | 0.873 | +0.630 | +0.644 | +0.096 | +0.146 | 4/4 |
| lightgbm | 0.531 | 0.593 | 0.658 | 0.863 | +0.637 | +0.645 | +0.106 | +0.156 | 4/4 |
| bilstm | 0.662 | 0.718 | 0.699 | 0.932 | +0.547 | +0.570 | +0.050 | +0.088 | 4/4 |
| cnn_lstm | 0.708 | 0.728 | 0.710 | 0.957 | +0.515 | +0.564 | +0.036 | +0.063 | 4/4 |

### Classification: macro F1 and Low recall

| model | 6 h macroF1 | 12 h macroF1 | 24 h macroF1 | 48 h macroF1 | 6 h Low recall | 12 h Low recall | 24 h Low recall | 48 h Low recall | passed |
|---|---|---|---|---|---|---|---|---|---|
| persistence | 0.749 | 0.739 | 0.879 | 0.826 | 0.735 | 0.745 | 0.893 | 0.841 | (baseline) |
| linear | 0.888 | 0.881 | 0.872 | 0.828 | 0.926 | 0.923 | 0.915 | 0.881 | 3/4 |
| random_forest | 0.896 | 0.898 | 0.891 | 0.849 | 0.923 | 0.918 | 0.916 | 0.884 | 4/4 |
| xgboost | 0.909 | 0.904 | 0.892 | 0.852 | 0.921 | 0.916 | 0.914 | 0.881 | 4/4 |
| lightgbm | 0.909 | 0.903 | 0.891 | 0.849 | 0.918 | 0.916 | 0.913 | 0.878 | 4/4 |
| bilstm | 0.873 | 0.870 | 0.872 | 0.836 | 0.901 | 0.878 | 0.901 | 0.880 | 3/4 |
| cnn_lstm | 0.880 | 0.876 | 0.876 | 0.840 | 0.909 | 0.913 | 0.910 | 0.888 | 3/4 |

## Who goes forward to the ensemble

A model needs to pass at 3 of 4 horizons.

| model | task | horizons passed | goes forward? |
|---|---|---|---|
| lightgbm | classification | 4/4 | YES |
| random_forest | classification | 4/4 | YES |
| xgboost | classification | 4/4 | YES |
| bilstm | classification | 3/4 | YES |
| cnn_lstm | classification | 3/4 | YES |
| linear | classification | 3/4 | YES |
| bilstm | regression | 4/4 | YES |
| cnn_lstm | regression | 4/4 | YES |
| lightgbm | regression | 4/4 | YES |
| linear | regression | 4/4 | YES |
| random_forest | regression | 4/4 | YES |
| xgboost | regression | 4/4 | YES |

### Where models fail, and why

| model | task | horizon | reason |
|---|---|---|---|
| linear | classification | 24 h | macroF1 0.872 <= persistence 0.879 |
| bilstm | classification | 24 h | macroF1 0.872 <= persistence 0.879 |
| cnn_lstm | classification | 24 h | macroF1 0.876 <= persistence 0.879 |

## Honest notes

- **GPU LSTM training is not bit-for-bit reproducible.** All the seeds are fixed (python, numpy, torch, cuda) and `cudnn.benchmark` is off, but cuDNN's recurrent kernels have no deterministic implementation, so `torch.use_deterministic_algorithms(True, warn_only=True)` warns rather than enforces. Two runs of the same command can differ in the last decimal. Everything else in the project is exactly reproducible.
- **The deep models watch 2024 too.** Early stopping and the learning-rate schedule both read the validation loss, the same arrangement as XGBoost and LightGBM in step 3. The Random Forest is the only model that never sees 2024, so it is the most conservative number in these tables.
- The deep models see raw hourly history, the tree models see 211 engineered features. That is the comparison on purpose: it asks whether a network can learn from the raw series what the feature engineering spells out.
- Nothing here was tuned. These are the starting settings in config.yaml.
