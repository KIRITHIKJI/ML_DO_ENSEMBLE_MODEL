# Step 5: tuning the deep models (validation only)

- Trained on **2023**, scored on **2024**. The 2025 test split was not read by any script in this step.
- torch 2.2.2+cu121, CUDA 12.1, device: **NVIDIA GeForce RTX 3050 6GB Laptop GPU**
- Step 4 left both deep models behind every tree model and overfitting by epoch 3. Four changes were tried, each added on top of the last, and the winner was picked by **validation loss** - the quantity the networks minimise - rather than by any single metric.
- The chosen configuration of each model was then retrained with seeds [42, 43, 44] and the three predictions averaged. That average is what the tables below score.

## The search: 5 variants x 2 models

| variant | what it changes |
|---|---|
| `step4` | step-4 reference |
| `reg` | (a) regularisation |
| `reg_hybrid` | (a)+(b) hybrid input |
| `reg_hybrid_72h` | (a)+(b)+(c) 72 h window |
| `reg_hybrid_72h_small` | (a)+(b)+(c)+(d) hidden 32 |

### Results (single seed 42, chosen row marked)

| model | variant | params | epochs | best epoch | val loss | 24 h RMSE | 24 h macro F1 | 24 h Low recall | min | chosen |
|---|---|---|---|---|---|---|---|---|---|---|
| bilstm | `step4` | 160,064 | 7 | 2 | 0.3686 | 0.6991 | 0.8720 | 0.9012 | 2.9 |  |
| cnn_lstm | `step4` | 48,640 | 9 | 4 | 0.3445 | 0.6931 | 0.8739 | 0.8996 | 2.9 |  |
| bilstm | `reg` | 160,064 | 6 | 2 | 0.3768 | 0.7120 | 0.8705 | 0.8926 | 2.8 |  |
| cnn_lstm | `reg` | 48,640 | 8 | 4 | 0.3511 | 0.7092 | 0.8730 | 0.9005 | 3.0 |  |
| bilstm | `reg_hybrid` | 177,728 | 6 | 2 | 0.3588 | 0.6809 | 0.8719 | 0.8918 | 2.8 |  |
| cnn_lstm | `reg_hybrid` | 66,304 | 7 | 3 | 0.3333 | 0.6810 | 0.8767 | 0.9017 | 3.0 |  |
| bilstm | `reg_hybrid_72h` | 177,728 | 6 | 2 | 0.3518 | 0.6803 | 0.8717 | 0.8843 | 3.4 |  |
| cnn_lstm | `reg_hybrid_72h` | 66,304 | 7 | 3 | 0.3310 | 0.6817 | 0.8745 | 0.9103 | 3.5 |  |
| bilstm | `reg_hybrid_72h_small` | 65,856 | 7 | 3 | 0.3447 | 0.6811 | 0.8686 | 0.8980 | 3.9 | **chosen** |
| cnn_lstm | `reg_hybrid_72h_small` | 43,520 | 8 | 4 | 0.3287 | 0.6787 | 0.8768 | 0.9126 | 4.1 | **chosen** |

## Before and after, 24 hours ahead

| model |  | RMSE | R2 | macro F1 | Low recall | regression | classification |
|---|---|---|---|---|---|---|---|
| bilstm | step 4 | 0.6991 | 0.9432 | 0.8720 | 0.9012 | PASS | fail |
| bilstm | step 5 | 0.6800 | 0.9463 | 0.8730 | 0.9050 | PASS | fail |
| cnn_lstm | step 4 | 0.7098 | 0.9414 | 0.8764 | 0.9103 | PASS | fail |
| cnn_lstm | step 5 | 0.6777 | 0.9466 | 0.8788 | 0.9102 | PASS | fail |

## How much is luck? The spread over three seeds

| model | variant | seeds | mean val loss | 24 h RMSE (sd) | 24 h macro F1 (sd) | averaged RMSE | averaged macro F1 |
|---|---|---|---|---|---|---|---|
| bilstm | `reg_hybrid_72h_small` | 42,43,44 | 0.3437 | 0.6827 (± 0.0023) | 0.8699 (± 0.0014) | **0.6800** | **0.8730** |
| cnn_lstm | `reg_hybrid_72h_small` | 42,43,44 | 0.3321 | 0.6804 (± 0.0017) | 0.8742 (± 0.0041) | **0.6777** | **0.8788** |

The last two columns are the averaged prediction, which is the final model. It is better than the average single seed, which is the point of averaging.


## Regression, 24 hours ahead - all models

| model | R2 | RMSE | MAE | NSE | skill vs persistence | pass? |
|---|---|---|---|---|---|---|
| persistence | 0.9370 | 0.7362 | 0.4528 | 0.9370 | - | (baseline) |
| linear | 0.9467 | 0.6774 | 0.4247 | 0.9467 | +0.080 | PASS |
| random_forest | 0.9486 | 0.6653 | 0.4121 | 0.9486 | +0.096 | PASS |
| xgboost | 0.9486 | 0.6654 | 0.4126 | 0.9486 | +0.096 | PASS |
| lightgbm | 0.9497 | 0.6581 | 0.4089 | 0.9497 | +0.106 | PASS |
| bilstm | 0.9463 | 0.6800 | 0.4205 | 0.9463 | +0.076 | PASS |
| cnn_lstm | 0.9466 | 0.6777 | 0.4193 | 0.9466 | +0.079 | PASS |

## Classification, 24 hours ahead - all models

| model | accuracy | balanced acc | macro F1 | ROC-AUC | Low recall | Low precision | pass? |
|---|---|---|---|---|---|---|---|
| persistence | 0.900 | 0.879 | 0.879 | - | 0.893 | 0.893 | (baseline) |
| linear | 0.889 | 0.884 | 0.872 | 0.973 | 0.915 | 0.872 | fail |
| random_forest | 0.907 | 0.896 | 0.891 | 0.981 | 0.916 | 0.899 | PASS |
| xgboost | 0.908 | 0.898 | 0.892 | 0.982 | 0.914 | 0.903 | PASS |
| lightgbm | 0.906 | 0.897 | 0.891 | 0.981 | 0.913 | 0.903 | PASS |
| bilstm | 0.889 | 0.882 | 0.873 | 0.976 | 0.905 | 0.888 | fail |
| cnn_lstm | 0.895 | 0.888 | 0.879 | 0.978 | 0.910 | 0.891 | fail |

## All horizons, all models

### Regression: RMSE (mg/L) and skill vs persistence

| model | 6 h RMSE | 12 h RMSE | 24 h RMSE | 48 h RMSE | 6 h skill | 12 h skill | 24 h skill | 48 h skill | passed |
|---|---|---|---|---|---|---|---|---|---|
| persistence | 1.462 | 1.671 | 0.736 | 1.022 | - | - | - | - | (baseline) |
| linear | 0.615 | 0.639 | 0.677 | 0.901 | +0.579 | +0.617 | +0.080 | +0.119 | 4/4 |
| random_forest | 0.613 | 0.619 | 0.665 | 0.879 | +0.580 | +0.630 | +0.096 | +0.140 | 4/4 |
| xgboost | 0.540 | 0.595 | 0.665 | 0.873 | +0.630 | +0.644 | +0.096 | +0.146 | 4/4 |
| lightgbm | 0.531 | 0.593 | 0.658 | 0.863 | +0.637 | +0.645 | +0.106 | +0.156 | 4/4 |
| bilstm | 0.645 | 0.657 | 0.680 | 0.907 | +0.559 | +0.607 | +0.076 | +0.113 | 4/4 |
| cnn_lstm | 0.641 | 0.660 | 0.678 | 0.901 | +0.561 | +0.605 | +0.079 | +0.118 | 4/4 |

### Classification: macro F1 and Low recall

| model | 6 h macroF1 | 12 h macroF1 | 24 h macroF1 | 48 h macroF1 | 6 h Low recall | 12 h Low recall | 24 h Low recall | 48 h Low recall | passed |
|---|---|---|---|---|---|---|---|---|---|
| persistence | 0.749 | 0.739 | 0.879 | 0.826 | 0.735 | 0.745 | 0.893 | 0.841 | (baseline) |
| linear | 0.888 | 0.881 | 0.872 | 0.828 | 0.926 | 0.923 | 0.915 | 0.881 | 3/4 |
| random_forest | 0.896 | 0.898 | 0.891 | 0.849 | 0.923 | 0.918 | 0.916 | 0.884 | 4/4 |
| xgboost | 0.909 | 0.904 | 0.892 | 0.852 | 0.921 | 0.916 | 0.914 | 0.881 | 4/4 |
| lightgbm | 0.909 | 0.903 | 0.891 | 0.849 | 0.918 | 0.916 | 0.913 | 0.878 | 4/4 |
| bilstm | 0.883 | 0.878 | 0.873 | 0.840 | 0.911 | 0.906 | 0.905 | 0.877 | 3/4 |
| cnn_lstm | 0.890 | 0.882 | 0.879 | 0.844 | 0.917 | 0.908 | 0.910 | 0.887 | 3/4 |

## Who goes forward to the ensemble

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

### Where models still fail

| model | task | horizon | reason |
|---|---|---|---|
| linear | classification | 24 h | macroF1 0.872 <= persistence 0.879 |
| bilstm | classification | 24 h | macroF1 0.873 <= persistence 0.879 |
| cnn_lstm | classification | 24 h | macroF1 0.879 <= persistence 0.879 |

## Plots

- `results/val/plots/rmse_vs_horizon.png`
- `results/val/plots/bilstm_training_curve.png`
- `results/val/plots/cnn_lstm_training_curve.png`
- `results\val\plots\confusion_persistence_h24.png`
- `results\val\plots\confusion_linear_h24.png`
- `results\val\plots\confusion_random_forest_h24.png`
- `results\val\plots\confusion_xgboost_h24.png`
- `results\val\plots\confusion_lightgbm_h24.png`
- `results\val\plots\confusion_bilstm_h24.png`
- `results\val\plots\confusion_cnn_lstm_h24.png`

## What helped and what did not

For **bilstm**, validation loss against the step-4 reference (0.3686): `reg` +0.0082, `reg_hybrid` -0.0098, `reg_hybrid_72h` -0.0168, `reg_hybrid_72h_small` -0.0238. For **cnn_lstm**, validation loss against the step-4 reference (0.3445): `reg` +0.0065, `reg_hybrid` -0.0112, `reg_hybrid_72h` -0.0136, `reg_hybrid_72h_small` -0.0159.

**What helped: the hybrid input, and nothing else very much.** Handing the network the same 211 engineered features the tree models use is the one change that clearly moved the validation loss, and it moved the 24 h RMSE with it. That is the useful finding of this step: the gap in step 4 was mostly about the *inputs*, not the architecture. A recurrent network is supposed to learn lags and rolling statistics from the raw series, and on one year of training data it simply does not learn them as well as step 2 computes them.

**What did not help: regularisation on its own.** This is the opposite of what step 4's overfitting suggested. Dropout 0.3 with AdamW and a lower learning rate made the validation loss *worse* for the BiLSTM and roughly neutral for the CNN-LSTM. Early stopping was already handling the overfitting; slowing learning down on top of that mostly cost accuracy. The longer 72-hour window (c) gave a small gain - 48 hours already covers two daily cycles, so there is not much more to see - and halving the hidden size (d) helped slightly rather than hurting, which says these networks were larger than one year of data supports. Averaging three seeds is a small, free and reliable gain, and the seed spread above is the honest scale against which to read any of these differences: a gap smaller than the spread is noise.

## Honest notes

- **Everything here was chosen on 2024.** Five variants were compared on the validation year and the winner retrained on it; that is what a validation set is for, but it does mean these numbers are the optimistic end of what to expect. Only the 2025 test set, still untouched, settles it.
- **GPU LSTM training is still not bit-for-bit reproducible** (cuDNN has no deterministic recurrent kernel). The seed spread table is the honest measure of how much that matters: anything smaller than it is noise.
- The tree models were not retuned in this step, so the comparison is a tuned deep model against an untuned LightGBM. If the deep models still lose, that conclusion is safe; if they win narrowly, it is not.
