# Final results - the 2025 test year

**Summary.** Thirty USGS river stations, three data sources (water-quality sensors, ERA5 weather, CAMS air quality), hourly, 2023-2025. Models were built on 2023, every choice was made on 2024, and the system was frozen - hyperparameters, ensemble, warning thresholds and file hashes all recorded in `FROZEN.md` and committed - **before any 2025 target was read**. The 2025 year was then scored once. Twenty-four hours ahead, the final system (a stacking ensemble over five models) predicts dissolved oxygen with **R2 0.951, RMSE 0.672 mg/L** (+11.1% better than assuming no change), classifies Low/Medium/Good with **macro F1 0.886** and, with the frozen warning rule, catches **49.6% of the hours when safe water turns dangerous** at a false-alarm rate of 1.50%. Of the 230 real low-oxygen events in 2025, **52% were warned about in advance**, with a median lead time of **20 hours**.


## 1. Regression, 24 hours ahead, 2025

| method | R2 | RMSE | MAE | NSE | skill vs persistence | danger-zone RMSE |
|---|---|---|---|---|---|---|
| persistence | 0.9375 | 0.7553 | 0.4569 | 0.9375 | - | 0.7263 |
| linear | 0.9471 | 0.6950 | 0.4294 | 0.9471 | +0.080 | 0.6553 |
| random_forest | 0.9475 | 0.6924 | 0.4189 | 0.9475 | +0.083 | 0.6339 |
| xgboost | 0.9471 | 0.6951 | 0.4207 | 0.9471 | +0.080 | 0.6297 |
| lightgbm | 0.9490 | 0.6822 | 0.4158 | 0.9490 | +0.097 | 0.6323 |
| bilstm | 0.9480 | 0.6887 | 0.4197 | 0.9480 | +0.088 | 0.6537 |
| cnn_lstm | 0.9484 | 0.6860 | 0.4159 | 0.9484 | +0.092 | 0.6535 |
| simple_average | 0.9503 | 0.6737 | 0.4086 | 0.9503 | +0.108 | 0.6311 |
| atae | 0.9513 | 0.6670 | 0.4041 | 0.9513 | +0.117 | 0.6291 |
| **stacking** | 0.9505 | 0.6718 | 0.4074 | 0.9505 | +0.111 | 0.6339 |

Danger-zone RMSE covers the rows whose true DO lies between 3.0 and 7.0 mg/L - the band where a river actually crosses into trouble.


## 2. Classification, 24 hours ahead, 2025 - with the frozen warning rule

| method | accuracy | balanced acc | macro F1 | ROC-AUC | Low recall | onset recall | false alarm | transition acc |
|---|---|---|---|---|---|---|---|---|
| persistence | 0.901 | 0.881 | 0.8806 | - | 0.886 | 0.000 | 0.0000 | 0.000 |
| linear | 0.887 | 0.885 | 0.8722 | 0.973 | 0.918 | 0.371 | 0.0125 | 0.316 |
| random_forest | 0.906 | 0.895 | 0.8875 | 0.982 | 0.936 | 0.450 | 0.0131 | 0.253 |
| xgboost | 0.905 | 0.895 | 0.8864 | 0.982 | 0.941 | 0.500 | 0.0156 | 0.302 |
| lightgbm | 0.905 | 0.896 | 0.8869 | 0.982 | 0.940 | 0.499 | 0.0156 | 0.319 |
| bilstm | 0.896 | 0.888 | 0.8795 | 0.978 | 0.914 | 0.424 | 0.0132 | 0.354 |
| cnn_lstm | 0.894 | 0.887 | 0.8753 | 0.979 | 0.935 | 0.522 | 0.0200 | 0.358 |
| simple_average | 0.903 | 0.896 | 0.8855 | 0.982 | 0.943 | 0.515 | 0.0161 | 0.314 |
| atae | 0.904 | 0.896 | 0.8859 | 0.982 | 0.942 | 0.510 | 0.0157 | 0.312 |
| **stacking** | 0.904 | 0.896 | 0.8860 | 0.982 | 0.941 | 0.496 | 0.0150 | 0.291 |

## 2. Classification, 24 hours ahead, 2025 - with argmax

| method | accuracy | balanced acc | macro F1 | ROC-AUC | Low recall | onset recall | false alarm | transition acc |
|---|---|---|---|---|---|---|---|---|
| persistence | 0.901 | 0.881 | 0.8806 | - | 0.886 | 0.000 | 0.0000 | 0.000 |
| linear | 0.888 | 0.886 | 0.8741 | 0.973 | 0.907 | 0.312 | 0.0092 | 0.311 |
| random_forest | 0.909 | 0.898 | 0.8928 | 0.982 | 0.906 | 0.251 | 0.0039 | 0.233 |
| xgboost | 0.909 | 0.899 | 0.8934 | 0.982 | 0.905 | 0.299 | 0.0053 | 0.294 |
| lightgbm | 0.909 | 0.899 | 0.8936 | 0.982 | 0.907 | 0.324 | 0.0059 | 0.315 |
| bilstm | 0.897 | 0.888 | 0.8810 | 0.978 | 0.895 | 0.348 | 0.0093 | 0.352 |
| cnn_lstm | 0.897 | 0.890 | 0.8804 | 0.979 | 0.918 | 0.422 | 0.0130 | 0.351 |
| simple_average | 0.908 | 0.899 | 0.8928 | 0.982 | 0.910 | 0.318 | 0.0057 | 0.302 |
| atae | 0.908 | 0.899 | 0.8930 | 0.982 | 0.910 | 0.314 | 0.0056 | 0.300 |
| **stacking** | 0.908 | 0.899 | 0.8930 | 0.982 | 0.908 | 0.285 | 0.0046 | 0.274 |

## 3. All horizons: the final system against the bar

| method | RMSE 6h | RMSE 12h | RMSE 24h | RMSE 48h | macroF1 6h | macroF1 12h | macroF1 24h | macroF1 48h | onset recall 6h | onset recall 12h | onset recall 24h | onset recall 48h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| persistence | 1.344 | 1.532 | 0.755 | 1.007 | 0.752 | 0.738 | 0.881 | 0.833 | 0.000 | 0.000 | 0.000 | 0.000 |
| lightgbm (best single, regression) | 0.529 | 0.603 | 0.682 | 0.875 | 0.905 | 0.899 | 0.887 | 0.858 | 0.867 | 0.858 | 0.499 | 0.445 |
| xgboost (best single, classification) | 0.539 | 0.615 | 0.695 | 0.878 | 0.905 | 0.896 | 0.886 | 0.857 | 0.864 | 0.860 | 0.500 | 0.473 |
| stacking | 0.514 | 0.588 | 0.672 | 0.853 | 0.905 | 0.897 | 0.886 | 0.856 | 0.861 | 0.861 | 0.496 | 0.485 |

### Is the final system really better?

1,000 station-week resamples of the 2025 rows. Negative RMSE favours stacking; positive macro F1 and onset recall favour stacking.

| stacking vs | horizon | metric | difference | 95% CI | verdict |
|---|---|---|---|---|---|
| persistence | 6 h | RMSE | -0.8301 | [-0.8829, -0.7813] | **stacking better** |
| persistence | 6 h | macro F1 | +0.1536 | [+0.1423, +0.1658] | **stacking better** |
| persistence | 6 h | onset recall | +0.8607 | [+0.8431, +0.8778] | **stacking better** |
| persistence | 12 h | RMSE | -0.9421 | [-1.0070, -0.8743] | **stacking better** |
| persistence | 12 h | macro F1 | +0.1589 | [+0.1451, +0.1720] | **stacking better** |
| persistence | 12 h | onset recall | +0.8603 | [+0.8383, +0.8788] | **stacking better** |
| persistence | 24 h | RMSE | -0.0836 | [-0.0949, -0.0700] | **stacking better** |
| persistence | 24 h | macro F1 | +0.0055 | [+0.0027, +0.0079] | **stacking better** |
| persistence | 24 h | onset recall | +0.4962 | [+0.4610, +0.5298] | **stacking better** |
| persistence | 48 h | RMSE | -0.1537 | [-0.1705, -0.1359] | **stacking better** |
| persistence | 48 h | macro F1 | +0.0237 | [+0.0194, +0.0282] | **stacking better** |
| persistence | 48 h | onset recall | +0.4850 | [+0.4489, +0.5231] | **stacking better** |
| best single model | 6 h | RMSE | -0.0150 | [-0.0209, -0.0103] | **stacking better** |
| best single model | 6 h | macro F1 | -0.0002 | [-0.0011, +0.0008] | no significant difference |
| best single model | 6 h | onset recall | -0.0027 | [-0.0076, +0.0015] | no significant difference |
| best single model | 12 h | RMSE | -0.0152 | [-0.0199, -0.0112] | **stacking better** |
| best single model | 12 h | macro F1 | +0.0008 | [-0.0001, +0.0018] | no significant difference |
| best single model | 12 h | onset recall | +0.0011 | [-0.0030, +0.0054] | no significant difference |
| best single model | 24 h | RMSE | -0.0104 | [-0.0165, -0.0051] | **stacking better** |
| best single model | 24 h | macro F1 | -0.0004 | [-0.0014, +0.0006] | no significant difference |
| best single model | 24 h | onset recall | -0.0040 | [-0.0153, +0.0069] | no significant difference |
| best single model | 48 h | RMSE | -0.0217 | [-0.0298, -0.0148] | **stacking better** |
| best single model | 48 h | macro F1 | -0.0005 | [-0.0018, +0.0008] | no significant difference |
| best single model | 48 h | onset recall | +0.0120 | [+0.0016, +0.0229] | **stacking better** |

## 4. What the warning rule is worth

Argmax answers *which class is most likely*. A warning system asks *is the chance of Low high enough to act on*. The rule issues a warning when P(Low) reaches a threshold tau, frozen before the test from the out-of-fold 2024 probabilities (stacking: tau = 0.24 at 24 h).

| horizon | onset recall (argmax) | onset recall (rule) | change [95% CI] | false alarm (argmax) | false alarm (rule) | change [95% CI] |
|---|---|---|---|---|---|---|
| 6 h | 0.800 | 0.861 | +0.061 [+0.054, +0.068] | 0.0103 | 0.0175 | +0.0072 [+0.0063, +0.0081] |
| 12 h | 0.795 | 0.861 | +0.066 [+0.056, +0.075] | 0.0096 | 0.0173 | +0.0077 [+0.0067, +0.0088] |
| 24 h | 0.285 | 0.496 | +0.211 [+0.194, +0.229] | 0.0046 | 0.0150 | +0.0104 [+0.0090, +0.0119] |
| 48 h | 0.380 | 0.485 | +0.106 [+0.094, +0.117] | 0.0070 | 0.0133 | +0.0062 [+0.0052, +0.0072] |

The rule was tuned against a 1.8% internal budget so that the 2% promise would survive contact with unseen data. On 2025 it delivers **1.50%** at 24 h - inside the promise.


## 5. Lead time: how much warning

A low-oxygen event starts when DO drops below 5.0 mg/L after at least 24 h at or above it. A warning counts if the 24-hour-ahead rule fired at any point in the 24 h before that moment.

|  | value |
|---|---|
| low-DO events in 2025 | 230 |
| warned in advance | 119 (51.7%) |
| median lead time | 20 h |
| mean lead time | 14.6 h |
| stations with at least one event | 30 of 30 |

## 6. What each data source is worth

LightGBM alone, retrained on 2023+2024 with the frozen settings, three input sets. **W** = water + time + station, **W+M** adds weather, **W+M+A** adds air quality. The thresholds were frozen in the same step as everything else.

| input set | features | RMSE 6h | RMSE 12h | RMSE 24h | RMSE 48h | onset recall 6h | onset recall 12h | onset recall 24h | onset recall 48h |
|---|---|---|---|---|---|---|---|---|---|
| W | 156 | 0.535 | 0.618 | 0.700 | 0.895 | 0.865 | 0.859 | 0.487 | 0.440 |
| W+M | 195 | 0.531 | 0.604 | 0.683 | 0.874 | 0.867 | 0.856 | 0.488 | 0.452 |
| W+M+A | 211 | 0.531 | 0.604 | 0.681 | 0.872 | 0.866 | 0.861 | 0.494 | 0.461 |

## 7. Per station, 24 hours ahead (stacking, warning rule)

Sorted by how many onsets each station actually had.

| station | rows | RMSE | macro F1 | onset recall | false alarm | onsets |
|---|---|---|---|---|---|---|
| 02110701 | 7,714 | 0.987 | 0.811 | 0.556 | 0.0692 | 475 |
| 02276998 | 8,203 | 0.845 | 0.782 | 0.490 | 0.0434 | 457 |
| 08048000 | 8,204 | 1.351 | 0.699 | 0.248 | 0.0184 | 451 |
| 02198955 | 8,697 | 0.428 | 0.891 | 0.736 | 0.0338 | 439 |
| 02198920 | 8,721 | 0.422 | 0.899 | 0.808 | 0.0321 | 432 |
| 02292900 | 8,215 | 0.896 | 0.817 | 0.448 | 0.0498 | 299 |
| 01484080 | 8,736 | 0.625 | 0.850 | 0.622 | 0.0225 | 299 |
| 02264051 | 8,735 | 0.544 | 0.826 | 0.216 | 0.0054 | 278 |
| 02198950 | 7,964 | 0.389 | 0.880 | 0.737 | 0.0271 | 266 |
| 03430200 | 8,681 | 0.888 | 0.754 | 0.133 | 0.0131 | 248 |
| 02239501 | 8,736 | 0.472 | 0.835 | 0.342 | 0.0445 | 243 |
| 03353415 | 8,349 | 1.094 | 0.864 | 0.696 | 0.0281 | 230 |
| 01302020 | 8,727 | 0.720 | 0.794 | 0.169 | 0.0047 | 219 |
| 02110500 | 8,090 | 0.279 | 0.937 | 0.665 | 0.0321 | 218 |
| 02322700 | 8,214 | 0.230 | 0.840 | 0.551 | 0.0572 | 198 |
| 05474500 | 6,821 | 0.919 | 0.886 | 0.222 | 0.0177 | 167 |
| 02110550 | 8,701 | 0.237 | 0.952 | 0.608 | 0.0205 | 148 |
| 02135000 | 8,347 | 0.222 | 0.904 | 0.906 | 0.0246 | 117 |
| 07049000 | 8,590 | 0.707 | 0.713 | 0.184 | 0.0051 | 114 |
| 02110704 | 8,722 | 0.225 | 0.972 | 0.821 | 0.0095 | 78 |
| 07048600 | 8,390 | 0.446 | 0.863 | 0.672 | 0.0059 | 61 |
| 03431083 | 8,568 | 0.995 | 0.613 | 0.000 | 0.0014 | 53 |
| 01484525 | 8,732 | 0.906 | 0.529 | 0.044 | 0.0012 | 45 |
| 03290500 | 8,243 | 0.579 | 0.791 | 0.250 | 0.0057 | 36 |
| 040851385 | 8,734 | 0.547 | 0.652 | 0.152 | 0.0023 | 33 |
| 11273400 | 8,732 | 0.290 | 0.852 | 0.370 | 0.0022 | 27 |
| 03353200 | 8,550 | 0.722 | 0.767 | 0.318 | 0.0020 | 22 |
| 03612600 | 8,510 | 0.346 | 0.829 | 0.375 | 0.0001 | 8 |
| 07239450 | 7,319 | 0.683 | 0.571 | 0.000 | 0.0001 | 4 |
| 02198840 | 8,726 | 0.160 | 0.648 | 0.000 | 0.0002 | 3 |

## 8. Did the 2024 result hold up in 2025?

| horizon | RMSE 2024 | RMSE 2025 | macro F1 2024 | macro F1 2025 | onset recall 2024 | onset recall 2025 |
|---|---|---|---|---|---|---|
| 6 h | 0.522 | 0.514 | 0.893 | 0.905 | 0.904 | 0.861 |
| 12 h | 0.581 | 0.588 | 0.885 | 0.897 | 0.891 | 0.861 |
| 24 h | 0.651 | 0.672 | 0.876 | 0.886 | 0.594 | 0.496 |
| 48 h | 0.853 | 0.853 | 0.841 | 0.856 | 0.555 | 0.485 |

*2024 numbers are the out-of-fold cross-fitted ones from step 7, which is the fairest in-sample comparison available; 2024's thresholds were chosen on 2024 data, so a small drop in 2025 is expected rather than alarming.*


## 9. Robustness: how much did the extra training year help?

The headline numbers use members retrained on 2023+2024 (stage 2). The same chain with the original 2023-only members gives:

| horizon | RMSE (2023 only) | RMSE (2023+2024) | onset recall (2023 only) | onset recall (2023+2024) |
|---|---|---|---|---|
| 6 h | 0.546 | 0.514 | 0.845 | 0.861 |
| 12 h | 0.609 | 0.588 | 0.852 | 0.861 |
| 24 h | 0.687 | 0.672 | 0.495 | 0.496 |
| 48 h | 0.874 | 0.853 | 0.487 | 0.485 |

## Plots

- `results\test\plots\test_station_02198920_2025-05.png`
- `results\test\plots\test_station_02198955_2025-10.png`
- `results\test\plots\test_confusion_h24.png`
- `results\test\plots\test_onset_vs_false_alarm.png`
- `results\test\plots\test_rmse_vs_horizon.png`
- `results\test\plots\test_ablation.png`

## Limitations

- **The members' hyperparameters were chosen on 2024.** The 2025 test is honest about the *system*, but the models inside it were tuned on the validation year, so 2025 is a fair test of the pipeline rather than of each model.
- **The stacking meta-model was fitted on 2024 predictions of 2023-only members**, then applied on top of members retrained on 2023+2024. The inputs it sees at test time come from slightly better-trained models than the ones it learned to blend. Section 9 shows the effect is small.
- **One station stops reporting on 2025-11-19**, so the last six weeks of the test year have 29 stations rather than 30.
- **US rivers only**, 30 stations, three years. Nothing here says how the system behaves on a river it has never seen - every station in the test appears in training.
- **The deep models are the weakest members.** The BiLSTM and CNN-LSTM were tuned in step 5 and still trail the gradient-boosted trees; they earn their place in the ensemble by being different, not by being better.
- **Two ensemble ideas were tested and rejected.** ATAE (adaptive threshold-aware weighting, step 6) matched a plain average and was not worth its four tuned parameters. CAE (context-aware meta-model, step 7) failed its pre-set decision rule outright - worse onset recall and worse RMSE than stacking. Both are still scored here for the record.
- **The gains are concentrated in the decision rule, not the ensemble.** Most of the onset-detection performance comes from choosing a probability threshold properly; the difference between combining five models cleverly and averaging them is small.
