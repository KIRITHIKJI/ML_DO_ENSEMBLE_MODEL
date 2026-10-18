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


## 2a. Classification, 24 hours ahead, 2025 - with the frozen warning rule

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

## 2b. Classification, 24 hours ahead, 2025 - with argmax

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

## Stage-3 models (what the app uses)

**Every number in this report comes from the stage-2 models, trained on 2023+2024.** Those are the ones that were scored on 2025.

A second set, in `models/final/`, was retrained afterwards on **2023 + 2024 + 2025** so the app can forecast using every hour ever recorded. At the time this report was first written those models were **untested and untestable**: they had seen the test year, so there was no data left to score them on. None of the 2025 numbers above comes from them.

**They have since been tested.** 2026 arrived, which is a year `models/final` did not train on, and the criteria were pre-registered before any 2026 value was read. See **Deployed models on 2026** below; nothing in sections 1-9 changed.

They were checked only for sanity: every file loads and predicts without error, and their 2025 predictions still agree closely with the stage-2 models'. Across 40 model-task-horizon comparisons the correlation runs from **0.9790 to 0.9998** (median 0.9923), with a median absolute difference of 0.0400 and a worst of 0.2226. Nothing fell below the 0.95 correlation threshold, so nothing is flagged. A small difference is expected and healthy: stage 3 saw 2025 during training and stage 2 did not.

| model | task | horizon | correlation | mean abs. difference |
|---|---|---|---|---|
| xgboost | classification | 12 h | 0.9790 | 0.0202 |
| xgboost | classification | 6 h | 0.9800 | 0.0199 |
| xgboost | classification | 48 h | 0.9806 | 0.0241 |

*(the three lowest correlations, for reference)*


## Plots

- `results\test\plots\test_station_02198920_2025-05.png`
- `results\test\plots\test_station_02198955_2025-10.png`
- `results\test\plots\test_confusion_h24.png`
- `results\test\plots\test_onset_vs_false_alarm.png`
- `results\test\plots\test_rmse_vs_horizon.png`
- `results\test\plots\test_ablation.png`
- `results\test\plots\error_correlation_h24.png`

## Design choices and how they were checked

Six choices shaped this system. Each one is written down here with the check that was run on it. The checks in this section are **additional analyses**: they re-score predictions that already existed, and none of them was used to choose or change the deployed system, which was fixed in `FROZEN.md` before any 2025 target was read.

**1. Hyperparameters were chosen on the 2024 validation year.** This is the standard three-way split: 2023 trains, 2024 decides, 2025 judges. What 2025 tests is therefore the whole frozen pipeline - architecture, rounds, ensemble and warning thresholds together - rather than each model in isolation, which is the quantity an operator actually deploys. Every one of those settings, plus a SHA-256 for each model file, was committed before the test. The freeze is commit `e4c53b8209c7`, so the claim is checkable rather than asserted.

**2. The stacking meta-model was fitted on 2024 predictions of 2023-only members, then applied on top of members retrained on 2023+2024.** The headline chain therefore has a train/apply mismatch. A second chain has none: 2023-only members, the same meta-model fitted on their 2024 predictions, applied to 2025 (stage 1). If the mismatch were flattering the headline, the consistent chain would be the better of the two.
It is not. At 24 h the consistent chain gives RMSE 0.6871 against the headline's 0.6718 and onset recall 0.495 against 0.496 - a difference of -0.001 [-0.014, +0.013], which contains zero. The consistent chain is slightly *worse* on RMSE (+0.0153 [+0.0091, +0.0228]) and macro F1 (-0.0022 [-0.0033, -0.0011]), and the gap is the size of what one extra training year bought elsewhere (section 9). On onset recall - the metric the system is built for - the two chains are indistinguishable at 2 of the 4 horizons. The mismatch costs nothing the test can detect; the headline is better because its members saw a second year, not because the meta-model was handed inputs it had not learned to blend.

| horizon | RMSE consistent | RMSE headline | RMSE diff [95% CI] | R2 consistent | R2 headline | macroF1 consistent | macroF1 headline | macroF1 diff [95% CI] | onset consistent | onset headline | onset diff [95% CI] | FA consistent | FA headline | FA diff [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 h | 0.5459 | 0.5144 | +0.0314 [+0.0247, +0.0394] | 0.9674 | 0.9710 | 0.9001 | 0.9052 | -0.0051 [-0.0062, -0.0040] | 0.845 | 0.861 | -0.016 [-0.022, -0.010] | 0.0182 | 0.0175 | +0.0007 [+0.0001, +0.0013] |
| 12 h | 0.6089 | 0.5882 | +0.0206 [+0.0156, +0.0265] | 0.9594 | 0.9621 | 0.8936 | 0.8972 | -0.0036 [-0.0046, -0.0026] | 0.852 | 0.861 | -0.008 [-0.012, -0.005] | 0.0174 | 0.0173 | +0.0001 [-0.0004, +0.0006] |
| 24 h | 0.6871 | 0.6718 | +0.0153 [+0.0091, +0.0228] | 0.9483 | 0.9505 | 0.8838 | 0.8860 | -0.0022 [-0.0033, -0.0011] | 0.495 | 0.496 | -0.001 [-0.014, +0.013] | 0.0163 | 0.0150 | +0.0012 [+0.0003, +0.0022] |
| 48 h | 0.8739 | 0.8535 | +0.0207 [+0.0137, +0.0282] | 0.9163 | 0.9202 | 0.8517 | 0.8562 | -0.0045 [-0.0065, -0.0028] | 0.487 | 0.485 | +0.001 [-0.012, +0.014] | 0.0150 | 0.0133 | +0.0018 [+0.0006, +0.0030] |

*Consistent = stage 1 (2023-only members, no train/apply mismatch); headline = stage 2 (members retrained on 2023+2024). Both chains use the same meta-models and the same frozen thresholds, and predict the same 2025 rows. Differences are consistent minus headline, with 1,000 station-week bootstrap resamples; a CI containing zero means the two chains cannot be told apart. No model was retrained for this table. Source: `results/test/step10_pipeline.csv`.*

**3. One station stops reporting on 2025-11-19**, so the last six weeks of the test year have 29 stations rather than 30. The same 2025 predictions were re-scored two more ways - without that station for the whole year, and with all thirty stations but only up to 2025-11-18 - so the question is whether the uneven tail moved the headline.
It did not. At 24 h, onset recall is 0.496 in the headline and 0.496 to 0.504 across the three subsets; RMSE runs 0.6637 to 0.6718 and macro F1 0.8841 to 0.8860. The false-alarm rate stays inside the 2% promise at every horizon and in every subset (worst 1.96%). Dropping the station helps slightly - it is one of the harder ones - and cutting the year short costs a little on false alarms, because December is a quiet month whose safe hours were diluting the rate. Neither moves a conclusion.

| horizon | subset | rows | onsets | RMSE | R2 | macro F1 | onset recall | false alarm |
|---|---|---|---|---|---|---|---|---|
| 6 h | headline (30 stations, full year) | 253,954 | 13,602 | 0.5144 | 0.9710 | 0.9052 | 0.861 | 0.0175 |
| 6 h | without 05474500 (29 stations, full year) | 247,031 | 13,448 | 0.5052 | 0.9706 | 0.9049 | 0.864 | 0.0176 |
| 6 h | 30 stations, 2025-01-01..2025-11-18 | 226,165 | 12,924 | 0.5138 | 0.9708 | 0.9034 | 0.863 | 0.0196 |
| 12 h | headline (30 stations, full year) | 253,954 | 14,063 | 0.5882 | 0.9621 | 0.8972 | 0.861 | 0.0173 |
| 12 h | without 05474500 (29 stations, full year) | 247,031 | 13,864 | 0.5806 | 0.9611 | 0.8972 | 0.865 | 0.0172 |
| 12 h | 30 stations, 2025-01-01..2025-11-18 | 226,165 | 13,254 | 0.5846 | 0.9621 | 0.8954 | 0.860 | 0.0193 |
| 24 h | headline (30 stations, full year) | 253,954 | 5,668 | 0.6718 | 0.9505 | 0.8860 | 0.496 | 0.0150 |
| 24 h | without 05474500 (29 stations, full year) | 247,031 | 5,501 | 0.6637 | 0.9492 | 0.8860 | 0.504 | 0.0150 |
| 24 h | 30 stations, 2025-01-01..2025-11-18 | 226,165 | 5,477 | 0.6669 | 0.9507 | 0.8841 | 0.502 | 0.0167 |
| 48 h | headline (30 stations, full year) | 253,954 | 8,032 | 0.8535 | 0.9202 | 0.8562 | 0.485 | 0.0133 |
| 48 h | without 05474500 (29 stations, full year) | 247,031 | 7,806 | 0.8429 | 0.9182 | 0.8562 | 0.494 | 0.0132 |
| 48 h | 30 stations, 2025-01-01..2025-11-18 | 226,165 | 7,793 | 0.8502 | 0.9197 | 0.8529 | 0.491 | 0.0148 |

*Stacking with the frozen warning rule, re-scored only - no model was retrained and no threshold was re-chosen. Source: `results/test/step10_dropout.csv`.*

**4. The two deep models are the weakest members** and were kept anyway. The reason is diversity, not accuracy: an ensemble gains from members that are wrong about different hours, so what matters is how correlated their errors are with the trees'.
The gradient-boosted trees' 24 h errors correlate 0.963 to 0.976 with each other. A deep model against a boosted tree is lower: XGBoost and LightGBM sit at 0.906 and 0.918 against the BiLSTM, with the whole cross-family range running 0.905 to 0.955. The two deep models are close to interchangeable with each other (0.995), which is why there would be little to gain from a third. **Lower correlation means the members miss different hours, so averaging or stacking them cancels more error than combining models that fail together.** That is what the deep members are in the ensemble for, and it is why they stay in even though each one on its own is beaten by every tree.

|  | random_forest | xgboost | lightgbm | bilstm | cnn_lstm |
|---|---|---|---|---|---|
| random_forest | 1.000 | 0.965 | 0.963 | 0.955 | 0.954 |
| xgboost | 0.965 | 1.000 | 0.976 | 0.906 | 0.905 |
| lightgbm | 0.963 | 0.976 | 1.000 | 0.918 | 0.918 |
| bilstm | 0.955 | 0.906 | 0.918 | 1.000 | 0.995 |
| cnn_lstm | 0.954 | 0.905 | 0.918 | 0.995 | 1.000 |

*Pearson correlation of 24 h errors (`y_pred - y_true`) over the 2025 rows, from the predictions step 8 already wrote. Heatmap: `results/test/plots/error_correlation_h24.png`. **Analysis only - not used to choose or change the system.** Source: `results/test/step10_error_correlation_h24.csv`.*

**5. The warning rule is the main contribution of the decision layer.** Argmax answers *which class is most likely*; a warning system asks *is the chance of Low high enough to act on*, which is a lower bar and a different question. Separating the two is the decision layer's whole job, and on 2025 it is worth more than any choice of ensemble.
At 24 h the rule lifts onset recall from 0.285 to 0.496 - **1.7 times** as many onsets caught - for a false-alarm rate that rises from 0.46% to 1.50%, still inside the 2% the system promises. For comparison, choosing stacking over the best single model is worth 0.0104 mg/L of RMSE and nothing detectable on macro F1 or onset recall (section 3). The threshold was chosen on out-of-fold 2024 probabilities against a 1.8% internal budget - tighter than the promise on purpose - and frozen before the test.

| horizon | onset recall (argmax) | onset recall (rule) | false alarm (argmax) | false alarm (rule) |
|---|---|---|---|---|
| 6 h | 0.800 | 0.861 | 0.0103 | 0.0175 |
| 12 h | 0.795 | 0.861 | 0.0096 | 0.0173 |
| 24 h | 0.285 | 0.496 | 0.0046 | 0.0150 |
| 48 h | 0.380 | 0.485 | 0.0070 | 0.0133 |

**6. The deep models were trained on a GPU with non-deterministic kernels**, so a rerun does not reproduce a run bit for bit. The frozen system handles that by averaging three seeds, and the size of the remaining wobble was measured rather than assumed.
Across seeds 42-44 the single-model spread at 24 h is bilstm RMSE +-0.0023, macro F1 +-0.0014; cnn_lstm RMSE +-0.0017, macro F1 +-0.0041 (standard deviation over three seeds). The differences the 2025 tables are asked to resolve are of the same order: stacking beats the best single model by 0.0104 mg/L of RMSE and the simple average by 0.0019, against a worst-case seed spread of 0.0023. On macro F1 the gaps (0.0008 and 0.0005) are smaller than the spread (0.0041). So the ranking *among the ensembles* sits inside run-to-run noise. That is exactly why step 7 chose stacking by a rule written before the numbers were known rather than by whichever value came out highest, and why section 3 reports bootstrap intervals instead of point differences. The margin over persistence, and the margin the warning rule adds, are both an order of magnitude larger than the noise.

| model | seeds | RMSE mean | RMSE std | RMSE range | macro F1 mean | macro F1 std | macro F1 range |
|---|---|---|---|---|---|---|---|
| bilstm | 42;43;44 | 0.6827 | 0.0023 | 0.0042 | 0.8699 | 0.0014 | 0.0028 |
| cnn_lstm | 42;43;44 | 0.6804 | 0.0017 | 0.0033 | 0.8742 | 0.0041 | 0.0074 |

*Seed-to-seed spread of one deep model at 24 h, from `models/dl/*_seed_spread.csv` - measured on the 2024 validation year, the only year on which repeated runs were scored. The frozen system averages all three seeds, so the wobble inside the deployed ensemble is smaller than this. Sources: `results/test/step10_seed_spread.csv`, `results/test/step10_method_gaps.csv`.*


## Deployed models on 2026 (pre-registered in plan_2026.md)

`models/final` - the five members retrained on **2023+2024+2025** with the frozen settings, combined by the same stacking meta-model and thresholded by the same frozen warning rule - had never been scored: 2025 was inside its training data, so there was no year left to score it on. 2026 is that year. The test period, the stations, the metrics and the pass criteria were written into [`reports/plan_2026.md`](plan_2026.md) and committed **before any 2026 dissolved-oxygen value was read**; nothing was retuned, and the model version id `final-35ab6d1d27d1` is the SHA-256 over the per-file hashes of the folder that produced these numbers.

**Window** 2026-01-01 00:00 .. 2026-10-13 23:00 UTC (the last complete ERA5 day), **181,203 scored station-hours**, **28 of 30 stations**.

> **One correction, recorded rather than hidden.** The first run of this scoring was discarded. `clean.clean_station` had grown an `index` argument so the deployment could clean a 2026 window with the project's own code, but the weather and air tables were still being put on the hard-coded 2023-2025 spine; the join then left every weather and air feature NaN, and the models were being fed water only. The giveaway was the live-data penalty coming out as *exactly* zero at all four horizons - weather a model never reads cannot change its answer. The bug is fixed, `tests/test_deploy.py` now fails if it comes back (both as a unit test on `clean_station` and as a check on the real 2026 table), and the window, the stations, the models and the criteria below are the ones `plan_2026.md` fixed in advance - none of them was touched. The 2023-2025 cleaning is unaffected, because it takes the default index.

**Headline, 24 hours ahead.** R2 0.936, RMSE **0.738 mg/L** (+15.1% against assuming no change), macro F1 0.872, **47.9% of onsets caught** at a false-alarm rate of **1.65%** against the 2% the system promises.


### 2026, all horizons

| horizon | rows | R2 | RMSE | skill vs no-change | macro F1 | Low recall | onset recall | false alarms |
|---|---|---|---|---|---|---|---|---|
| 6 h | 180,545 | 0.9554 | 0.6172 | +0.650 | 0.8924 | 0.939 | 0.884 | 1.86% |
| 12 h | 180,250 | 0.9449 | 0.6863 | +0.656 | 0.8836 | 0.936 | 0.878 | 2.02% |
| 24 h | 179,811 | 0.9361 | 0.7383 | +0.151 | 0.8723 | 0.922 | 0.479 | 1.65% |
| 48 h | 179,039 | 0.9046 | 0.9010 | +0.199 | 0.8374 | 0.884 | 0.490 | 1.75% |

*stacking with the frozen warning rule, on `models/final`. Persistence and every member are in `results/live_2026/metrics_*.csv`.*


### 2026 against the 2025 95% intervals

The verdicts are the three words fixed in section 7.2 of the plan: **holds** means the 2026 value is inside the 2025 station-week interval or better than it, **degraded** means it is outside in the worse direction, **improved** outside in the better direction. The intervals are 1,000 station-week resamples of the 2025 predictions, computed before 2026 was read.

| horizon | metric | 2025 | 2025 95% CI | 2026 | change | verdict |
|---|---|---|---|---|---|---|
| 6 h | rmse | 0.5146 | [0.4908, 0.5378] | 0.6172 | +0.1026 | **degraded** |
| 6 h | f1_macro | 0.9050 | [0.8998, 0.9103] | 0.8924 | -0.0126 | **degraded** |
| 6 h | onset_recall | 0.8608 | [0.8424, 0.8782] | 0.8842 | +0.0233 | improved |
| 6 h | false_alarm_rate | 0.0175 | [0.0155, 0.0198] | 0.0186 | +0.0011 | holds |
| 12 h | rmse | 0.5885 | [0.5602, 0.6156] | 0.6863 | +0.0979 | **degraded** |
| 12 h | f1_macro | 0.8970 | [0.8911, 0.9034] | 0.8836 | -0.0134 | **degraded** |
| 12 h | onset_recall | 0.8603 | [0.8395, 0.8791] | 0.8783 | +0.0179 | holds |
| 12 h | false_alarm_rate | 0.0173 | [0.0153, 0.0194] | 0.0202 | +0.0029 | **degraded** |
| 24 h | rmse | 0.6719 | [0.6407, 0.7011] | 0.7383 | +0.0665 | **degraded** |
| 24 h | f1_macro | 0.8858 | [0.8794, 0.8922] | 0.8723 | -0.0134 | **degraded** |
| 24 h | onset_recall | 0.4960 | [0.4626, 0.5310] | 0.4794 | -0.0166 | holds |
| 24 h | false_alarm_rate | 0.0151 | [0.0132, 0.0171] | 0.0165 | +0.0015 | holds |
| 48 h | rmse | 0.8534 | [0.8188, 0.8848] | 0.9010 | +0.0475 | **degraded** |
| 48 h | f1_macro | 0.8561 | [0.8480, 0.8645] | 0.8374 | -0.0187 | **degraded** |
| 48 h | onset_recall | 0.4852 | [0.4473, 0.5235] | 0.4897 | +0.0045 | holds |
| 48 h | false_alarm_rate | 0.0133 | [0.0113, 0.0153] | 0.0175 | +0.0042 | **degraded** |

**Verdict against plan_2026.md: PARTIAL.** the 2% false-alarm promise holds (1.65% at 24 h), but rmse at 24 h falls outside the 2025 interval.


### Lead time on 2026

|  | value |
|---|---|
| low-DO events in 2026 | 204 |
| warned in advance | 106 (52.0%) |
| median lead time | 18 h |
| mean lead time | 14.3 h |
| stations with at least one event | 26 |

### The live-data penalty

A deployed system cannot wait five days for ERA5. On the last **30 days** of the window the same 18,673 station-hours were scored twice: once on ERA5 archive weather and CAMS, and once on the Open-Meteo forecast API's recent past (`ecmwf_ifs025`, the only forecast model that carries `soil_moisture_0_to_7cm`) with the CAMS forecast feed. The water data is identical in both, so the difference is the price of running live. A real live run has forecast weather only for roughly the last five hours-to-days of its window, so this is an **upper bound** on the penalty, not the everyday cost.

| horizon | RMSE ERA5 | RMSE forecast | RMSE diff | onset ERA5 | onset forecast | onset diff | false alarms ERA5 | false alarms forecast |
|---|---|---|---|---|---|---|---|---|
| 6 h | 0.5427 | 0.5408 | -0.0020 | 0.899 | 0.903 | +0.003 | 2.32% | 2.31% |
| 12 h | 0.5820 | 0.5809 | -0.0011 | 0.864 | 0.866 | +0.002 | 2.56% | 2.61% |
| 24 h | 0.6261 | 0.6248 | -0.0012 | 0.412 | 0.412 | +0.000 | 1.97% | 2.00% |
| 48 h | 0.7864 | 0.7853 | -0.0012 | 0.494 | 0.500 | +0.006 | 1.76% | 1.71% |

*Forecast minus ERA5; a positive RMSE difference and a negative onset difference both mean the forecast feed is worse. Source: `results/live_2026/penalty/penalty.csv`.*


### Walk-forward: three test years

The same frozen system, three expanding training windows. The 2024 row is **not a clean test**: 2024 is the year every hyperparameter, the ensemble and the warning thresholds were chosen on, so its numbers are an in-sample upper bound, and they are the blocked cross-fitted out-of-fold ones from step 7. No model was retrained for this table and nothing was retuned.

| train | test | horizon | rows | RMSE | macro F1 | onset recall | false alarms | note |
|---|---|---|---|---|---|---|---|---|
| 2023 | 2024 | 6 h | 252,598 | 0.5216 | 0.8931 | 0.904 | 2.48% | not clean: 2024 was the tuning year |
| 2023 | 2024 | 12 h | 252,259 | 0.5806 | 0.8850 | 0.891 | 2.49% | not clean: 2024 was the tuning year |
| 2023 | 2024 | 24 h | 251,715 | 0.6509 | 0.8761 | 0.594 | 2.40% | not clean: 2024 was the tuning year |
| 2023 | 2024 | 48 h | 250,736 | 0.8532 | 0.8407 | 0.555 | 2.46% | not clean: 2024 was the tuning year |
| 2023-2024 | 2025 | 6 h | 252,900 | 0.5144 | 0.9052 | 0.861 | 1.75% | the frozen result |
| 2023-2024 | 2025 | 12 h | 252,447 | 0.5882 | 0.8972 | 0.861 | 1.73% | the frozen result |
| 2023-2024 | 2025 | 24 h | 251,671 | 0.6718 | 0.8860 | 0.496 | 1.50% | the frozen result |
| 2023-2024 | 2025 | 48 h | 250,317 | 0.8535 | 0.8562 | 0.485 | 1.33% | the frozen result |
| 2023-2025 | 2026 | 6 h | 180,545 | 0.6172 | 0.8924 | 0.884 | 1.86% | deployed models, pre-registered |
| 2023-2025 | 2026 | 12 h | 180,250 | 0.6863 | 0.8836 | 0.878 | 2.02% | deployed models, pre-registered |
| 2023-2025 | 2026 | 24 h | 179,811 | 0.7383 | 0.8723 | 0.479 | 1.65% | deployed models, pre-registered |
| 2023-2025 | 2026 | 48 h | 179,039 | 0.9010 | 0.8374 | 0.490 | 1.75% | deployed models, pre-registered |

### Per station on 2026, 24 hours ahead

Sorted by how many onsets each station actually had.

| station | rows | RMSE | macro F1 | onset recall | false alarms | onsets |
|---|---|---|---|---|---|---|
| 08048000 | 6,267 | 1.718 | 0.646 | 0.202 | 0.0429 | 580 |
| 02110701 | 6,436 | 1.040 | 0.765 | 0.600 | 0.0947 | 380 |
| 02198955 | 6,855 | 0.441 | 0.889 | 0.797 | 0.0496 | 350 |
| 02198920 | 6,838 | 0.443 | 0.903 | 0.883 | 0.0328 | 333 |
| 02198950 | 6,864 | 0.439 | 0.898 | 0.809 | 0.0323 | 330 |
| 03430200 | 6,864 | 1.279 | 0.627 | 0.084 | 0.0101 | 262 |
| 02110550 | 6,707 | 0.324 | 0.882 | 0.645 | 0.0810 | 248 |
| 02264051 | 6,363 | 0.670 | 0.781 | 0.300 | 0.0130 | 213 |
| 02322700 | 6,864 | 0.289 | 0.863 | 0.686 | 0.0446 | 204 |
| 02110500 | 6,435 | 0.478 | 0.879 | 0.438 | 0.0373 | 201 |
| 02239501 | 6,571 | 0.535 | 0.827 | 0.423 | 0.0419 | 189 |
| 02110704 | 6,277 | 0.253 | 0.911 | 0.567 | 0.0507 | 178 |
| 01484080 | 6,800 | 0.556 | 0.886 | 0.797 | 0.0244 | 153 |
| 07239450 | 6,229 | 1.280 | 0.847 | 0.214 | 0.0042 | 145 |
| 01302020 | 6,779 | 0.664 | 0.835 | 0.275 | 0.0029 | 138 |
| 07049000 | 6,771 | 0.912 | 0.817 | 0.169 | 0.0030 | 118 |
| 03612600 | 6,851 | 0.270 | 0.749 | 0.055 | 0.0042 | 109 |
| 03353415 | 5,272 | 0.958 | 0.777 | 0.511 | 0.0150 | 94 |
| 05474500 | 4,574 | 0.459 | 0.843 | 0.224 | 0.0058 | 85 |
| 03290500 | 6,756 | 0.840 | 0.789 | 0.148 | 0.0024 | 81 |
| 11273400 | 6,475 | 0.442 | 0.910 | 0.364 | 0.0034 | 33 |
| 03431083 | 6,746 | 0.854 | 0.628 | 0.053 | 0.0026 | 19 |
| 07048600 | 4,986 | 0.528 | 0.750 | 0.158 | 0.0029 | 19 |
| 040851385 | 6,750 | 0.541 | 0.553 | 0.000 | 0.0034 | 18 |
| 03353200 | 6,292 | 0.827 | 0.692 | 0.000 | 0.0005 | 9 |
| 02135000 | 6,853 | 0.248 | 0.762 | 0.600 | 0.0004 | 5 |
| 02198840 | 6,864 | 0.207 | 0.677 | 0.333 | 0.0001 | 3 |
| 01484525 | 6,864 | 0.791 | 0.431 | nan | 0.0000 | 0 |

### Stations not scored on 2026

| station | stage | reason |
|---|---|---|
| 02292900 | fetch | the DO sensor reported nothing in the window |
| 02276998 | fetch | the DO sensor reported nothing in the window |

*The plan said a station whose DO sensor did not report in 2026 would be dropped and listed, and that the comparison would be marked unreliable below 20 stations. 28 reported.*


*Sources: `results/live_2026/` (metrics, bootstrap, per-station, lead times, skipped stations, `compare_2025_2026.csv`, `walk_forward.csv`) and `results/live_2026/penalty/`. Written by `src/deploy/score_2026.py` and `src/deploy/walk_forward.py`.*


## Scope and future work

- **Evaluated at monitored stations.** That is the intended use: a warning for a river where a sensor already sits, turning a reading into a forecast. Every station in the 2025 test also appears in training, so nothing here measures transfer to a river the system has never seen - testing on unseen stations is future work.
- **US rivers, USGS gauges.** Water data comes from USGS instantaneous values, with ERA5 weather and CAMS air quality. Applying the same pipeline to Indian rivers - the CPCB real-time stations, for example - is future work, and would need its own freeze and its own untouched test year.
- **Two test years now, not one.** 2025 was scored once against `FROZEN.md`, and 2026 was then scored once against `plan_2026.md` with the deployed models (see **Deployed models on 2026**). Two draws are better than one and still not many; the walk-forward table is the honest summary of what is known, and the 2024 row in it is in-sample and marked as such.
- **Adaptive weighting (ATAE) is worth revisiting.** It was rejected on validation by a rule written in advance, and it then came out level with or just ahead of stacking on 2025 (sections 1 and 2a). That is a promising hint, not evidence: the comparison happened after the test year was opened, so it cannot be used to change the deployed system. Only a new, unseen year can settle it.
