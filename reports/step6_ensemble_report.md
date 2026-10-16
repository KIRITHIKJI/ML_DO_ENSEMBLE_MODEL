# Step 6: ATAE, the adaptive threshold-aware ensemble

- Members: `random_forest`, `xgboost`, `lightgbm`, `bilstm`, `cnn_lstm`. `persistence` and `linear` stay as baselines, not members.
- Tuned on **val-A (2024-01-01 .. 2024-06-30)**, reported on **val-B (2024-07-01 .. 2024-12-31)**. No model was retrained in this step. The 2025 test split was not read.
- Before anything else, the step-5b probability calibration was **undone**: it raised macro F1 but pushed Low recall below persistence, which is the one thing the pass rule forbids. The members here are the uncalibrated step-5 models.

## The algorithm, step by step

For a forecast made at station *s* and hour *t*, for horizon *h*, with five member models *m*:

**1. Threshold-aware error.** A past prediction with observed value *y* and prediction *p* costs

```
regression:      e = w(y) * (y - p)^2
classification:  e = w(y) * Brier(probabilities, class(y))

                           (y - 5)^2
w(y) = 1 + alpha * exp( - ----------- )
                           2 * sigma^2
```

`w` peaks at the Low threshold of 5.0 mg/L and decays to 1 far away from it: an error at 4.9 mg/L turns a warning into a miss, an error at 12 mg/L costs nothing. `alpha = 0` switches the idea off.

**2. Only known errors.** A prediction made at *t'* targets *t' + h*, so its error cannot be known before then. At time *t* the ensemble uses model *m*'s errors that became known in the last *W* hours at station *s*:

```
err_m(s, t) = mean{ e_m(t') : t' + h <= t, t - W < t' + h <= t, station = s }
```

This is what keeps the ensemble honest - it is never weighted by an error it could not have seen. With fewer than 24 known errors the all-station error over the same window is used; with none at all, the weights are equal.

**3. Weights.** Normalise by the mean error across models, then softmax the negative:

```
normErr_m = err_m / mean_k(err_k)

              exp( -normErr_m / T )
weight_m = ---------------------------
            sum_k exp( -normErr_k / T )
```

A small *T* backs the current best model hard; a large *T* spreads the weight. Normalising first makes *T* mean the same thing at every horizon.

**4. The forecast.**

```
regression:      pred = sum_m weight_m * pred_m
classification:  probs = sum_m weight_m * probs_m,  class = argmax(probs)
```

## Grid search on val-A

Scored by the threshold-weighted RMSE (regression) and by the mean of macro F1 and onset recall (classification). The scoring weight is fixed at alpha=5.0, sigma=1.0 so that every candidate is marked by the same exam. One setting is chosen for all four horizons.


### regression: top 10 of 63

| alpha | sigma | W (h) | T | val-A score |
|---|---|---|---|---|
| 10.0 | 1.0 | 336.0 | 0.25 | -0.60777 |
| 5.0 | 1.0 | 336.0 | 0.25 | -0.60800 |
| 2.0 | 1.0 | 336.0 | 0.25 | -0.60837 |
| 10.0 | 0.5 | 336.0 | 0.25 | -0.60843 |
| 5.0 | 0.5 | 336.0 | 0.25 | -0.60849 |
| 2.0 | 0.5 | 336.0 | 0.25 | -0.60869 |
| 10.0 | 1.0 | 168.0 | 0.25 | -0.60876 |
| 0.0 | 0.5 | 336.0 | 0.25 | -0.60907 |
| 5.0 | 1.0 | 168.0 | 0.25 | -0.60909 |
| 10.0 | 1.0 | 72.0 | 0.25 | -0.60944 |

### classification: top 10 of 63

| alpha | sigma | W (h) | T | val-A score |
|---|---|---|---|---|
| 10.0 | 0.5 | 336.0 | 1.0 | 0.76067 |
| 2.0 | 0.5 | 336.0 | 1.0 | 0.76065 |
| 5.0 | 0.5 | 336.0 | 1.0 | 0.76064 |
| 2.0 | 1.0 | 336.0 | 1.0 | 0.76059 |
| 10.0 | 1.0 | 336.0 | 1.0 | 0.76055 |
| 5.0 | 1.0 | 336.0 | 1.0 | 0.76055 |
| 0.0 | 0.5 | 336.0 | 1.0 | 0.76050 |
| 2.0 | 0.5 | 168.0 | 1.0 | 0.75988 |
| 10.0 | 1.0 | 168.0 | 1.0 | 0.75987 |
| 5.0 | 0.5 | 168.0 | 1.0 | 0.75986 |

**Chosen** - regression: alpha=10, sigma=1, W=336 h, T=0.25; classification: alpha=10, sigma=0.5, W=336 h, T=1. Both are saved in `results/val/atae_chosen.json`.

The best single model on val-A, used as the comparison throughout, is `lightgbm` for regression and `cnn_lstm` for classification.


## Regression at 24 hours, val-B

| method | R2 | RMSE | MAE | NSE | skill vs persistence | danger-zone RMSE |
|---|---|---|---|---|---|---|
| persistence | 0.9317 | 0.7483 | 0.4675 | 0.9317 | - | 0.7243 |
| linear | 0.9422 | 0.6884 | 0.4407 | 0.9422 | +0.080 | 0.6415 |
| random_forest | 0.9445 | 0.6742 | 0.4282 | 0.9445 | +0.099 | 0.6138 |
| xgboost | 0.9443 | 0.6755 | 0.4309 | 0.9443 | +0.097 | 0.6040 |
| lightgbm | 0.9457 | 0.6668 | 0.4257 | 0.9457 | +0.109 | 0.5993 |
| bilstm | 0.9421 | 0.6885 | 0.4331 | 0.9421 | +0.080 | 0.6452 |
| cnn_lstm | 0.9425 | 0.6866 | 0.4328 | 0.9425 | +0.082 | 0.6415 |
| best_single | 0.9457 | 0.6668 | 0.4257 | 0.9457 | +0.109 | 0.5993 |
| simple_average | 0.9464 | 0.6627 | 0.4175 | 0.9464 | +0.114 | 0.6072 |
| inverse_error | 0.9464 | 0.6625 | 0.4174 | 0.9464 | +0.115 | 0.6067 |
| stacking | 0.9471 | 0.6584 | 0.4158 | 0.9471 | +0.120 | 0.6020 |
| atae_adaptive_only | 0.9469 | 0.6593 | 0.4140 | 0.9469 | +0.119 | 0.6047 |
| atae_threshold_only | 0.9466 | 0.6612 | 0.4173 | 0.9466 | +0.116 | 0.6033 |
| atae | 0.9469 | 0.6595 | 0.4145 | 0.9469 | +0.119 | 0.6027 |

Danger-zone RMSE is the error over rows whose true DO is between 3.0 and 7.0 mg/L - the band where a river actually crosses into trouble.


## Classification at 24 hours, val-B

| method | accuracy | balanced acc | macro F1 | ROC-AUC | Low recall | onset recall | false alarm rate | transition accuracy |
|---|---|---|---|---|---|---|---|---|
| persistence | 0.883 | 0.871 | 0.8704 | - | 0.898 | 0.000 | 0.0000 | 0.000 |
| linear | 0.873 | 0.868 | 0.8627 | 0.967 | 0.918 | 0.323 | 0.0224 | 0.288 |
| random_forest | 0.892 | 0.886 | 0.8826 | 0.977 | 0.918 | 0.278 | 0.0078 | 0.225 |
| xgboost | 0.893 | 0.887 | 0.8839 | 0.978 | 0.915 | 0.326 | 0.0095 | 0.294 |
| lightgbm | 0.892 | 0.887 | 0.8829 | 0.977 | 0.914 | 0.352 | 0.0121 | 0.325 |
| bilstm | 0.877 | 0.874 | 0.8688 | 0.971 | 0.905 | 0.380 | 0.0169 | 0.362 |
| cnn_lstm | 0.883 | 0.880 | 0.8744 | 0.974 | 0.910 | 0.364 | 0.0154 | 0.352 |
| best_single | 0.883 | 0.880 | 0.8744 | 0.974 | 0.910 | 0.364 | 0.0154 | 0.352 |
| simple_average | 0.892 | 0.888 | 0.8839 | 0.977 | 0.917 | 0.338 | 0.0101 | 0.310 |
| inverse_error | 0.893 | 0.888 | 0.8842 | 0.978 | 0.917 | 0.336 | 0.0099 | 0.308 |
| stacking | 0.890 | 0.886 | 0.8814 | 0.975 | 0.926 | 0.363 | 0.0115 | 0.286 |
| atae_adaptive_only | 0.892 | 0.888 | 0.8840 | 0.978 | 0.917 | 0.333 | 0.0099 | 0.306 |
| atae_threshold_only | 0.893 | 0.888 | 0.8842 | 0.978 | 0.917 | 0.337 | 0.0099 | 0.309 |
| atae | 0.892 | 0.888 | 0.8841 | 0.978 | 0.917 | 0.334 | 0.0099 | 0.308 |

At 24 h on val-B there are **3,717 onset rows** (not Low now, Low at t+h) and **14,659 transition rows** out of 124,812 scored rows. **Persistence scores 0 on onset recall and 0 on transition accuracy by definition**: it predicts no change, so it is wrong on every row where the class changes.


## All horizons, val-B

| method | RMSE 6h | RMSE 12h | RMSE 24h | RMSE 48h | macroF1 6h | macroF1 12h | macroF1 24h | macroF1 48h | onset recall 6h | onset recall 12h | onset recall 24h | onset recall 48h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| persistence | 1.531 | 1.696 | 0.748 | 1.037 | 0.730 | 0.720 | 0.870 | 0.816 | 0.000 | 0.000 | 0.000 | 0.000 |
| linear | 0.656 | 0.658 | 0.688 | 0.914 | 0.876 | 0.871 | 0.863 | 0.815 | 0.795 | 0.747 | 0.323 | 0.368 |
| random_forest | 0.643 | 0.635 | 0.674 | 0.886 | 0.886 | 0.889 | 0.883 | 0.841 | 0.759 | 0.711 | 0.278 | 0.372 |
| xgboost | 0.574 | 0.615 | 0.675 | 0.883 | 0.900 | 0.896 | 0.884 | 0.844 | 0.774 | 0.725 | 0.326 | 0.408 |
| lightgbm | 0.564 | 0.611 | 0.667 | 0.874 | 0.900 | 0.895 | 0.883 | 0.841 | 0.768 | 0.720 | 0.352 | 0.453 |
| bilstm | 0.688 | 0.683 | 0.689 | 0.917 | 0.876 | 0.872 | 0.869 | 0.834 | 0.781 | 0.710 | 0.380 | 0.382 |
| cnn_lstm | 0.688 | 0.686 | 0.687 | 0.909 | 0.881 | 0.876 | 0.874 | 0.838 | 0.781 | 0.686 | 0.364 | 0.397 |
| best_single | 0.564 | 0.611 | 0.667 | 0.874 | 0.881 | 0.876 | 0.874 | 0.838 | 0.781 | 0.686 | 0.364 | 0.397 |
| simple_average | 0.591 | 0.611 | 0.663 | 0.868 | 0.898 | 0.894 | 0.884 | 0.845 | 0.774 | 0.715 | 0.338 | 0.409 |
| inverse_error | 0.583 | 0.609 | 0.663 | 0.867 | 0.899 | 0.895 | 0.884 | 0.845 | 0.774 | 0.717 | 0.336 | 0.409 |
| stacking | 0.558 | 0.599 | 0.658 | 0.861 | 0.896 | 0.891 | 0.881 | 0.840 | 0.805 | 0.751 | 0.363 | 0.423 |
| atae_adaptive_only | 0.559 | 0.599 | 0.659 | 0.865 | 0.899 | 0.895 | 0.884 | 0.845 | 0.775 | 0.718 | 0.333 | 0.406 |
| atae_threshold_only | 0.567 | 0.604 | 0.661 | 0.865 | 0.898 | 0.895 | 0.884 | 0.845 | 0.775 | 0.717 | 0.337 | 0.409 |
| atae | 0.561 | 0.599 | 0.659 | 0.865 | 0.898 | 0.894 | 0.884 | 0.846 | 0.775 | 0.717 | 0.334 | 0.406 |

## Ablation at 24 hours, val-B

What each half of ATAE is worth, against the simpler things it could have been.

| method | what it is | RMSE | danger-zone RMSE | macro F1 | onset recall | false alarm rate |
|---|---|---|---|---|---|---|
| atae | adaptive + threshold-aware (full) | 0.6595 | 0.6027 | 0.8841 | 0.334 | 0.0099 |
| atae_adaptive_only | adaptive only (alpha = 0) | 0.6593 | 0.6047 | 0.8840 | 0.333 | 0.0099 |
| atae_threshold_only | threshold-aware only (static weights) | 0.6612 | 0.6033 | 0.8842 | 0.337 | 0.0099 |
| simple_average | equal weights | 0.6627 | 0.6072 | 0.8839 | 0.338 | 0.0101 |
| stacking | Ridge / logistic fitted on val-A | 0.6584 | 0.6020 | 0.8814 | 0.363 | 0.0115 |
| inverse_error | 1/error weights from val-A | 0.6625 | 0.6067 | 0.8842 | 0.336 | 0.0099 |
| best_single | lightgbm / cnn_lstm | 0.6668 | 0.5993 | 0.8744 | 0.364 | 0.0154 |

## Is the difference real? Block bootstrap

ATAE minus the best single model, 1,000 resamples of **station-weeks** (so a whole week at a station moves together, which respects the fact that neighbouring hours are not independent). 95% percentile intervals. For RMSE a NEGATIVE difference favours ATAE; for macro F1 and onset recall a POSITIVE one does.

| horizon | metric | ATAE - best single | 95% CI | verdict |
|---|---|---|---|---|
| 6 h | RMSE | -0.0032 | [-0.0067, +0.0004] | no significant difference |
| 6 h | macro F1 | +0.0172 | [+0.0146, +0.0198] | **ATAE better** |
| 6 h | onset recall | -0.0064 | [-0.0200, +0.0085] | no significant difference |
| 12 h | RMSE | -0.0118 | [-0.0162, -0.0075] | **ATAE better** |
| 12 h | macro F1 | +0.0185 | [+0.0160, +0.0210] | **ATAE better** |
| 12 h | onset recall | +0.0316 | [+0.0159, +0.0491] | **ATAE better** |
| 24 h | RMSE | -0.0073 | [-0.0139, -0.0014] | **ATAE better** |
| 24 h | macro F1 | +0.0097 | [+0.0074, +0.0120] | **ATAE better** |
| 24 h | onset recall | -0.0300 | [-0.0523, -0.0065] | best single better |
| 48 h | RMSE | -0.0093 | [-0.0180, -0.0010] | **ATAE better** |
| 48 h | macro F1 | +0.0081 | [+0.0051, +0.0112] | **ATAE better** |
| 48 h | onset recall | +0.0096 | [-0.0196, +0.0400] | no significant difference |

## Which model the ensemble trusts

Average ATAE weight on val-B:

| task | horizon | random_forest | xgboost | lightgbm | bilstm | cnn_lstm |
|---|---|---|---|---|---|---|
| regression | 6 h | 0.183 | 0.297 | 0.310 | 0.100 | 0.110 |
| regression | 12 h | 0.230 | 0.262 | 0.277 | 0.112 | 0.119 |
| regression | 24 h | 0.204 | 0.213 | 0.228 | 0.173 | 0.182 |
| regression | 48 h | 0.201 | 0.214 | 0.227 | 0.173 | 0.185 |
| classification | 6 h | 0.172 | 0.239 | 0.238 | 0.170 | 0.180 |
| classification | 12 h | 0.169 | 0.236 | 0.233 | 0.179 | 0.183 |
| classification | 24 h | 0.173 | 0.225 | 0.220 | 0.189 | 0.193 |
| classification | 48 h | 0.175 | 0.222 | 0.226 | 0.189 | 0.188 |

## Plots

- `results\val\plots\atae_station_month.png`
- `results\val\plots\atae_average_weights.png`
- `results\val\plots\atae_onset_vs_false_alarm.png`

## What ATAE is actually worth

Against the best single model chosen on val-A, and judged by a station-week bootstrap on val-B, ATAE is significantly **better** on 8 of 12 model-metric comparisons (macro F1 at 6 h, RMSE at 12 h, macro F1 at 12 h, onset recall at 12 h, RMSE at 24 h, macro F1 at 24 h, RMSE at 48 h, macro F1 at 48 h), significantly **worse** on 1 (onset recall at 24 h), and indistinguishable on 3 (RMSE at 6 h, onset recall at 6 h, onset recall at 48 h).

In plain numbers at 24 hours: ATAE's RMSE is 0.6595 against 0.6668 for the best single model and 0.6627 for a simple average of the same five; its macro F1 is 0.8841 against 0.8744 and 0.8839; its onset recall is 0.334 against 0.364 and 0.338. The gaps are small. The ablation table is the more informative read: most of what the ensemble gains over a single model comes from averaging five models at all, and the adaptive and threshold-aware parts move the result by much less than that first step does.

Two things should be said against the method rather than for it. First, it has four tuned numbers (alpha, sigma, W, T) chosen on val-A, while a simple average has none - so ATAE has to beat the average by a clear margin to have earned its complexity, not just edge past it. Second, the whole grid spanned a narrow range of scores, which says the result is not sensitive to these settings: comforting for robustness, but also a sign there was not much to find. What the ensemble does buy reliably is insurance - it is never much worse than the best member, and it does not require knowing in advance which member that will be, which on the 2025 test set is a real advantage because that choice has to be made blind.

## Honest notes

- The step-5b calibration was reverted (see the top of this report). The members are the uncalibrated step-5 models.
- ATAE's settings were chosen on val-A and every number here is val-B, so this comparison is clean. The MEMBERS, however, were tuned on the whole of 2024 in steps 3-5, so all methods here inherit that optimism equally.
- The ensemble can only be as good as its members allow. It has no mechanism for being right when all five are wrong.
- 2025 remains untouched.
