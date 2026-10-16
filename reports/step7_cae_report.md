# Step 7: the Context-Aware Ensemble and a warning rule

- Everything here is **out-of-fold over 2024**: six two-month folds, each predicted by models fitted on the other five, with **14 days purged** on both sides. Every row of 2024 is predicted exactly once, by a model that never saw it.
- That replaces the val-A / val-B split for this step, and **every** method - including persistence, the members, the simple average, stacking and ATAE - is re-scored on the same rows, so the comparison is like for like. These numbers are not comparable with step 6's.
- No member was retrained. The 2025 test split was not read.

## The protocol

| fold | period | test rows | train rows | purged rows |
|---|---|---|---|---|
| 0 | Jan-Feb | 41,983 | 202,038 | 9,582 |
| 1 | Mar-Apr | 42,944 | 190,869 | 19,790 |
| 2 | May-Jun | 42,879 | 191,058 | 19,666 |
| 3 | Jul-Aug | 41,963 | 192,480 | 19,160 |
| 4 | Sep-Oct | 42,477 | 192,022 | 19,104 |
| 5 | Nov-Dec | 41,357 | 202,367 | 9,879 |

The purge has to cover two leaks at once: a target 48 h ahead, and the 336 h window the ensembles look back over. Fourteen days covers both.


## CAE in one paragraph

ATAE asked *which member has been right lately*. Step 6 showed that is worth almost nothing over a plain average, because the members do not take turns being good - they are good at different **situations**. The BiLSTM catches the most onsets at 24 h and has the worst macro F1; LightGBM is the reverse. So CAE asks a different question: *given what the river is doing right now, whom should I believe?* It is a small LightGBM (15 leaves, 200 samples per leaf) per task and horizon, taking the five members' outputs, how much they disagree, how each has been doing over the last 336 h (known errors only), and the river's state at t: DO, its distance from 5 mg/L, its 6 h and 24 h change, its daily swing, saturation, water temperature, hours since it was last below 5, and the time of day and year. There is no `station_id` input, on purpose - it should learn situations, not stations.


## The warning rule

Argmax answers *which class is most likely*. A warning system asks *is the chance of Low high enough to act on*, which is a lower bar - and at 24 h the most likely class is almost never Low even when the risk is real. So each method issues a warning when P(Low) >= tau, with tau chosen on the training folds to catch as many onsets as possible while keeping the false-alarm rate at or below **2%**. Both rules are reported below.

Median tau chosen across folds at 24 h: `stacking` 0.21, `xgboost` 0.28, `lightgbm` 0.29, `cae` 0.30, `atae` 0.32, `simple_average` 0.33, `random_forest` 0.33, `cnn_lstm` 0.40, `bilstm` 0.42, `linear` 0.42.


## Regression at 24 hours, out-of-fold 2024

| method | R2 | RMSE | MAE | skill vs persistence | danger-zone RMSE |
|---|---|---|---|---|---|
| persistence | 0.9370 | 0.7362 | 0.4528 | - | 0.7072 |
| linear | 0.9467 | 0.6774 | 0.4247 | +0.080 | 0.6283 |
| random_forest | 0.9486 | 0.6653 | 0.4121 | +0.096 | 0.6020 |
| xgboost | 0.9486 | 0.6654 | 0.4126 | +0.096 | 0.5882 |
| lightgbm | 0.9497 | 0.6581 | 0.4089 | +0.106 | 0.5870 |
| bilstm | 0.9463 | 0.6800 | 0.4205 | +0.076 | 0.6326 |
| cnn_lstm | 0.9466 | 0.6777 | 0.4193 | +0.079 | 0.6291 |
| simple_average | 0.9503 | 0.6540 | 0.4027 | +0.112 | 0.5955 |
| stacking | 0.9508 | 0.6509 | 0.4008 | +0.116 | 0.5914 |
| atae | 0.9506 | 0.6517 | 0.3999 | +0.115 | 0.5912 |
| cae | 0.9482 | 0.6675 | 0.4182 | +0.093 | 0.6015 |

## Classification at 24 hours, with the warning rule

| method | accuracy | macro F1 | ROC-AUC | Low recall | onset recall | false alarm rate | transition acc |
|---|---|---|---|---|---|---|---|
| persistence | 0.900 | 0.8793 | - | 0.893 | 0.000 | 0.0000 | 0.000 |
| linear | 0.884 | 0.8648 | 0.973 | 0.937 | 0.473 | 0.0235 | 0.313 |
| random_forest | 0.898 | 0.8759 | 0.981 | 0.953 | 0.567 | 0.0240 | 0.254 |
| xgboost | 0.899 | 0.8773 | 0.982 | 0.955 | 0.608 | 0.0246 | 0.308 |
| lightgbm | 0.898 | 0.8772 | 0.981 | 0.951 | 0.579 | 0.0240 | 0.322 |
| bilstm | 0.884 | 0.8646 | 0.976 | 0.933 | 0.528 | 0.0243 | 0.360 |
| cnn_lstm | 0.889 | 0.8688 | 0.978 | 0.940 | 0.545 | 0.0243 | 0.356 |
| simple_average | 0.897 | 0.8766 | 0.981 | 0.953 | 0.588 | 0.0235 | 0.315 |
| stacking | 0.897 | 0.8761 | 0.980 | 0.955 | 0.594 | 0.0240 | 0.301 |
| atae | 0.897 | 0.8768 | 0.982 | 0.954 | 0.597 | 0.0238 | 0.313 |
| cae | 0.897 | 0.8765 | 0.981 | 0.951 | 0.571 | 0.0231 | 0.317 |

## Classification at 24 hours, with argmax

| method | accuracy | macro F1 | ROC-AUC | Low recall | onset recall | false alarm rate | transition acc |
|---|---|---|---|---|---|---|---|
| persistence | 0.900 | 0.8793 | - | 0.893 | 0.000 | 0.0000 | 0.000 |
| linear | 0.889 | 0.8720 | 0.973 | 0.915 | 0.324 | 0.0135 | 0.295 |
| random_forest | 0.907 | 0.8905 | 0.981 | 0.916 | 0.278 | 0.0051 | 0.220 |
| xgboost | 0.908 | 0.8919 | 0.982 | 0.914 | 0.328 | 0.0062 | 0.287 |
| lightgbm | 0.906 | 0.8907 | 0.981 | 0.913 | 0.357 | 0.0078 | 0.317 |
| bilstm | 0.889 | 0.8730 | 0.976 | 0.905 | 0.398 | 0.0132 | 0.359 |
| cnn_lstm | 0.895 | 0.8788 | 0.978 | 0.910 | 0.400 | 0.0116 | 0.353 |
| simple_average | 0.906 | 0.8912 | 0.981 | 0.917 | 0.352 | 0.0069 | 0.305 |
| stacking | 0.906 | 0.8914 | 0.980 | 0.917 | 0.323 | 0.0059 | 0.282 |
| atae | 0.906 | 0.8914 | 0.982 | 0.917 | 0.346 | 0.0067 | 0.301 |
| cae | 0.906 | 0.8903 | 0.981 | 0.915 | 0.349 | 0.0073 | 0.311 |

## All horizons, out-of-fold (warning rule)

| method | RMSE 6h | RMSE 12h | RMSE 24h | RMSE 48h | macroF1 6h | macroF1 12h | macroF1 24h | macroF1 48h | onset recall 6h | onset recall 12h | onset recall 24h | onset recall 48h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| persistence | 1.462 | 1.671 | 0.736 | 1.022 | 0.749 | 0.739 | 0.879 | 0.826 | 0.000 | 0.000 | 0.000 | 0.000 |
| linear | 0.615 | 0.639 | 0.677 | 0.901 | 0.878 | 0.870 | 0.865 | 0.824 | 0.869 | 0.853 | 0.473 | 0.441 |
| random_forest | 0.613 | 0.619 | 0.665 | 0.879 | 0.885 | 0.882 | 0.876 | 0.839 | 0.870 | 0.874 | 0.567 | 0.507 |
| xgboost | 0.540 | 0.595 | 0.665 | 0.873 | 0.896 | 0.887 | 0.877 | 0.843 | 0.902 | 0.891 | 0.608 | 0.568 |
| lightgbm | 0.531 | 0.593 | 0.658 | 0.863 | 0.897 | 0.888 | 0.877 | 0.843 | 0.903 | 0.888 | 0.579 | 0.544 |
| bilstm | 0.645 | 0.657 | 0.680 | 0.907 | 0.875 | 0.868 | 0.865 | 0.832 | 0.862 | 0.842 | 0.528 | 0.515 |
| cnn_lstm | 0.641 | 0.660 | 0.678 | 0.901 | 0.882 | 0.870 | 0.869 | 0.836 | 0.868 | 0.849 | 0.545 | 0.530 |
| simple_average | 0.554 | 0.591 | 0.654 | 0.859 | 0.894 | 0.885 | 0.877 | 0.842 | 0.903 | 0.892 | 0.588 | 0.557 |
| stacking | 0.522 | 0.581 | 0.651 | 0.853 | 0.893 | 0.885 | 0.876 | 0.841 | 0.904 | 0.891 | 0.594 | 0.555 |
| atae | 0.525 | 0.581 | 0.652 | 0.857 | 0.896 | 0.887 | 0.877 | 0.843 | 0.905 | 0.894 | 0.597 | 0.559 |
| cae | 0.526 | 0.591 | 0.667 | 0.890 | 0.897 | 0.888 | 0.877 | 0.840 | 0.902 | 0.885 | 0.571 | 0.553 |

## Is CAE really different? Block bootstrap

1,000 resamples of station-weeks, out-of-fold rows. Negative RMSE favours CAE; positive macro F1 and onset recall favour CAE.


### CAE minus `stacking`

| horizon | metric | CAE - stacking | 95% CI | verdict |
|---|---|---|---|---|
| 6 h | RMSE | +0.0038 | [+0.0002, +0.0077] | stacking better |
| 6 h | macro F1 | +0.0041 | [+0.0027, +0.0055] | **CAE better** |
| 6 h | onset recall | -0.0020 | [-0.0077, +0.0042] | no significant difference |
| 12 h | RMSE | +0.0107 | [+0.0062, +0.0154] | stacking better |
| 12 h | macro F1 | +0.0029 | [+0.0012, +0.0045] | **CAE better** |
| 12 h | onset recall | -0.0066 | [-0.0131, -0.0010] | stacking better |
| 24 h | RMSE | +0.0165 | [+0.0124, +0.0209] | stacking better |
| 24 h | macro F1 | +0.0004 | [-0.0015, +0.0024] | no significant difference |
| 24 h | onset recall | -0.0225 | [-0.0423, -0.0025] | stacking better |
| 48 h | RMSE | +0.0369 | [+0.0261, +0.0480] | stacking better |
| 48 h | macro F1 | -0.0011 | [-0.0043, +0.0017] | no significant difference |
| 48 h | onset recall | -0.0015 | [-0.0190, +0.0177] | no significant difference |

### CAE minus `simple_average`

| horizon | metric | CAE - simple_average | 95% CI | verdict |
|---|---|---|---|---|
| 6 h | RMSE | -0.0285 | [-0.0345, -0.0223] | **CAE better** |
| 6 h | macro F1 | +0.0031 | [+0.0016, +0.0045] | **CAE better** |
| 6 h | onset recall | -0.0011 | [-0.0079, +0.0058] | no significant difference |
| 12 h | RMSE | +0.0004 | [-0.0049, +0.0057] | no significant difference |
| 12 h | macro F1 | +0.0027 | [+0.0006, +0.0047] | **CAE better** |
| 12 h | onset recall | -0.0067 | [-0.0145, +0.0013] | no significant difference |
| 24 h | RMSE | +0.0135 | [+0.0092, +0.0181] | simple_average better |
| 24 h | macro F1 | -0.0001 | [-0.0023, +0.0022] | no significant difference |
| 24 h | onset recall | -0.0169 | [-0.0366, +0.0042] | no significant difference |
| 48 h | RMSE | +0.0310 | [+0.0205, +0.0421] | simple_average better |
| 48 h | macro F1 | -0.0028 | [-0.0059, +0.0005] | no significant difference |
| 48 h | onset recall | -0.0043 | [-0.0251, +0.0157] | no significant difference |

## What the meta-model learned


### Top inputs, classification, 24 h

| feature | importance |
|---|---|
| doy_cos | 998 |
| doy_sin | 849 |
| water_temp_c | 788 |
| do_swing24 | 668 |
| proba_lightgbm_Good | 587 |
| proba_xgboost_Low | 540 |
| recent_err_xgboost | 494 |
| proba_random_forest_Medium | 475 |
| recent_err_random_forest | 463 |
| do_mgl_chg24 | 420 |
| recent_err_lightgbm | 418 |
| recent_err_cnn_lstm | 415 |

### Top inputs, regression, 24 h

| feature | importance |
|---|---|
| pred_lightgbm | 557 |
| pred_xgboost | 269 |
| pred_random_forest | 235 |
| pred_cnn_lstm | 142 |
| doy_sin | 77 |
| doy_cos | 67 |
| pred_bilstm | 62 |
| recent_err_bilstm | 61 |
| do_mgl_chg24 | 60 |
| water_temp_c | 39 |
| do_saturation | 34 |
| do_swing24 | 31 |

### Which member CAE leans on, by situation

Share of the meta-model's attention (mean absolute SHAP over each member's own inputs), 24 h classification:

| member | DO falling, within 1.5 mg/L of 5 | DO stable and above 7 |
|---|---|---|
| random_forest | 0.104 | 0.085 |
| xgboost | 0.369 | 0.343 |
| lightgbm | 0.379 | 0.421 |
| bilstm | 0.054 | 0.046 |
| cnn_lstm | 0.094 | 0.106 |

## Plots

- `results\val\plots\step7_onset_vs_false_alarm.png`
- `results\val\plots\step7_cae_shap_by_situation.png`

## DECISION

The rule, written down before any of this was run:

> CAE becomes the proposed method if, at 24 h with the warning rule, its onset recall is higher than stacking's with a confidence interval excluding 0, AND its RMSE is not significantly worse than stacking's. Otherwise the final system is stacking.

At 24 h, CAE minus stacking: onset recall -0.0225 [-0.0423, -0.0025] - not significantly higher; RMSE +0.0165 [+0.0124, +0.0209] - significantly worse.

**final_system = `stacking`** (written to config.yaml).

CAE does not earn it. At 24 h it catches 57.1% of onsets against stacking's 59.4%, with an RMSE of 0.6675 against 0.6509. The pre-set rule required a clear win on onset recall, and that did not happen, so the final system is the simpler method. This is the intended behaviour of writing the rule down first: a method does not get adopted because it is the newest thing tried.


## What actually moved the needle

The honest headline of step 7 is not CAE. It is the **warning rule**. At 24 h, switching from argmax to a tuned P(Low) threshold takes CAE's onset recall from 0.349 to 0.571 - the same model, the same probabilities, a different question asked of them - while keeping false alarms inside the 2% budget. Every method gains from it. Changing how the decision is made was worth far more than changing how the members are combined, which is a useful thing to know and slightly deflating for the ensemble work.


## The false-alarm budget slipped out-of-fold

tau is chosen as the lowest threshold whose false-alarm rate still fits the 2% budget, which means it lands *exactly* on the boundary: across methods at 24 h the in-sample rate averages 1.94%. Out-of-fold the same thresholds deliver 2.39% - over budget. Nothing is broken; this is what happens when a constraint is tuned to the limit, because the threshold that just fits on five-sixths of the year will not quite fit on the sixth. If the 2% were a real operational contract rather than a design target, tau should be chosen against a tighter internal budget (say 1.5%) to leave headroom. That is a one-line change in config.yaml and it is left as a finding rather than quietly applied, because changing the budget after seeing the result is how a constraint stops meaning anything.

## Honest notes

- The members were tuned on all of 2024 in steps 3-5, so every method here inherits that optimism equally. The cross-fitting makes the COMBINERS honest, not the members.
- The warning threshold is one number per method and horizon, chosen on five-sixths of the year. It is the least overfittable thing in this project, but it is still fitted.
- CAE has no station_id, but its context features come from stations it did see. It is honest across time, not across rivers; a genuinely new river is a different question this project does not answer.
- 2025 remains untouched.
