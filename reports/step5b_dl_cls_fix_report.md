# Step 5b: one attempt at the deep models' classification

- Trained on **2023**. The validation year is split: **val-A 2024-01-01 .. 2024-06-30** chose the loss weight and the calibration factors, **val-B 2024-07-01 .. 2024-12-31** is what every number below is scored on. Nothing in 2025 was read.
- Early stopping still watches the loss on the whole of 2024, as in step 5, so the split changes what is tuned, not how training stops.
- Every model, including the tree models and persistence, is re-scored on the same val-B rows, so the comparison is like for like. These numbers are therefore NOT comparable with the full-2024 numbers in the step-5 report.
- Pass rule, unchanged: macro F1 above persistence AND Low recall at least persistence's.

## What was tried, and what was chosen (on val-A)

| model | classification loss weight | seeds | val-A mean macro F1 | val-A RMSE 24 h | regression ok? | chosen | note |
|---|---|---|---|---|---|---|---|
| bilstm | 1.0 | 42,43,44 | 0.8706 | 0.6714 | yes | **chosen** | step 5 model |
| bilstm | 2.0 | 42 | 0.8676 | 0.6913 | **no** |  | worst RMSE change +11.1% |
| bilstm | 3.0 | 42 | 0.8655 | 0.7017 | **no** |  | worst RMSE change +19.0% |
| cnn_lstm | 1.0 | 42,43,44 | 0.8772 | 0.6689 | yes | **chosen** | step 5 model |
| cnn_lstm | 2.0 | 42 | 0.8764 | 0.6803 | **no** |  | worst RMSE change +11.3% |
| cnn_lstm | 3.0 | 42 | 0.8768 | 0.6913 | **no** |  | worst RMSE change +16.9% |

### Chosen settings

| model | loss weight | seeds | calibration (Low, Medium, Good) |
|---|---|---|---|
| bilstm | 1.0 | 42,43,44 (from step 5) | 6 h: (0.7, 0.7, 1.3); 12 h: (0.7, 0.7, 1.3); 24 h: (0.7, 0.85, 1.3); 48 h: (0.7, 0.85, 1.3) |
| cnn_lstm | 1.0 | 42,43,44 (from step 5) | 6 h: (0.7, 0.85, 1.3); 12 h: (0.7, 0.85, 1.3); 24 h: (0.7, 0.85, 1.3); 48 h: (0.7, 1, 1.3) |

- `bilstm`: calibration moved val-A mean macro F1 from 0.8706 to 0.8764.
- `cnn_lstm`: calibration moved val-A mean macro F1 from 0.8772 to 0.8825.


## Before and after, on val-B

| model | version | macro F1 6 h | macro F1 12 h | macro F1 24 h | macro F1 48 h | Low recall 24 h | RMSE 24 h | pass at 24 h? |
|---|---|---|---|---|---|---|---|---|
| bilstm | step 5 | 0.8760 | 0.8716 | 0.8688 | 0.8339 | 0.9052 | 0.6885 | fail |
| bilstm | step 5b | 0.8790 | 0.8748 | 0.8740 | 0.8396 | 0.8947 | 0.6885 | fail |
| cnn_lstm | step 5 | 0.8812 | 0.8761 | 0.8744 | 0.8376 | 0.9095 | 0.6866 | PASS |
| cnn_lstm | step 5b | 0.8843 | 0.8793 | 0.8792 | 0.8438 | 0.8973 | 0.6866 | fail |
| persistence | - | 0.7297 | 0.7197 | 0.8704 | 0.8157 | 0.8980 | 0.7483 | (baseline) |

## Classification at 24 hours, val-B, all models

| model | accuracy | balanced acc | macro F1 | ROC-AUC | Low recall | Low precision | pass? |
|---|---|---|---|---|---|---|---|
| persistence | 0.883 | 0.871 | 0.8704 | - | 0.898 | 0.893 | (baseline) |
| linear | 0.873 | 0.868 | 0.8627 | 0.967 | 0.918 | 0.872 | fail |
| random_forest | 0.892 | 0.886 | 0.8826 | 0.977 | 0.918 | 0.900 | PASS |
| xgboost | 0.893 | 0.887 | 0.8839 | 0.978 | 0.915 | 0.905 | PASS |
| lightgbm | 0.892 | 0.887 | 0.8829 | 0.977 | 0.914 | 0.904 | PASS |
| bilstm (step 5) | 0.877 | 0.874 | 0.8688 | 0.971 | 0.905 | 0.897 | fail |
| cnn_lstm (step 5) | 0.883 | 0.880 | 0.8744 | 0.974 | 0.910 | 0.899 | PASS |
| bilstm (step 5b) | 0.883 | 0.877 | 0.8740 | 0.972 | 0.895 | 0.908 | fail |
| cnn_lstm (step 5b) | 0.889 | 0.882 | 0.8792 | 0.975 | 0.897 | 0.911 | fail |

## Pass/fail at every horizon, val-B

| model | 6 h | 12 h | 24 h | 48 h | passed |
|---|---|---|---|---|---|
| linear | PASS | PASS | fail | fail | 2/4 |
| random_forest | PASS | PASS | PASS | PASS | 4/4 |
| xgboost | PASS | PASS | PASS | PASS | 4/4 |
| lightgbm | PASS | PASS | PASS | PASS | 4/4 |
| bilstm (step 5) | PASS | PASS | fail | PASS | 3/4 |
| cnn_lstm (step 5) | PASS | PASS | PASS | PASS | 4/4 |
| bilstm (step 5b) | PASS | PASS | fail | PASS | 3/4 |
| cnn_lstm (step 5b) | PASS | PASS | fail | PASS | 3/4 |

## Did it work?

**bilstm**: macro F1 at 24 h went from 0.8688 to 0.8740 against persistence's 0.8704, and it still fails. **cnn_lstm**: macro F1 at 24 h went from 0.8744 to 0.8792 against persistence's 0.8704, and it still fails.

### The fix made it worse by the project's own rule

**`cnn_lstm` passed at 24 h before this step and fails after it.** Calibration did what it was asked to do - macro F1 rose from 0.8744 to 0.8792 - but it bought that by predicting Low less often, and Low recall fell from 0.910 to 0.897, below persistence's 0.898. The pass rule requires both, so the model is now worse where it counts.

The cause is a mismatch that is worth stating plainly: the calibration grid was told to maximise **macro F1**, but the project's rule is **macro F1 above persistence AND Low recall not below it**. Optimising one half of a two-part rule is exactly how you lose the other half. Macro F1 treats a missed low-oxygen hour and a false alarm as equally bad; this project does not, which is why the Low-recall condition was written into the criteria in the first place.

The uncalibrated step-5 predictions are kept in `results/val/archive/predictions_step5/`. Restoring them brings `cnn_lstm` back to passing at 24 h. That decision is left open rather than taken quietly here, because changing the calibration objective after seeing which way the number moved would be precisely the kind of search this step was meant to avoid.

Neither deep classifier clears the bar at 24 hours, and that is the result. The agreed plan was one principled attempt, and this was it: a louder classification loss and a calibration of the class priors, both chosen on val-A. Twenty-four hours ahead is the horizon where persistence is strongest, because DO runs on a daily cycle and "the same as this time yesterday" is a very good guess. The deep models beat it comfortably at 6, 12 and 48 hours and lose to it at 24. The tree models do clear it at 24 h, so the ensemble in step 6 has classifiers that pass at every horizon regardless.

A note on size: the differences being argued over here are in the third decimal place of macro F1, while the spread between random seeds in step 5 was 0.0014 to 0.0041. Any conclusion resting on a gap smaller than that is noise, which is exactly why this step reports on val-B rather than on the rows the settings were chosen from.

## Honest notes

- The step-5 numbers in this report are re-scored on val-B, so they differ from the step-5 report, which used the whole of 2024. Comparing a number here with a number there is a mistake.
- Calibration is fitted per model and per horizon on val-A, about 128,000 rows. It is a three-number adjustment, so the risk of overfitting it is low, but it is still a choice made on 2024 and will carry its optimism into any 2024 number.
- The regression side was protected, not improved: a louder classification loss was only allowed if no horizon's RMSE got worse by more than 1%.
- 2025 remains untouched.
