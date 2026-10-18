# Model card - DO early warning, deployed models

**Version id** `final-35ab6d1d27d1`

The id is the SHA-256 over the per-file hashes of `models/final`, truncated to 12 characters. It changes if and only if a model file changes, and it is written into **every row of every forecast** this system produces (`results/live/forecast_*.csv` and `results/live/log.parquet`), so any past forecast can be matched to the models that made it. The previous version is kept under `models/archive/<version id>/` whenever a retrain is promoted.


## What it is

A forecast of dissolved oxygen (DO) in a river 6, 12, 24 and 48 hours ahead, and a warning when the water is likely to drop below **5.0 mg/L**.

- **Inputs**: 211 engineered features at the forecast hour (water sensors, ERA5 weather, CAMS air quality, time of day and year, station identity) plus the last 72 hours of raw hourly readings for the two recurrent members. Every feature uses only data at or before the forecast hour.
- **Outputs**: predicted DO in mg/L, a probability for each of Low (< 5.0), Medium (5.0-7.0) and Good (> 7.0), and a yes/no warning.
- **Structure**: 5 members (random_forest, xgboost, lightgbm, bilstm, cnn_lstm) combined by a **stacking** meta-model fitted on 2024, with a warning rule on top: *warn when P(Low) >= tau*.
- **The warning rule** is the single most valuable part. tau was chosen on out-of-fold 2024 probabilities as the threshold catching the most onsets at a false-alarm rate at or below 1.8% - deliberately tighter than the 2% the system promises.


The thresholds in force, from `models/ensemble/warning_thresholds.json`:

| horizon | tau |
|---|---|
| 6 h | 0.27 |
| 12 h | 0.24 |
| 24 h | 0.24 |
| 48 h | 0.35 |

## Training data

- **30 USGS river stations** across 15 US states, hourly, UTC.
- **2023-01-01 to 2025-12-31** - all three years. These are the *stage-3* models: the members were retrained on 2023 + 2024 + 2025 so a live forecast uses every hour ever recorded.
- **Three sources**: USGS instantaneous values (DO, water temperature, conductivity, pH, turbidity, discharge), ERA5 reanalysis via Open-Meteo (8 variables), CAMS via Open-Meteo (8 variables).
- **Nothing was tuned on that third year.** Every hyperparameter, every number of boosting rounds and every epoch count came from `FROZEN.md`, committed 2026-10-16 22:51:20 at commit `e4c53b8209c7`, which was written before any 2025 target was read. There is no early stopping in these models, because the only year left to watch would have been the test year.

- **Stations are chosen, not arbitrary**: continuous DO, water temperature and conductivity, at least 80% DO coverage in *every* year, and real low-oxygen episodes in at least two of the three. The list is `data/raw/stations.csv`.


## Test results

Two separate things are measured here, and they must not be confused:

1. **The method**, tested on **2025**. Those numbers come from the *stage-2* models (trained on 2023+2024), which is what makes 2025 a hold-out for them. They are the paper's result and they are **not this model's** result.
2. **These models**, tested on **2026**. The models in this folder trained on 2025, so 2025 cannot score them. 2026 can, and the criteria were pre-registered in `reports/plan_2026.md` and committed before any 2026 value was read.


### The method, on 2025 (stage-2 models, not these)

| horizon | R2 | RMSE | macro F1 | onset recall | false alarms |
|---|---|---|---|---|---|
| 6 h | 0.971 | 0.514 | 0.905 | 0.861 | 1.75% |
| 12 h | 0.962 | 0.588 | 0.897 | 0.861 | 1.73% |
| 24 h | 0.951 | 0.672 | 0.886 | 0.496 | 1.50% |
| 48 h | 0.920 | 0.853 | 0.856 | 0.485 | 1.33% |

Of the 230 real low-oxygen events in 2025, 52% were warned about in advance, median lead 20 h.


### These models, on 2026 (the deployed models' own test)

| horizon | R2 | RMSE | macro F1 | onset recall | false alarms |
|---|---|---|---|---|---|
| 6 h | 0.955 | 0.617 | 0.892 | 0.884 | 1.86% |
| 12 h | 0.945 | 0.686 | 0.884 | 0.878 | 2.02% |
| 24 h | 0.936 | 0.738 | 0.872 | 0.479 | 1.65% |
| 48 h | 0.905 | 0.901 | 0.837 | 0.490 | 1.75% |

2026-01-01 to 2026-10-13 UTC, 181,203 scored station-hours, 28 of 30 stations. **Verdict against the pre-registered criteria: PARTIAL** - the 2% false-alarm promise holds (1.65% at 24 h), but rmse at 24 h falls outside the 2025 interval.

Of the 204 low-oxygen events in the 2026 window, 52% were warned about in advance, median lead 18 h.

Full tables, intervals and the per-station breakdown: `reports/final_results.md`, section **Deployed models on 2026**, and `results/live_2026/`.


### The cost of running live

ERA5 runs about 5 days behind, so a live forecast uses the Open-Meteo forecast API for its most recent hours. Scoring the same 18,673 station-hours both ways:

| horizon | RMSE ERA5 | RMSE forecast | diff | onset ERA5 | onset forecast | diff |
|---|---|---|---|---|---|---|
| 6 h | 0.543 | 0.541 | -0.002 | 0.899 | 0.903 | +0.003 |
| 12 h | 0.582 | 0.581 | -0.001 | 0.864 | 0.866 | +0.002 |
| 24 h | 0.626 | 0.625 | -0.001 | 0.412 | 0.412 | +0.000 |
| 48 h | 0.786 | 0.785 | -0.001 | 0.494 | 0.500 | +0.006 |

Every hour in every live file is tagged with its source, so a forecast made on forecast weather is never mistaken for one made on a reanalysis.


## Intended use

**Use it for**: an early warning at a river station that already has a DO sensor, so an operator gets hours of notice instead of a reading after the fact - time to open a sluice, pause a discharge, or send someone to look. It is designed to be run **hourly** (`run_live.bat`, or `python -m src.deploy.run`), with the forecasts logged and scored as the sensors catch up (`python -m src.deploy.monitor`).

**Do not use it for**:

- a river with no DO sensor. Every station in the test set is also in the training set; nothing here measures transfer to an unseen river, and the station identity is an input the models use.
- rivers outside the US, or outside the USGS network. The water inputs are USGS instantaneous values; another network would need its own freeze and its own untouched test year.
- a regulatory or safety-of-life decision on its own. It is a forecast with a measured false-alarm rate, not a measurement.
- scoring the method on 2025. These models trained on 2025. The app enforces this: a 2025 hour is forecast with the stage-2 models only.


## Limitations

- **The DO value is predicted less accurately on 2026 than on 2025.** At 24 h, RMSE went from 0.672 to 0.738 mg/L - about 10% worse - and macro F1 slipped as well. RMSE is outside the 2025 bootstrap interval at every horizon. That is the honest size of the drop between a year the method was tested on and a year nobody prepared for; the pre-registered verdict is PARTIAL, and the metrics it marks as degraded are f1_macro, false_alarm_rate, rmse.
- **What did hold is the thing the system is for.** Onset recall stayed inside the 2025 interval at every horizon (0.496 to 0.479 at 24 h), and the share of real events warned about in advance barely moved. Accuracy on the DO *value* degraded more than the *decision* did.
- **The false-alarm promise was exceeded at 12 h (2.02%)**, against the 2% budget. It held at the 24 h headline horizon (1.65%, against 1.50% on 2025), which is what the pre-registered pass/fail criterion rests on, but the other horizons are reported as they came out and this is a real exceedance, not a rounding one. tau was chosen on 2024 and is **not** being re-chosen on 2026.
- **Low DO is rare, and onsets are rarer.** Overall accuracy flatters any model here; the numbers that matter are onset recall and the false-alarm rate, which is why they are the ones the system is judged on.
- **24 h is the hard horizon.** 6 and 12 h are much easier - the river's own recent history carries most of the signal - and 48 h is harder still.
- **Stations differ enormously.** Per-station onset recall spans most of the possible range; a station-level table is in the report and in the app. A single headline number is an average over very unlike rivers.
- **The deep members are not bit-for-bit reproducible.** cuDNN has no deterministic recurrent kernel, so the system averages three seeds; the remaining spread was measured rather than assumed (design check 6 in the report).
- **A sensor that STOPS reporting is caught; one that lies is not.** Past 6 h of silence a station gets `status` "stale - sensor offline" and no prediction at all, because every input is a lag of the DO it last sent. A stuck or drifting sensor that keeps reporting is fed straight in; `python -m src.deploy.monitor` flags features that leave their training range, which is a warning about the data, not a repair of it.
- **A predictable daily cycle can look like an event.** At the tidal stations on the lower Savannah (02198920 and 02198950) the DO follows a twice-daily cycle, and the rule fires on the dips in it: they appear as rows whose `alert_type` is `early_warning`, and the forecast is correct. They are correct forecasts of a normal cycle, not deterioration events. Nothing here tells the two apart, so a count of early warnings at a tidal station is not a count of events, and an operator there should read the tide alongside the row. Why the oxygen moves on that cycle is not something this project tested, and no mechanism is claimed here.
- **An alert is not automatically an early warning.** `alert_type` separates water predicted to go Low (`early_warning`) from water already below the threshold (`ongoing_low`), which an operator can read off the sensor. Only the first is what this model is for, and the monitoring scores it on its own.
- **Gaps are carried forward, not interpolated**, for at most 3 h. Longer gaps stay missing and the hour is not forecast; `data_age_hours` in every forecast row says how stale the newest reading was.
- **Air quality earns almost nothing.** The ablation in the report shows weather pays for itself and CAMS adds very little - worth knowing before anyone pays for an air-quality feed.

## How it is run and watched

| | |
|---|---|
| forecast | `python -m src.deploy.run` → `results/live/forecast_<UTC timestamp>.csv` |
| schedule | `run_live.bat`, hourly, Windows Task Scheduler |
| log | `results/live/log.parquet` |
| stale-data guard | no forecast when the newest DO reading is older than **6 h**; the row carries `stale - sensor offline` and never a warning |
| alert types | `early_warning` safe water predicted to go Low, the only row an operator could not read off the sensor; `ongoing_low` already below 5.0 mg/L and the rule fires; `none` forecast made, rule did not fire; `stale` no forecast was made |
| the class now | `class_now` is written as Low/Medium/Good, not as 0/1/2. Forecast files written before that change carry the numbers and are still read |
| monitoring | `python -m src.deploy.monitor` → `results/live/monitor.csv`, `results/live/drift.csv` |
| drift limits | false alarms > 2% over 30 days; features outside the training range |
| API | `python -m src.deploy.api` → /health, /warnings, /forecast/{station} |
| retraining | `python -m src.deploy.retrain` - gated: a candidate replaces these models only if it is at least as good on the most recent 2 months it did not train on, on RMSE **and** false-alarm rate, at **every** horizon |
| settings | `config.yaml` (the science) + `config_deploy.yaml` (the deployment) |


*This card is generated by `python -m src.deploy.model_card` from the model folder and the results files, so none of its numbers is typed by hand.*
