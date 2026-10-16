# Feature report

- Built from `data/interim/hourly.parquet` by `src/features/build_features.py`
- Horizons: **[6, 12, 24, 48] hours ahead**
- Classes: Low < 5.0 mg/L, Medium 5.0-7.0, Good > 7.0
- Every feature at time t uses only data at times <= t, computed per station. No statistic is computed over the whole dataset.
- Rows with no DO reading at t are dropped; rows with a missing target are kept but must be skipped for that horizon.
- **The test columns are descriptive only. 2025 must not influence any choice until the final step.**

## Rows per split

| split | years | rows | stations | time range |
|---|---|---|---|---|
| train | [2023] | 255,233 | 30 | 2023-01-04 .. 2023-12-29 |
| val | [2024] | 253,603 | 30 | 2024-01-01 .. 2024-12-29 |
| test | [2025] | 253,954 | 30 | 2025-01-01 .. 2025-12-31 |

## Features per group

| group | columns | examples |
|---|---|---|
| water | 148 | do_mgl, do_mgl_lag1, do_mgl_lag2 ... |
| weather | 39 | temperature_2m, temperature_2m_lag1, temperature_2m_lag6 ... |
| air | 16 | pm10, pm10_roll24_mean, pm2_5 ... |
| time | 5 | hour_sin, hour_cos, doy_sin ... |
| station | 3 | station_id, lat, lon |
| **total features** | 211 |  |
| targets | 8 | y_reg_6, y_cls_6, y_reg_12, y_cls_12 ... |

## Usable targets, and their Low / Medium / Good shares

| split | horizon | rows with target | % of split | % Low | % Medium | % Good |
|---|---|---|---|---|---|---|
| train | 6 h | 254,253 | 99.6 | 19.3 | 27.5 | 53.2 |
| train | 12 h | 253,954 | 99.5 | 19.3 | 27.6 | 53.1 |
| train | 24 h | 253,435 | 99.3 | 19.3 | 27.5 | 53.2 |
| train | 48 h | 252,790 | 99.0 | 19.4 | 27.5 | 53.1 |
| val | 6 h | 252,598 | 99.6 | 20.5 | 24.3 | 55.2 |
| val | 12 h | 252,259 | 99.5 | 20.5 | 24.3 | 55.2 |
| val | 24 h | 251,715 | 99.3 | 20.5 | 24.3 | 55.2 |
| val | 48 h | 250,736 | 98.9 | 20.6 | 24.3 | 55.1 |
| test | 6 h | 252,900 | 99.6 | 19.7 | 24.9 | 55.4 |
| test | 12 h | 252,447 | 99.4 | 19.7 | 24.9 | 55.4 |
| test | 24 h | 251,671 | 99.1 | 19.7 | 24.9 | 55.3 |
| test | 48 h | 250,317 | 98.6 | 19.8 | 25.0 | 55.2 |

## Missing values after feature building

Share of rows with no value, averaged over the columns of each group (train split):

| group | columns | mean % missing | worst column % |
|---|---|---|---|
| water | 148 | 9.8 | 35.1 |
| weather | 39 | 0.0 | 0.0 |
| air | 16 | 0.0 | 0.0 |
| time | 5 | 0.0 | 0.0 |
| station | 3 | 0.0 | 0.0 |

### The 10 worst columns (train split)

| column | group | % missing |
|---|---|---|
| turbidity_fnu_chg24 | water | 35.1 |
| turbidity_fnu_chg6 | water | 34.6 |
| turbidity_fnu_lag48 | water | 34.6 |
| turbidity_fnu_lag24 | water | 34.6 |
| turbidity_fnu_lag12 | water | 34.5 |
| turbidity_fnu_lag6 | water | 34.4 |
| turbidity_fnu_chg1 | water | 34.4 |
| turbidity_fnu_lag3 | water | 34.4 |
| turbidity_fnu_lag2 | water | 34.4 |
| turbidity_fnu_lag1 | water | 34.4 |

Turbidity and discharge are the worst: several stations have no such sensor at all, and the cleaning step blanks a variable that is more than half missing in the training year. Tree models handle these NaNs natively; the deep-learning models will need them imputed, fitted on train only.

## Water variables dropped per station (cleaning step)

A variable more than 50% missing in the training years is blanked for that station. Measured on [2023] only, so the validation and test years never affect the choice.

| station | variable | % missing in train |
|---|---|---|
| 02239501 | turbidity_fnu | 100 |
| 02322700 | turbidity_fnu | 100 |
| 02110704 | discharge_cfs | 100 |
| 02110550 | ph | 68 |
| 01484080 | turbidity_fnu | 100 |
| 01484080 | discharge_cfs | 79 |
| 02264051 | ph | 100 |
| 02264051 | turbidity_fnu | 100 |
| 05474500 | discharge_cfs | 100 |
| 03430200 | turbidity_fnu | 100 |
| 03612600 | water_temp_c | 76 |
| 03612600 | turbidity_fnu | 76 |
| 03431083 | discharge_cfs | 100 |
| 07049000 | turbidity_fnu | 62 |
| 07239450 | turbidity_fnu | 100 |
| 01484525 | turbidity_fnu | 100 |

## Gap filling uses the past only

Cleaning fills water-variable gaps of up to 3 h by carrying the last observed value FORWARD (`cleaning.gap_fill_method: ffill`). Linear interpolation was used at first, but it averages the readings on both sides of a gap, so a filled input value at time t would have contained up to 3 h of future sensor information. A forward fill is exactly what a real-time system would have at time t. Every filled hour is still marked by `<var>_was_missing = 1`; this affects about 0.2% of training rows for DO.


## Negative discharge is kept

At tidal stations (Savannah River, Middle River, Waccamaw, Murderkill) a negative discharge is the real tide pushing water upstream, not a sensor fault, and tidal flushing is physically linked to DO. The lower bound on `discharge_cfs` is therefore `null` in config.yaml. That keeps about 72,000 hourly values an earlier version discarded: discharge is now 15.8% missing in the training split instead of 22%.
