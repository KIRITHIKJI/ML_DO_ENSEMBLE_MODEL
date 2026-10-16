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
| train | [2023] | 254,960 | 30 | 2023-01-04 .. 2023-12-29 |
| val | [2024] | 253,336 | 30 | 2024-01-01 .. 2024-12-29 |
| test | [2025] | 253,741 | 30 | 2025-01-01 .. 2025-12-31 |

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
| train | 6 h | 254,180 | 99.7 | 19.3 | 27.5 | 53.2 |
| train | 12 h | 253,854 | 99.6 | 19.3 | 27.6 | 53.1 |
| train | 24 h | 253,307 | 99.4 | 19.3 | 27.5 | 53.1 |
| train | 48 h | 252,607 | 99.1 | 19.4 | 27.5 | 53.1 |
| val | 6 h | 252,547 | 99.7 | 20.5 | 24.3 | 55.2 |
| val | 12 h | 252,187 | 99.5 | 20.5 | 24.3 | 55.2 |
| val | 24 h | 251,610 | 99.3 | 20.5 | 24.3 | 55.2 |
| val | 48 h | 250,599 | 98.9 | 20.6 | 24.3 | 55.1 |
| test | 6 h | 252,857 | 99.7 | 19.7 | 24.9 | 55.4 |
| test | 12 h | 252,390 | 99.5 | 19.7 | 24.9 | 55.4 |
| test | 24 h | 251,594 | 99.2 | 19.7 | 24.9 | 55.3 |
| test | 48 h | 250,201 | 98.6 | 19.8 | 25.0 | 55.2 |

## Missing values after feature building

Share of rows with no value, averaged over the columns of each group (train split):

| group | columns | mean % missing | worst column % |
|---|---|---|---|
| water | 148 | 10.5 | 35.3 |
| weather | 39 | 0.0 | 0.0 |
| air | 16 | 0.0 | 0.0 |
| time | 5 | 0.0 | 0.0 |
| station | 3 | 0.0 | 0.0 |

### The 10 worst columns (train split)

| column | group | % missing |
|---|---|---|
| turbidity_fnu_chg24 | water | 35.3 |
| turbidity_fnu_chg6 | water | 34.8 |
| turbidity_fnu_lag48 | water | 34.8 |
| turbidity_fnu_lag24 | water | 34.7 |
| turbidity_fnu_lag12 | water | 34.6 |
| turbidity_fnu_lag6 | water | 34.5 |
| turbidity_fnu_chg1 | water | 34.5 |
| turbidity_fnu_lag3 | water | 34.5 |
| turbidity_fnu_lag2 | water | 34.5 |
| turbidity_fnu_lag1 | water | 34.5 |

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
| 01484080 | discharge_cfs | 89 |
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

## Known caveat: short-gap interpolation

Cleaning fills water-variable gaps of up to 3 h by linear interpolation, as specified. Linear interpolation looks at the reading on *both* sides of a gap, so a filled input value at time t can contain up to 3 h of future sensor information. This affects inputs only: the targets are taken from observed values (`*_was_missing == 0`), and the `<var>_was_missing` columns mark every filled hour, so the effect can be measured or the filling switched off (`cleaning.max_gap_fill_hours: 0`). It touches about 0.1% of training rows for DO.


## Note on negative discharge

The cleaning rule `discharge_cfs >= 0` removes about 72,000 hourly values, almost all of them at tidal stations (Savannah River, Middle River, Waccamaw, Murderkill), where a negative value is the real tide pushing water upstream rather than a sensor fault. Those hours are marked by `discharge_cfs_was_missing = 1`. To keep them instead, set `cleaning.valid_ranges.discharge_cfs.min: null` in config.yaml and re-run the cleaning step.
