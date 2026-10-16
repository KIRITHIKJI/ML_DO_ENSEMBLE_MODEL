"""
STEP 2.2-2.4 - Build features, targets and the train/val/test split.

Run from the project root (after src.data.clean):

    python -m src.features.build_features

THE ONE RULE: every feature at time t uses only data at times <= t, and is
computed PER STATION. Nothing is ever computed across stations, and no
statistic is computed over the whole dataset (no global means, no target
encoding), so there is no information leaking from the future or from the
validation and test years. Scaling for the deep-learning models happens
later and is fitted on the training years only.

Features (all per station, all backward-looking):
  water    current value, lags, rolling mean/min/max/std, changes,
           plus the `_was_missing` flags from the cleaning step
  DO       daily swing, hours since DO was last below Low, hours below Low
           in the last 24 h, and DO saturation (DO / solubility at the
           current water temperature, Benson-Krause)
  weather  current value, lags, 24 h means, precipitation sums, hours
           since it last rained
  air      current value, 24 h mean
  time     hour of day and day of year as sin/cos, month
  station  integer station code and the station's latitude / longitude

Targets, for each horizon h in config.yaml (6, 12, 24, 48 hours):
  y_reg_h  the hourly DO at t + h, taken from the OBSERVED values only: if
           DO at t + h was originally missing (and merely interpolated), the
           target is NaN
  y_cls_h  0 = Low (< 5), 1 = Medium (5-7), 2 = Good (> 7)

Rows are dropped when DO at time t is missing (a forecast needs a current
reading), for the first 72 h of each station (rolling windows not yet full),
and when the furthest target time (t + 48 h) would fall in the next split's
year, so no target ever crosses a split boundary.

Output:
    data/processed/train.parquet   (t in 2023)
    data/processed/val.parquet     (t in 2024)
    data/processed/test.parquet    (t in 2025 - the hold-out)
    data/processed/feature_groups.json
"""

import json
import os

import numpy as np
import pandas as pd

from src.config import load_config, project_path, set_seeds

CFG = load_config()
P = CFG["paths"]
F = CFG["features"]
MASK = CFG["cleaning"]["mask_suffix"]

WATER_VARS = F["water_vars"]
WEATHER_VARS = CFG["download"]["weather_vars"]
AIR_VARS = CFG["download"]["air_vars"]
HORIZONS = CFG["horizons"]
MAX_H = max(HORIZONS)
LOW = CFG["target"]["thresholds"]["low"]
GOOD = CFG["target"]["thresholds"]["good"]
DO = CFG["target"]["column"]

# year -> split name, straight from config.yaml
SPLIT_OF_YEAR = {}
for name, key in [("train", "train_years"), ("val", "val_years"),
                  ("test", "test_years")]:
    for y in CFG["split"][key]:
        SPLIT_OF_YEAR[int(y)] = name


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------
def hours_since_true(flag, cap):
    """
    For each hour, how many hours ago the condition was last true (0 if it is
    true right now). Only looks backwards. If it has never been true, the
    answer is the cap.

    `flag` must be a boolean Series on a complete hourly index.
    """
    position = np.arange(len(flag))
    last = pd.Series(np.where(flag.to_numpy(), position, np.nan),
                     index=flag.index).ffill()
    since = pd.Series(position, index=flag.index) - last
    return since.fillna(cap).clip(upper=cap)


def do_solubility(temp_c):
    """
    Oxygen solubility in fresh water at 1 atm, in mg/L, from the
    Benson-Krause equation as given in APHA Standard Methods:

        ln(DOs) = -139.34411 + 1.575701e5/T - 6.642308e7/T^2
                  + 1.243800e10/T^3 - 8.621949e11/T^4       (T in Kelvin)

    Used only as a denominator, to express DO as a share of what the water
    could hold at its current temperature.
    """
    t_k = temp_c + 273.15
    with np.errstate(invalid="ignore", divide="ignore"):
        ln_dos = (-139.34411
                  + 1.575701e5 / t_k
                  - 6.642308e7 / t_k ** 2
                  + 1.243800e10 / t_k ** 3
                  - 8.621949e11 / t_k ** 4)
        return np.exp(ln_dos)


# ---------------------------------------------------------------------------
# Features for ONE station
# ---------------------------------------------------------------------------
def water_features(df, out):
    """Current value, lags, rolling statistics and changes for each water variable."""
    for v in WATER_VARS:
        col = df[v]
        out[v] = col                                     # value at t
        for k in F["water_lags"]:
            out[f"{v}_lag{k}"] = col.shift(k)
        for w in F["water_roll_windows"]:
            roll = col.rolling(w, min_periods=1)         # includes t, never t+1
            for stat in F["water_roll_stats"]:
                if stat == "std":
                    # std needs at least two values to mean anything
                    out[f"{v}_roll{w}_std"] = col.rolling(w, min_periods=2).std()
                else:
                    out[f"{v}_roll{w}_{stat}"] = getattr(roll, stat)()
        for k in F["water_diffs"]:
            out[f"{v}_chg{k}"] = col - col.shift(k)
        # the cleaning step's "was there a reading this hour" flag
        out[v + MASK] = df[v + MASK]


def do_features(df, out):
    """DO-specific features: swing, time below the Low threshold, saturation."""
    do = df[DO]
    win = F["do_swing_window"]
    out[f"do_swing{win}"] = (do.rolling(win, min_periods=1).max()
                             - do.rolling(win, min_periods=1).min())
    low = do < LOW                                       # NaN counts as "not low"
    out["do_hours_since_low"] = hours_since_true(low, F["hours_since_low_cap"])
    lw = F["do_low_hours_window"]
    out[f"do_low_hours_{lw}"] = low.rolling(lw, min_periods=1).sum()
    if F["do_saturation"]:
        out["do_saturation"] = do / do_solubility(df["water_temp_c"])


def weather_features(df, out):
    """Weather: current value, lags, 24 h means, rain sums, hours since rain."""
    for v in WEATHER_VARS:
        col = df[v]
        out[v] = col
        for k in F["weather_lags"]:
            out[f"{v}_lag{k}"] = col.shift(k)
    w24 = F["weather_roll24_window"]
    for v in F["weather_roll24_vars"]:
        out[f"{v}_roll{w24}_mean"] = df[v].rolling(w24, min_periods=1).mean()
    rain = df[F["precip_var"]]
    for w in F["precip_sum_windows"]:
        out[f"precip_sum{w}"] = rain.rolling(w, min_periods=1).sum()
    out["hours_since_rain"] = hours_since_true(rain > F["rain_threshold_mm"],
                                               F["hours_since_rain_cap"])


def air_features(df, out):
    """Air quality: current value and 24 h mean."""
    w = F["air_roll_window"]
    for v in AIR_VARS:
        out[v] = df[v]
        out[f"{v}_roll{w}_mean"] = df[v].rolling(w, min_periods=1).mean()


def time_features(df, out):
    """
    Time of day and time of year as sin/cos pairs, so that 23:00 and 00:00
    are close together, plus the month as a plain number.
    """
    t = df.index
    hour, doy = t.hour.to_numpy(), t.dayofyear.to_numpy()
    out["hour_sin"] = pd.Series(np.sin(2 * np.pi * hour / 24), index=t)
    out["hour_cos"] = pd.Series(np.cos(2 * np.pi * hour / 24), index=t)
    out["doy_sin"] = pd.Series(np.sin(2 * np.pi * doy / 365.25), index=t)
    out["doy_cos"] = pd.Series(np.cos(2 * np.pi * doy / 365.25), index=t)
    out["month"] = pd.Series(t.month.to_numpy(), index=t)


def station_features(df, out, station_id, lat, lon):
    """Which station this is - a code for tree models, plus where it is."""
    out["station_id"] = pd.Series(station_id, index=df.index)
    out["lat"] = pd.Series(lat, index=df.index)
    out["lon"] = pd.Series(lon, index=df.index)


def targets(df, out):
    """
    DO at t + h, from OBSERVED values only. The cleaning step filled gaps of
    up to 3 h; those filled values are fine as inputs but must not be used as
    something to predict, so we undo the filling here using the mask.
    """
    observed = df[DO].where(df[DO + MASK] == 0)
    for h in HORIZONS:
        y = observed.shift(-h)
        out[f"y_reg_{h}"] = y
        # 0 = Low, 1 = Medium, 2 = Good; NaN stays NaN
        cls = pd.Series(np.nan, index=y.index)
        cls[y < LOW] = 0
        cls[(y >= LOW) & (y <= GOOD)] = 1
        cls[y > GOOD] = 2
        out[f"y_cls_{h}"] = cls


def build_station(df, station, station_id, lat, lon):
    """
    All features and targets for one station. `df` is that station's rows
    only, on a complete hourly index, sorted by time.
    """
    out = {}
    water_features(df, out)
    do_features(df, out)
    weather_features(df, out)
    air_features(df, out)
    time_features(df, out)
    station_features(df, out, station_id, lat, lon)
    targets(df, out)

    # build the table in one go instead of hundreds of inserts: much faster,
    # and it keeps the column order we built above
    feat = pd.DataFrame(out)
    feat.insert(0, "station", station)
    feat.insert(1, "time", df.index)
    return feat.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature groups (for later ablations)
# ---------------------------------------------------------------------------
def feature_groups(columns):
    """
    Split the column names into water / weather / air / time / station, so
    later steps can train on one modality at a time.
    """
    target_cols = [c for c in columns if c.startswith(("y_reg_", "y_cls_"))]
    meta = ["station", "time", "split"]
    time_cols = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "month"]
    station_cols = ["station_id", "lat", "lon"]

    air_cols, weather_cols, water_cols = [], [], []
    for c in columns:
        if c in meta or c in target_cols or c in time_cols or c in station_cols:
            continue
        base = c
        if any(c.startswith(v) for v in AIR_VARS):
            air_cols.append(c)
        elif any(base.startswith(v) for v in WEATHER_VARS) or \
                c.startswith(("precip_sum", "hours_since_rain")):
            weather_cols.append(c)
        else:
            water_cols.append(c)       # water variables, masks and DO features
    return {"water": water_cols, "weather": weather_cols, "air": air_cols,
            "time": time_cols, "station": station_cols,
            "targets": target_cols, "meta": meta}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    set_seeds(CFG["seed"])
    panel = pd.read_parquet(project_path(P["hourly"]))
    stations = pd.read_csv(project_path(P["stations"]), dtype={"site_no": str})
    coords = stations.set_index("site_no")[["lat", "lon"]].to_dict("index")
    # a stable integer code per station: position in the sorted station list
    codes = {s: i for i, s in enumerate(sorted(panel["station"].unique()))}

    print(f"Building features for {len(codes)} stations "
          f"({len(panel):,} station-hours)")
    print(f"Horizons: {HORIZONS} h | warm-up dropped: {F['warmup_hours']} h")

    pieces = {"train": [], "val": [], "test": []}
    counts = {"rows_in": 0, "no_do_now": 0, "warmup": 0, "purged": 0,
              "other_year": 0}

    for n, (station, g) in enumerate(panel.groupby("station", sort=True), 1):
        g = g.sort_values("time").set_index("time")
        feat = build_station(g, station, codes[station],
                             coords[station]["lat"], coords[station]["lon"])
        counts["rows_in"] += len(feat)

        t = feat["time"]
        keep = pd.Series(True, index=feat.index)

        # a forecast needs a current DO reading
        no_do = feat[DO].isna()
        counts["no_do_now"] += int((keep & no_do).sum())
        keep &= ~no_do

        # the first hours of a station: rolling windows are not full yet
        warm = t < t.min() + pd.Timedelta(hours=F["warmup_hours"])
        counts["warmup"] += int((keep & warm).sum())
        keep &= ~warm

        # PURGE: no target may cross a split boundary. If the furthest target
        # time (t + 48 h) lands in a year belonging to a different split, the
        # row goes - that is the last 48 h of 2023 and of 2024.
        split_now = t.dt.year.map(SPLIT_OF_YEAR)
        split_far = (t + pd.Timedelta(hours=MAX_H)).dt.year.map(SPLIT_OF_YEAR)
        crosses = split_far.notna() & (split_far != split_now)
        counts["purged"] += int((keep & crosses).sum())
        keep &= ~crosses

        # anything outside the three split years (should not happen)
        unknown = split_now.isna()
        counts["other_year"] += int((keep & unknown).sum())
        keep &= ~unknown

        feat = feat[keep].copy()
        feat["split"] = split_now[keep].to_numpy()

        # keep the files small: 32-bit floats are plenty for these values
        float_cols = [c for c in feat.columns
                      if c not in ("station", "time", "split")
                      and not c.startswith("y_cls_")]
        feat[float_cols] = feat[float_cols].astype("float32")
        for h in HORIZONS:
            feat[f"y_cls_{h}"] = feat[f"y_cls_{h}"].astype("Int8")
        feat["station_id"] = feat["station_id"].astype("int16")
        feat["month"] = feat["month"].astype("int8")

        for name in pieces:
            part = feat[feat["split"] == name]
            if len(part):
                pieces[name].append(part)
        print(f"  [{n}/{len(codes)}] {station}: {len(feat):,} rows kept")

    # ---------------- write the three splits ----------------
    groups = None
    print()
    for name in ["train", "val", "test"]:
        df = pd.concat(pieces[name], ignore_index=True)
        df = df.sort_values(["station", "time"]).reset_index(drop=True)
        if groups is None:
            groups = feature_groups(list(df.columns))
        path = project_path(P[name])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"Wrote {path}: {len(df):,} rows x {df.shape[1]} columns "
              f"({df['time'].min():%Y-%m-%d} .. {df['time'].max():%Y-%m-%d})")
        pieces[name] = None      # free the memory

    gpath = project_path(P["feature_groups"])
    with open(gpath, "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2)
    n_feat = sum(len(groups[g]) for g in ["water", "weather", "air", "time", "station"])
    print(f"Wrote {gpath}")
    print("  features per group: "
          + ", ".join(f"{g}={len(groups[g])}"
                      for g in ["water", "weather", "air", "time", "station"])
          + f"  -> {n_feat} features total")

    print("\nRows dropped on the way:")
    print(f"  station-hours in           {counts['rows_in']:>10,}")
    print(f"  no DO reading at t        -{counts['no_do_now']:>10,}")
    print(f"  first {F['warmup_hours']} h of a station    "
          f"-{counts['warmup']:>10,}")
    print(f"  purged at split boundary  -{counts['purged']:>10,}")
    if counts["other_year"]:
        print(f"  outside the split years   -{counts['other_year']:>10,}")
    print("\nNext: python -m src.features.make_report")


if __name__ == "__main__":
    main()
