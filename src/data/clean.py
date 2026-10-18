"""
STEP 2.1 - Clean the raw data and build one hourly table.

Run from the project root:

    python -m src.data.clean

For every station in data/raw/stations.csv:

  1. Resample the 15-minute USGS readings to HOURLY MEANS (UTC).
  2. Set physically impossible values to NaN (ranges from config.yaml).
  3. Set obvious DO sensor spikes to NaN: a jump of more than 4 mg/L in one
     hour that reverses within 2 hours.
  4. Record, for every water variable, a 0/1 column `<var>_was_missing`
     BEFORE any filling.
  5. Fill only short gaps (<= 3 h) by carrying the last observed value
     forward. Longer gaps stay NaN, so nothing is invented, and a filled
     value at time t never contains information from after t.
  6. Drop a water variable at a station if it is more than 50% missing in the
     TRAIN years (2023). The column stays but is all NaN, and the drop is
     logged - 2024 and 2025 never influence this decision.
  7. Join the hourly weather (ERA5) and air quality (CAMS) on time.

Output: data/interim/hourly.parquet with one row per station-hour over the
whole 2023-2025 period, so later steps can use lags and rolling windows
without worrying about missing hours.

Safe to re-run: it simply rebuilds the file. Reads only; data/raw is untouched.
"""

import os

import numpy as np
import pandas as pd

from src.config import load_config, project_path, set_seeds

CFG = load_config()
P = CFG["paths"]
CL = CFG["cleaning"]

WATER_VARS = list(CFG["download"]["usgs_params"].values())
WEATHER_VARS = CFG["download"]["weather_vars"]
AIR_VARS = CFG["download"]["air_vars"]
MASK = CL["mask_suffix"]
TRAIN_YEARS = CFG["split"]["train_years"]

# The full hourly spine every station is put on, so that a "lag of 1 hour"
# really means one hour and not "the previous row that happens to exist".
FULL_INDEX = pd.date_range(CFG["period"]["start"],
                           pd.Timestamp(CFG["period"]["end"]) + pd.Timedelta("23h"),
                           freq=CL["resample_rule"])


def to_hourly(path, index=None):
    """
    Read one raw water file and average the 15-minute readings per hour.

    `index` is the hourly spine to put the station on. It defaults to the
    project's 2023-2025 period (FULL_INDEX); step 12's deployment layer
    passes the live window instead, so the same cleaning code serves both.
    """
    df = pd.read_csv(path, parse_dates=["time"]).set_index("time").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    hourly = df.resample(CL["resample_rule"]).mean()
    return hourly.reindex(FULL_INDEX if index is None else index)


def apply_ranges(df):
    """Set values outside the physically possible range to NaN."""
    n_removed = {}
    for var, rng in CL["valid_ranges"].items():
        if var not in df.columns:
            continue
        col = df[var]
        bad = pd.Series(False, index=df.index)
        if rng.get("min") is not None:
            bad |= (col <= rng["min"]) if rng.get("min_exclusive") else (col < rng["min"])
        if rng.get("max") is not None:
            bad |= col > rng["max"]
        bad &= col.notna()
        if bad.any():
            df.loc[bad, var] = np.nan
            n_removed[var] = int(bad.sum())
    return df, n_removed


def remove_spikes(df):
    """
    NaN-out single-hour spikes: the value jumps by more than `max_change`
    and comes back within `reversal_hours`. A real river does not move 4 mg/L
    in an hour and straight back again - that is the sensor, not the water.
    """
    spike_cfg = CL["spike"]
    limit = spike_cfg["max_change"]
    window = int(spike_cfg["reversal_hours"])
    n_removed = {}
    for var in spike_cfg["columns"]:
        if var not in df.columns:
            continue
        col = df[var]
        jump = col.diff()                      # change from the previous hour
        # does the jump reverse within the next `window` hours?
        reverses = pd.Series(False, index=col.index)
        for k in range(1, window + 1):
            back = col.shift(-k) - col         # change from t to t+k
            reverses |= (jump.abs() > limit) & (back.abs() > limit) & \
                        (np.sign(back) == -np.sign(jump))
        spike = reverses & col.notna()
        if spike.any():
            df.loc[spike, var] = np.nan
            n_removed[var] = int(spike.sum())
    return df, n_removed


def fill_short_gaps(df, cols):
    """
    Fill short gaps by carrying the LAST OBSERVED value forward, for at most
    `max_gap_fill_hours` hours. Longer gaps stay NaN.

    Why forward fill and not interpolation: interpolation averages the
    readings on both sides of a gap, so the filled value at time t would
    contain information from after t. A real early-warning system at time t
    only has the past, and the last known reading is exactly what it would
    use. Forward fill therefore cannot leak the future.
    """
    limit = int(CL["max_gap_fill_hours"])
    method = CL.get("gap_fill_method", "ffill")
    if method != "ffill":
        raise ValueError(f"cleaning.gap_fill_method must be 'ffill', got {method!r}")
    if limit <= 0:
        return df
    for c in cols:
        if c not in df.columns:
            continue
        col = df[c]
        if col.notna().sum() == 0:
            continue
        # ffill with a limit only carries a value forward `limit` rows, so a
        # longer gap keeps its NaNs from hour limit+1 onwards automatically.
        df[c] = col.ffill(limit=limit)
    return df


def clean_station(site, water_path, weather_path, air_path, index=None,
                  drop_variables=None):
    """
    Clean one station and return its hourly table (no station column yet).

    `index`          the hourly spine (defaults to the 2023-2025 period).
    `drop_variables` the variables to blank out at this station. The default
                     decides it here from the TRAIN years, which is what
                     step 2 did. Step 12 passes the list step 2 already
                     decided (data/interim/dropped_variables.csv) instead,
                     because a deployed model must be fed the same columns it
                     was trained on - re-deciding on a live window could
                     hand it a column that was all-NaN in training.
    """
    index = FULL_INDEX if index is None else index
    w = to_hourly(water_path, index)
    # keep only the water variables we know about, in a fixed order
    for v in WATER_VARS:
        if v not in w.columns:
            w[v] = np.nan
    w = w[WATER_VARS]

    w, removed_range = apply_ranges(w)
    w, removed_spike = remove_spikes(w)

    # masks BEFORE filling: 1 = there was no reading for this hour
    masks = pd.DataFrame({v + MASK: w[v].isna().astype("int8") for v in WATER_VARS},
                         index=w.index)

    w = fill_short_gaps(w, WATER_VARS)

    # Variables that are mostly missing in the TRAIN years are unusable.
    dropped = []
    if drop_variables is None:
        train_mask = w.index.year.isin(TRAIN_YEARS)
        for v in WATER_VARS:
            share_missing = float(w.loc[train_mask, v].isna().mean())
            if share_missing > CL["drop_variable_if_missing_above"]:
                w[v] = np.nan
                dropped.append({"site_no": site, "variable": v,
                                "missing_in_train": round(share_missing, 3)})
    else:
        for v in drop_variables:
            if v in w.columns:
                w[v] = np.nan
                dropped.append({"site_no": site, "variable": v,
                                "missing_in_train": None})

    # weather and air are already hourly and gapless
    wx = pd.read_csv(weather_path, parse_dates=["time"]).set_index("time").sort_index()
    aq = pd.read_csv(air_path, parse_dates=["time"]).set_index("time").sort_index()
    # the same hourly spine as the water, so the three line up exactly; a
    # different index here would union in the concat below and leave the
    # weather and air columns NaN for every hour the water actually covers
    wx = wx.reindex(index)[WEATHER_VARS]
    aq = aq.reindex(index)[AIR_VARS]

    out = pd.concat([w, masks, wx, aq], axis=1)
    out.index.name = "time"
    return out, dropped, removed_range, removed_spike


def main():
    set_seeds(CFG["seed"])
    stations = pd.read_csv(project_path(P["stations"]), dtype={"site_no": str})
    print(f"Cleaning {len(stations)} stations, "
          f"{CFG['period']['start']} .. {CFG['period']['end']}")

    frames, all_dropped = [], []
    totals = {"range": 0, "spike": 0}
    for i, s in stations.iterrows():
        site = str(s["site_no"])
        paths = [os.path.join(project_path(P[kind]), f"{site}.csv")
                 for kind in ["water", "weather", "air"]]
        if not all(os.path.exists(p) for p in paths):
            print(f"  [{i + 1}/{len(stations)}] {site}: missing a raw file, skipped")
            continue
        df, dropped, rm_range, rm_spike = clean_station(site, *paths)
        df.insert(0, "station", site)
        frames.append(df.reset_index())
        all_dropped += dropped
        totals["range"] += sum(rm_range.values())
        totals["spike"] += sum(rm_spike.values())

        do_cov = 100 * (1 - df["do_mgl"].isna().mean())
        note = ""
        if rm_range:
            note += f" | out-of-range {rm_range}"
        if rm_spike:
            note += f" | spikes {rm_spike}"
        if dropped:
            note += f" | DROPPED {[d['variable'] for d in dropped]}"
        print(f"  [{i + 1}/{len(stations)}] {site}: {len(df)} hours, "
              f"DO {do_cov:.1f}% present{note}")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["station", "time"]).reset_index(drop=True)

    out_path = project_path(P["hourly"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    panel.to_parquet(out_path, index=False)

    drop_path = project_path(P["dropped_vars"])
    pd.DataFrame(all_dropped, columns=["site_no", "variable", "missing_in_train"]) \
      .to_csv(drop_path, index=False)

    print(f"\nWrote {out_path}")
    print(f"  {len(panel):,} station-hours, {panel.shape[1]} columns, "
          f"{panel['station'].nunique()} stations")
    print(f"  values removed: {totals['range']:,} out of range, "
          f"{totals['spike']:,} spikes")
    print(f"  variables dropped at a station: {len(all_dropped)} "
          f"(logged in {drop_path})")
    if all_dropped:
        print(pd.DataFrame(all_dropped).to_string(index=False))
    print("\nMissing per column after cleaning (water variables):")
    for v in WATER_VARS:
        print(f"  {v:<16} {100 * panel[v].isna().mean():5.1f}% missing")
    print("\nNext: python -m src.features.build_features")


if __name__ == "__main__":
    main()
