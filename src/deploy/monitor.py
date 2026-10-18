"""
STEP 12.6 - monitoring the deployed system.

    python -m src.deploy.monitor

A forecast is only scoreable once the hour it was about has arrived and the
sensor has reported it. This script joins the logged forecasts
(results/live/log.parquet) to the DO that actually turned up in the live
water data, and reports:

  * performance over a rolling `monitor.window_days` window - RMSE, macro
    F1, onset recall and false-alarm rate, per station and overall, at every
    horizon;
  * DATA DRIFT: features outside the range they had in training;
  * PERFORMANCE DRIFT: a false-alarm rate above the 2% the system promises,
    measured over the same 30-day window.

The metrics are `src/evaluation/metrics.py`. Nothing is retrained here and
nothing is tuned; a flag is a flag, and the decision to act on it belongs to
`src/deploy/retrain.py` and its promotion gate.

Writes results/live/monitor.csv and results/live/drift.csv.
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.config import project_path
from src.evaluation.metrics import (classification_metrics,
                                    early_warning_metrics, regression_metrics,
                                    to_class)

from src.deploy import common as D

CFG = D.CFG
HORIZONS = D.HORIZONS
LOW_CLASS = D.LOW_CLASS
MON = D.MON
RUN = D.RUN


# ---------------------------------------------------------------------------
# Truth: what the oxygen actually did
# ---------------------------------------------------------------------------
def observed():
    """
    The DO actually recorded, from the live hourly panel. Only hours the
    sensor really reported count - a value the cleaning step carried forward
    over a short gap is fine as an input but is not something to score a
    forecast against, which is the rule step 2 used for the training
    targets too.
    """
    frames = []
    for mode in ("live", "history"):
        path = D.dpath(D.DATA["hourly"], mode=mode)
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))
    if not frames:
        return None
    mask = CFG["cleaning"]["mask_suffix"]
    panel = pd.concat(frames, ignore_index=True)
    panel = panel[["station", "time", D.DO, D.DO + mask]].dropna(subset=[D.DO])
    panel = panel[panel[D.DO + mask] == 0]
    panel = (panel.sort_values(["station", "time"])
             .drop_duplicates(["station", "time"], keep="last"))
    return panel[["station", "time", D.DO]].rename(
        columns={"time": "valid_at", D.DO: "y_true_reg"})


def log():
    path = project_path(RUN["log_parquet"])
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    for c in ("issued_at", "valid_at", "observed_at"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def joined(now=None):
    """
    Logged forecasts with the truth attached, for the rows that can be
    scored. `class_now` is the class at the hour the forecast was made,
    which is what the early-warning metrics need.
    """
    lg, obs = log(), observed()
    if lg is None or obs is None or not len(lg):
        return None
    d = lg.merge(obs, on=["station", "valid_at"], how="left")
    d["y_true_cls"] = to_class(d["y_true_reg"])
    if "do_now_mgl" in d.columns:
        d["class_now"] = to_class(d["do_now_mgl"])
    else:
        d["class_now"] = np.nan
    d["pred_warn"] = np.where(d["warning"], LOW_CLASS, np.nan)
    # a non-warning row still has a predicted class: the better of Medium
    # and Good, which is what apply_warning would have returned. Only the
    # warning flag is logged, so Medium is the conservative reading of "not
    # Low" for the hours nothing fired.
    d.loc[~d["warning"].astype(bool), "pred_warn"] = 1
    return d


# ---------------------------------------------------------------------------
# Rolling performance
# ---------------------------------------------------------------------------
def score_window(d, label, station=None):
    """Every horizon, for one slice of the joined log."""
    rows = []
    for h in HORIZONS:
        g = d[(d["horizon"] == h) & d["y_true_reg"].notna()]
        if station is not None:
            g = g[g["station"] == station]
        n = len(g)
        row = {"window": label, "station": station or "ALL", "horizon": h,
               "n": n}
        if n < int(MON["min_rows"]):
            row.update({"rmse": np.nan, "r2": np.nan, "f1_macro": np.nan,
                        "onset_recall": np.nan, "false_alarm_rate": np.nan,
                        "n_onsets": 0, "judged": False,
                        "flag_performance_drift": False})
            rows.append(row)
            continue
        rm = regression_metrics(g["y_true_reg"], g["pred_do_mgl"])
        cm = classification_metrics(g["y_true_cls"], g["pred_warn"])
        ew = early_warning_metrics(g["class_now"], g["y_true_cls"],
                                   g["pred_warn"])
        far = ew["false_alarm_rate"]
        row.update({"rmse": rm["rmse"], "r2": rm["r2"],
                    "f1_macro": cm["f1_macro"],
                    "onset_recall": ew["onset_recall"],
                    "false_alarm_rate": far,
                    "n_onsets": ew["n_onsets"], "judged": True,
                    "flag_performance_drift": bool(
                        far == far and far > float(MON["max_false_alarm_rate"]))})
        rows.append(row)
    return rows


def performance(d, now=None):
    """The rolling window, overall and per station."""
    now = pd.Timestamp.utcnow().tz_localize(None) if now is None \
        else pd.Timestamp(now)
    days = int(MON["window_days"])
    recent = d[d["valid_at"] >= now - pd.Timedelta(days=days)]
    label = f"last_{days}_days"
    rows = score_window(recent, label)
    for station in sorted(set(recent["station"])):
        rows += score_window(recent, label, station)
    rows += score_window(d, "all_time")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data drift
# ---------------------------------------------------------------------------
def training_ranges(features=None):
    """
    The min and max each watched feature had in TRAINING (2023+2024+2025 -
    the years models/final learned from). Read once from data/processed.
    """
    cols = [c for c in MON["drift_features"]]
    frames = []
    for split in ("train", "val", "test"):
        path = project_path(CFG["paths"][split])
        if not os.path.exists(path):
            continue
        have = pd.read_parquet(path, columns=["station"] + cols)
        frames.append(have)
    if not frames:
        return None
    t = pd.concat(frames, ignore_index=True)
    out = []
    for c in cols:
        v = pd.to_numeric(t[c], errors="coerce")
        out.append({"feature": c, "train_min": float(v.min()),
                    "train_max": float(v.max()),
                    "train_mean": float(v.mean())})
    return pd.DataFrame(out)


def data_drift(now=None):
    """
    Which watched features fall outside the range they had in training, and
    how often. The tolerance is a fraction of the training spread, so a
    value a hair past the training maximum is not a drift event; a sensor
    reading nothing like anything the model ever saw is.
    """
    ranges = training_ranges()
    if ranges is None:
        return pd.DataFrame()
    path = D.dpath(D.DATA["features"], mode="live")
    if not os.path.exists(path):
        return pd.DataFrame()
    cols = list(ranges["feature"])
    live = pd.read_parquet(path, columns=["station", "time"] + cols)
    now = pd.Timestamp.utcnow().tz_localize(None) if now is None \
        else pd.Timestamp(now)
    live = live[live["time"] >= now - pd.Timedelta(days=int(MON["window_days"]))]
    tol = float(MON["range_tolerance"])
    limit = float(MON["max_out_of_range_share"])
    rows = []
    for r in ranges.itertuples():
        spread = r.train_max - r.train_min
        lo = r.train_min - tol * spread
        hi = r.train_max + tol * spread
        v = pd.to_numeric(live[r.feature], errors="coerce")
        seen = v.notna()
        out_of = seen & ((v < lo) | (v > hi))
        share = float(out_of.sum() / seen.sum()) if seen.sum() else np.nan
        worst = (live.loc[out_of, "station"].value_counts().index[0]
                 if out_of.any() else "")
        rows.append({"feature": r.feature, "n_rows": int(seen.sum()),
                     "train_min": r.train_min, "train_max": r.train_max,
                     "allowed_min": lo, "allowed_max": hi,
                     "live_min": float(v.min()) if seen.any() else np.nan,
                     "live_max": float(v.max()) if seen.any() else np.nan,
                     "share_out_of_range": share,
                     "worst_station": worst,
                     "flag_data_drift": bool(share == share and share > limit)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--now", default=None,
                    help="pretend it is this UTC time (for testing)")
    args = ap.parse_args()
    now = pd.Timestamp(args.now) if args.now else None

    D.log("=== STEP 12.6: monitoring ===")
    out_path = project_path(MON["out_csv"])
    drift_path = project_path(MON["drift_csv"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    d = joined(now)
    if d is None:
        D.log("No forecasts logged yet (results/live/log.parquet is "
              "missing or empty). Run `python -m src.deploy.run` first.")
        pd.DataFrame(columns=["window", "station", "horizon", "n"]).to_csv(
            out_path, index=False)
    else:
        scoreable = int(d["y_true_reg"].notna().sum())
        D.log(f"  {len(d):,} logged forecast rows, {scoreable:,} with the "
              f"real DO already in ({d['station'].nunique()} stations)")
        perf = performance(d, now)
        perf.to_csv(out_path, index=False)
        head = 24 if 24 in HORIZONS else HORIZONS[len(HORIZONS) // 2]
        overall = perf[(perf["station"] == "ALL")
                       & (perf["window"] != "all_time")]
        D.log(f"\n  rolling {MON['window_days']} days, all stations")
        D.log(f"  {'h':>4}{'n':>8}{'RMSE':>9}{'macroF1':>9}{'onset':>8}"
              f"{'falseAl':>9}  flag")
        for r in overall.itertuples():
            if not r.judged:
                D.log(f"  {r.horizon:>4}{r.n:>8}  too few scoreable rows yet")
                continue
            D.log(f"  {r.horizon:>4}{r.n:>8}{r.rmse:>9.4f}{r.f1_macro:>9.4f}"
                  f"{r.onset_recall:>8.3f}{r.false_alarm_rate:>9.4f}  "
                  + ("PERFORMANCE DRIFT" if r.flag_performance_drift else "ok"))

    drift = data_drift(now)
    drift.to_csv(drift_path, index=False)
    if len(drift):
        D.log(f"\n  data drift, last {MON['window_days']} days "
              f"(limit {float(MON['max_out_of_range_share']):.1%} of rows "
              f"outside the training range)")
        D.log(f"  {'feature':<24}{'rows':>8}{'out of range':>14}  flag")
        for r in drift.itertuples():
            D.log(f"  {r.feature:<24}{r.n_rows:>8}"
                  f"{r.share_out_of_range:>13.2%}  "
                  + (f"DATA DRIFT (worst {r.worst_station})"
                     if r.flag_data_drift else "ok"))
    D.log(f"\nwrote {D.shortpath(out_path)}")
    D.log(f"wrote {D.shortpath(drift_path)}")


if __name__ == "__main__":
    main()
