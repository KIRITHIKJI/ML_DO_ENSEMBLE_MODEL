"""
STEP 6 - ATAE: the Adaptive Threshold-Aware Ensemble.

Run from the project root:

    python -m src.models.ensemble            # grid search + final weights
    python -m src.models.ensemble --quick    # a short grid, for a smoke test

Five models go in - Random Forest, XGBoost, LightGBM, BiLSTM, CNN-LSTM -
and one forecast comes out. Nothing is retrained here: the ensemble only
reads the validation predictions those models already wrote.

The idea in one sentence: weight each model, at each station and each hour,
by how well it has been doing RECENTLY and NEAR THE THRESHOLD that matters.

Step by step, for one station s and one hour t:

  1. Threshold-aware error. A past prediction with observed value y and
     prediction p costs

         e = w(y) * (y - p)^2          (regression)
         e = w(y) * Brier(probs, y)    (classification)

     where the weight

         w(y) = 1 + alpha * exp( -(y - 5)^2 / (2 * sigma^2) )

     is largest right at the 5 mg/L Low threshold and falls to 1 far from
     it. Being wrong at 4.9 mg/L turns a warning into a miss; being wrong
     at 12 mg/L costs nothing. alpha = 0 turns this off entirely.

  2. Only KNOWN errors. A prediction made at t' for t' + h can only be
     scored once t' + h has happened. At time t the ensemble may therefore
     use a prediction made at t' only if t' + h <= t. The recent error of
     model m is the mean of e over the errors that became known in the last
     W hours at station s. This is the whole reason the ensemble cannot
     cheat: it never sees an error it would not have known about.

  3. Weights, by softmax of the negative normalised error:

         normErr_m = err_m / mean_k(err_k)
         weight_m  = exp(-normErr_m / T) / sum_k exp(-normErr_k / T)

     T is a temperature: small T backs the current best model hard, large T
     spreads the weight out. Normalising first makes T mean the same thing
     at every horizon.

  4. Fallbacks. With fewer than `min_known` known errors at a station, the
     all-station error over the same window is used instead; with none at
     all, every model gets an equal weight.

  5. The forecast. Regression: the weighted mean of the five predictions.
     Classification: the weighted mean of the five probability vectors,
     then the largest class.

Everything is fitted on val-A (the first half of 2024) and reported on
val-B (the second half). The 2025 test split is never read.
"""

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.evaluation.calibration import val_a_mask, val_b_mask
from src.evaluation.metrics import (brier_score, classification_metrics,
                                    early_warning_metrics, threshold_weight,
                                    weighted_rmse)

CFG = load_config()
M = CFG["models"]
HORIZONS = CFG["horizons"]
CLASS_NAMES = CFG["target"]["class_names"]
N_CLASSES = len(CLASS_NAMES)
LOW = CFG["target"]["thresholds"]["low"]
ENS = CFG["ensemble"]
MEMBERS = ENS["members"]
PRED_DIR = project_path(M["dirs"]["val_predictions"])


# ---------------------------------------------------------------------------
# Loading what the members already predicted
# ---------------------------------------------------------------------------
def load_member_predictions(h, members=None):
    """
    Everything the ensemble needs for one horizon, aligned on (station, t):

        keys      station, time, and the class at time t
        y_reg     the true DO at t + h
        y_cls     the true class at t + h
        reg       (n_rows, n_models) predicted DO
        proba     (n_rows, n_models, 3) predicted class probabilities
    """
    members = members or MEMBERS
    keys, reg, proba = None, [], []
    for name in members:
        r = pd.read_parquet(os.path.join(PRED_DIR, f"{name}_regression_h{h}.parquet"))
        c = pd.read_parquet(os.path.join(PRED_DIR,
                                         f"{name}_classification_h{h}.parquet"))
        if keys is None:
            keys = r[["station", "time"]].copy()
            y_reg = r["y_true"].to_numpy(dtype="float64")
            y_cls = c["y_true"].astype("float64").to_numpy()
        else:
            assert (r["time"].to_numpy() == keys["time"].to_numpy()).all(), \
                f"{name} is not aligned with the other members"
        reg.append(r["y_pred"].to_numpy(dtype="float64"))
        proba.append(c[[f"proba_{n}" for n in CLASS_NAMES]].to_numpy("float64"))

    # the class at time t, which the early-warning metrics need
    now = pd.read_parquet(project_path(CFG["paths"]["val"]),
                          columns=["station", "time", CFG["target"]["column"]])
    now = keys.merge(now, on=["station", "time"], how="left")
    from src.evaluation.metrics import to_class
    keys = keys.assign(do_now=now[CFG["target"]["column"]].to_numpy())
    keys = keys.assign(class_now=to_class(keys["do_now"]))
    return (keys, y_reg, y_cls,
            np.column_stack(reg), np.stack(proba, axis=1))


# ---------------------------------------------------------------------------
# 1. Threshold-aware errors
# ---------------------------------------------------------------------------
def member_errors(task, y_reg, y_cls, reg, proba, alpha, sigma):
    """
    e for every row and every model, with the threshold weight applied.
    The weight always uses the observed DO VALUE at the target time, for
    both tasks: "how close to the threshold was the water really".
    """
    w = threshold_weight(y_reg, alpha, sigma)[:, None]
    if task == "regression":
        return w * (y_reg[:, None] - reg) ** 2
    brier = np.column_stack([brier_score(y_cls, proba[:, m, :])
                             for m in range(proba.shape[1])])
    return w * brier


# ---------------------------------------------------------------------------
# 2. Recent KNOWN error, per station and hour
# ---------------------------------------------------------------------------
def recent_errors(keys, errors, h, window_hours, min_known):
    """
    For every row (station s, time t) and model m: the mean error over the
    predictions that had already come true by t, within the last
    `window_hours`.

    A prediction made at t' is scored at its target time t' + h, so the
    error becomes KNOWN at t' + h. We therefore index the errors by their
    availability time, roll over that, and read the result off at t. Using
    `closed="left"` means the value at t uses errors known strictly before
    t, never one that lands exactly at t.
    """
    n_models = errors.shape[1]
    out = np.full(errors.shape, np.nan)
    counts = np.zeros(errors.shape)

    df = pd.DataFrame(errors, columns=[f"m{i}" for i in range(n_models)])
    df["station"] = keys["station"].to_numpy()
    df["avail"] = pd.to_datetime(keys["time"].to_numpy()) + pd.Timedelta(hours=h)
    df["t"] = pd.to_datetime(keys["time"].to_numpy())
    cols = [f"m{i}" for i in range(n_models)]
    win = f"{int(window_hours)}h"

    # --- per station ---
    for station, g in df.groupby("station", sort=False):
        rows = g.index.to_numpy()
        # one hourly spine covering both the availability times and the
        # times we need to read off
        lo = min(g["avail"].min(), g["t"].min())
        hi = max(g["avail"].max(), g["t"].max())
        spine = pd.date_range(lo, hi, freq="h")
        e = (g.set_index("avail")[cols].reindex(spine))
        rolled = e.rolling(win, closed="left", min_periods=1)
        mean_e = rolled.mean()
        count_e = rolled.count()
        out[rows] = mean_e.reindex(g["t"]).to_numpy()
        counts[rows] = count_e.reindex(g["t"]).to_numpy()

    # --- the all-station fallback, same window ---
    allst = df.set_index("avail")[cols].sort_index()
    allst = allst.groupby(level=0).mean()
    spine = pd.date_range(allst.index.min(), max(allst.index.max(),
                                                 df["t"].max()), freq="h")
    allst = allst.reindex(spine)
    global_mean = allst.rolling(win, closed="left", min_periods=1).mean()
    fallback = global_mean.reindex(df["t"]).to_numpy()

    thin = (counts < min_known) | np.isnan(out)
    out = np.where(thin, fallback, out)
    return out, counts


# ---------------------------------------------------------------------------
# 3. Weights
# ---------------------------------------------------------------------------
def weights_from_errors(recent, temperature):
    """
    Normalise by the mean error across models, then softmax the negative.
    Rows with no usable history get equal weights.
    """
    n_models = recent.shape[1]
    err = np.array(recent, dtype="float64")
    known = ~np.isnan(err).all(axis=1)
    # a model with no history on a row takes that row's average error, so it
    # is neither rewarded nor punished for having been silent
    row_mean = np.full((len(err), 1), np.nan)
    if known.any():
        row_mean[known, 0] = np.nanmean(err[known], axis=1)
    err = np.where(np.isnan(err), row_mean, err)
    err = np.where(np.isnan(err), 0.0, err)

    mean = err.mean(axis=1, keepdims=True)
    mean[mean <= 0] = 1.0
    norm = err / mean
    z = -norm / float(temperature)
    z -= z.max(axis=1, keepdims=True)              # stable softmax
    w = np.exp(z)
    w /= w.sum(axis=1, keepdims=True)
    w[~known] = 1.0 / n_models                     # nothing known yet
    return w


# ---------------------------------------------------------------------------
# 4. The forecast
# ---------------------------------------------------------------------------
def combine(task, weights, reg, proba):
    """Weighted mean of the members' predictions."""
    if task == "regression":
        return (weights * reg).sum(axis=1)
    return (weights[:, :, None] * proba).sum(axis=1)


def atae(task, keys, y_reg, y_cls, reg, proba, alpha, sigma, window,
         temperature, h, min_known=None):
    """The whole algorithm for one task and horizon. Returns (pred, weights)."""
    min_known = ENS["min_known_errors"] if min_known is None else min_known
    e = member_errors(task, y_reg, y_cls, reg, proba, alpha, sigma)
    recent, _ = recent_errors(keys, e, h, window, min_known)
    w = weights_from_errors(recent, temperature)
    return combine(task, w, reg, proba), w


# ---------------------------------------------------------------------------
# Other ways of combining, for the comparison table
# ---------------------------------------------------------------------------
def simple_average(task, reg, proba):
    n = reg.shape[1]
    w = np.full((len(reg), n), 1.0 / n)
    return combine(task, w, reg, proba), w


def static_weights(task, errors_a, mask_a, temperature, n_rows):
    """
    One weight per model for the whole period, from the mean error on
    val-A. Used for "static inverse-error" and for the threshold-aware-only
    ablation - everything the full ATAE has except the adaptation in time.
    """
    mean_err = np.nanmean(errors_a[mask_a], axis=0)
    norm = mean_err / mean_err.mean()
    z = -norm / float(temperature)
    z -= z.max()
    w = np.exp(z)
    w /= w.sum()
    return np.tile(w, (n_rows, 1))


def inverse_error_weights(errors_a, mask_a, n_rows):
    """Plain 1/error weights from val-A, with no temperature and no threshold."""
    mean_err = np.nanmean(errors_a[mask_a], axis=0)
    w = 1.0 / np.clip(mean_err, 1e-12, None)
    w /= w.sum()
    return np.tile(w, (n_rows, 1))


def stacking(task, reg, proba, y_reg, y_cls, mask_a):
    """
    A model that learns how to blend: Ridge on the five predicted DO
    values, or multinomial logistic regression on the fifteen predicted
    probabilities. Fitted on val-A, applied everywhere.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    if task == "regression":
        X, y = reg, y_reg
        ok = mask_a & ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        model = Ridge(alpha=1.0).fit(X[ok], y[ok])
        return model.predict(X), model
    X = proba.reshape(len(proba), -1)
    y = y_cls
    ok = mask_a & ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    model = LogisticRegression(max_iter=1000, multi_class="multinomial",
                               class_weight="balanced",
                               random_state=CFG["seed"]).fit(X[ok], y[ok].astype(int))
    out = model.predict_proba(X)
    if out.shape[1] < N_CLASSES:                   # a class missing on val-A
        full = np.zeros((len(out), N_CLASSES))
        full[:, model.classes_.astype(int)] = out
        out = full
    return out, model


# ---------------------------------------------------------------------------
# Scoring used by the grid search
# ---------------------------------------------------------------------------
def tuning_score(task, y_reg, y_cls, class_now, pred, mask, alpha_score,
                 sigma_score):
    """
    What the grid search maximises (higher is better for both tasks).

    Regression: the negative threshold-weighted RMSE, so that being wrong
    near 5 mg/L counts for more. The weight used for SCORING is fixed by
    config, not by the alpha being searched - otherwise each candidate
    would be marked by its own exam.

    Classification: the mean of macro F1 and onset recall, which is the
    pair this project actually cares about.
    """
    if task == "regression":
        return -weighted_rmse(y_reg[mask], pred[mask], alpha_score, sigma_score)
    cls = pred[mask].argmax(axis=1)
    m = classification_metrics(y_cls[mask], cls)
    ew = early_warning_metrics(class_now[mask], y_cls[mask], cls)
    f1, onset = m["f1_macro"], ew["onset_recall"]
    if np.isnan(onset):
        return f1
    return 0.5 * (f1 + onset)


# ---------------------------------------------------------------------------
# Grid search on val-A
# ---------------------------------------------------------------------------
def grid_search(task, data, quick=False):
    """
    Try every setting on val-A and return the table, best first. One
    setting is chosen for all four horizons: a different alpha per horizon
    would be four more choices made on the validation data.
    """
    grid = ENS["grid"]
    alphas = [0.0, 2.0] if quick else grid["alpha"]
    sigmas = [1.0] if quick else grid["sigma"]
    windows = [168] if quick else grid["window_hours"]
    temps = [0.5] if quick else grid["temperature"]
    a_s, s_s = ENS["score_alpha"], ENS["score_sigma"]

    # alpha = 0 makes sigma irrelevant, so do not run it twice
    combos = []
    for a, s in itertools.product(alphas, sigmas):
        if a == 0 and any(c[0] == 0 for c in combos):
            continue
        combos.append((a, s))

    rows = []
    for alpha, sigma in combos:
        # the errors depend only on alpha and sigma
        errs = {h: member_errors(task, d["y_reg"], d["y_cls"], d["reg"],
                                 d["proba"], alpha, sigma)
                for h, d in data.items()}
        for window in windows:
            recent = {h: recent_errors(data[h]["keys"], errs[h], h, window,
                                       ENS["min_known_errors"])[0]
                      for h in data}
            for temperature in temps:
                scores = []
                for h, d in data.items():
                    w = weights_from_errors(recent[h], temperature)
                    pred = combine(task, w, d["reg"], d["proba"])
                    scores.append(tuning_score(task, d["y_reg"], d["y_cls"],
                                               d["class_now"], pred,
                                               d["mask_a"], a_s, s_s))
                rows.append({"alpha": alpha, "sigma": sigma,
                             "window_hours": window, "temperature": temperature,
                             "val_a_score": float(np.mean(scores)),
                             **{f"score_h{h}": s
                                for h, s in zip(data, scores)}})
                print(f"    alpha={alpha:<4} sigma={sigma:<4} W={window:<4} "
                      f"T={temperature:<5} -> {rows[-1]['val_a_score']:.5f}")
    return pd.DataFrame(rows).sort_values("val_a_score", ascending=False)


# ---------------------------------------------------------------------------
# main: search, then save the chosen settings
# ---------------------------------------------------------------------------
def load_all(members=None):
    """Member predictions and masks for every horizon."""
    data = {}
    for h in HORIZONS:
        keys, y_reg, y_cls, reg, proba = load_member_predictions(h, members)
        data[h] = {"keys": keys, "y_reg": y_reg, "y_cls": y_cls,
                   "reg": reg, "proba": proba,
                   "class_now": keys["class_now"].to_numpy(),
                   "mask_a": val_a_mask(keys["time"]),
                   "mask_b": val_b_mask(keys["time"])}
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--quick", action="store_true", help="a 2-point grid")
    args = ap.parse_args()

    print(f"ATAE members: {MEMBERS}")
    t0 = time.perf_counter()
    data = load_all()
    print(f"loaded {len(data[HORIZONS[0]]['keys']):,} rows x {len(MEMBERS)} "
          f"models x {len(HORIZONS)} horizons in "
          f"{time.perf_counter() - t0:.0f}s")
    print(f"val-A {data[24]['mask_a'].sum():,} rows (tuning) | "
          f"val-B {data[24]['mask_b'].sum():,} rows (reporting)\n")

    chosen, tables = {}, {}
    for task in ["regression", "classification"]:
        print(f"  --- grid search, {task} ---")
        t1 = time.perf_counter()
        table = grid_search(task, data, quick=args.quick)
        tables[task] = table
        best = table.iloc[0]
        chosen[task] = {"alpha": float(best["alpha"]),
                        "sigma": float(best["sigma"]),
                        "window_hours": int(best["window_hours"]),
                        "temperature": float(best["temperature"]),
                        "val_a_score": float(best["val_a_score"])}
        print(f"  chosen for {task}: {chosen[task]} "
              f"({time.perf_counter() - t1:.0f}s)\n")
        out = project_path(ENS["grid_results"].format(task=task))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        table.to_csv(out, index=False)

    path = project_path(ENS["chosen_file"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chosen, f, indent=2)
    print(f"Wrote {path}")
    print("Next: python -m src.evaluation.step6_report")


if __name__ == "__main__":
    main()
