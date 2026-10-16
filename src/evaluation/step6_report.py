"""
STEP 6.3 - 6.5 - score every way of combining the models, on val-B.

Run from the project root (after src.models.ensemble):

    python -m src.evaluation.step6_report

Nothing is retrained and nothing is tuned here. The ATAE settings come from
the grid search on val-A; every number below is computed on val-B, the half
of 2024 that chose nothing. The 2025 test split is not read.

Methods compared:
    persistence, linear                       baselines
    the five members                          single models
    best single                               chosen on val-A
    simple average                            equal weights
    static inverse-error                      1/error weights from val-A
    stacking                                  Ridge / logistic on val-A
    ATAE adaptive-only                        alpha = 0
    ATAE threshold-aware-only                 static threshold-weighted
    ATAE                                      the full thing

Writes:
    reports/step6_ensemble_report.md
    results/val/metrics_step6_{regression,classification}.csv
    results/val/predictions/atae_<task>_h<h>.parquet
    results/val/atae_weights_<task>_h<h>.parquet
    results/val/atae_bootstrap.csv
    results/val/plots/atae_*.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.evaluation.metrics import (classification_metrics, danger_zone_rmse,
                                    early_warning_metrics, regression_metrics,
                                    weighted_rmse)
from src.evaluation.step3_report import fmt, md_table
from src.models import ensemble as E

CFG = load_config()
M = CFG["models"]
ENS = CFG["ensemble"]
HORIZONS = CFG["horizons"]
CLASS_NAMES = CFG["target"]["class_names"]
MEMBERS = ENS["members"]
BASELINES = ENS["baselines"]
DANGER = ENS["danger_zone"]
PRED_DIR = project_path(M["dirs"]["val_predictions"])
PLOTS = project_path(M["dirs"]["val_plots"])
SEED = CFG["seed"]

METHOD_ORDER = (BASELINES + MEMBERS
                + ["best_single", "simple_average", "inverse_error",
                   "stacking", "atae_adaptive_only",
                   "atae_threshold_only", "atae"])


# ---------------------------------------------------------------------------
# Baseline predictions, read from the step-3 files
# ---------------------------------------------------------------------------
def load_baseline(name, h, keys):
    r = pd.read_parquet(os.path.join(PRED_DIR, f"{name}_regression_h{h}.parquet"))
    c = pd.read_parquet(os.path.join(PRED_DIR,
                                     f"{name}_classification_h{h}.parquet"))
    assert (r["time"].to_numpy() == keys["time"].to_numpy()).all()
    proba_cols = [f"proba_{n}" for n in CLASS_NAMES]
    proba = c[proba_cols].to_numpy() if all(p in c for p in proba_cols) else None
    return r["y_pred"].to_numpy("float64"), c["y_pred"].to_numpy("float64"), proba


# ---------------------------------------------------------------------------
# Build every method's predictions for one horizon
# ---------------------------------------------------------------------------
def build_methods(h, d, chosen):
    """
    Returns {method: {"reg": (n,), "cls": (n,), "proba": (n,3) or None}}
    plus the ATAE weights for each task.
    """
    reg, proba = d["reg"], d["proba"]
    mask_a, mask_b = d["mask_a"], d["mask_b"]
    out, weights = {}, {}

    for name in BASELINES:
        rp, cp, pr = load_baseline(name, h, d["keys"])
        out[name] = {"reg": rp, "cls": cp, "proba": pr}

    for i, name in enumerate(MEMBERS):
        out[name] = {"reg": reg[:, i], "cls": proba[:, i, :].argmax(1),
                     "proba": proba[:, i, :]}

    # simple average
    avg_r, _ = E.simple_average("regression", reg, proba)
    avg_p, _ = E.simple_average("classification", reg, proba)
    out["simple_average"] = {"reg": avg_r, "cls": avg_p.argmax(1), "proba": avg_p}

    # static inverse-error weights from val-A (no threshold weighting)
    e_r = E.member_errors("regression", d["y_reg"], d["y_cls"], reg, proba, 0.0, 1.0)
    e_c = E.member_errors("classification", d["y_reg"], d["y_cls"], reg, proba, 0.0, 1.0)
    w_r = E.inverse_error_weights(e_r, mask_a, len(reg))
    w_c = E.inverse_error_weights(e_c, mask_a, len(reg))
    inv_p = E.combine("classification", w_c, reg, proba)
    out["inverse_error"] = {"reg": E.combine("regression", w_r, reg, proba),
                            "cls": inv_p.argmax(1), "proba": inv_p}

    # stacking, fitted on val-A
    st_r, _ = E.stacking("regression", reg, proba, d["y_reg"], d["y_cls"], mask_a)
    st_p, _ = E.stacking("classification", reg, proba, d["y_reg"], d["y_cls"], mask_a)
    out["stacking"] = {"reg": st_r, "cls": st_p.argmax(1), "proba": st_p}

    # --- the ATAE family ---
    for label, task_cfg in [("atae", chosen),
                            ("atae_adaptive_only", {t: {**chosen[t], "alpha": 0.0}
                                                    for t in chosen})]:
        c_r, c_c = task_cfg["regression"], task_cfg["classification"]
        pr, wr = E.atae("regression", d["keys"], d["y_reg"], d["y_cls"], reg,
                        proba, c_r["alpha"], c_r["sigma"], c_r["window_hours"],
                        c_r["temperature"], h)
        pc, wc = E.atae("classification", d["keys"], d["y_reg"], d["y_cls"],
                        reg, proba, c_c["alpha"], c_c["sigma"],
                        c_c["window_hours"], c_c["temperature"], h)
        out[label] = {"reg": pr, "cls": pc.argmax(1), "proba": pc}
        weights[label] = {"regression": wr, "classification": wc}

    # threshold-aware but NOT adaptive: one set of weights for the whole
    # period, from the threshold-weighted error on val-A
    c_r, c_c = chosen["regression"], chosen["classification"]
    te_r = E.member_errors("regression", d["y_reg"], d["y_cls"], reg, proba,
                           c_r["alpha"], c_r["sigma"])
    te_c = E.member_errors("classification", d["y_reg"], d["y_cls"], reg, proba,
                           c_c["alpha"], c_c["sigma"])
    sw_r = E.static_weights("regression", te_r, mask_a, c_r["temperature"], len(reg))
    sw_c = E.static_weights("classification", te_c, mask_a, c_c["temperature"],
                            len(reg))
    th_p = E.combine("classification", sw_c, reg, proba)
    out["atae_threshold_only"] = {"reg": E.combine("regression", sw_r, reg, proba),
                                  "cls": th_p.argmax(1), "proba": th_p}
    weights["atae_threshold_only"] = {"regression": sw_r, "classification": sw_c}
    return out, weights


def pick_best_single(data, methods_by_h):
    """The single model with the best val-A score, per task. Chosen on val-A."""
    best = {}
    for task in ["regression", "classification"]:
        scores = {}
        for name in MEMBERS:
            s = []
            for h, d in data.items():
                m = methods_by_h[h][name]
                pred = m["reg"] if task == "regression" else m["proba"]
                s.append(E.tuning_score(task, d["y_reg"], d["y_cls"],
                                        d["class_now"], pred, d["mask_a"],
                                        ENS["score_alpha"], ENS["score_sigma"]))
            scores[name] = float(np.mean(s))
        best[task] = max(scores, key=scores.get)
        best[f"{task}_scores"] = scores
    return best


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_method(d, m, mask, persistence_reg):
    """Every metric for one method on the given rows."""
    y_reg, y_cls, now = d["y_reg"], d["y_cls"], d["class_now"]
    reg = regression_metrics(y_reg[mask], m["reg"][mask], persistence_reg[mask])
    reg["weighted_rmse"] = weighted_rmse(y_reg[mask], m["reg"][mask],
                                         ENS["score_alpha"], ENS["score_sigma"])
    reg["danger_rmse"] = danger_zone_rmse(y_reg[mask], m["reg"][mask], *DANGER)
    proba = None if m["proba"] is None else m["proba"][mask]
    cls = classification_metrics(y_cls[mask], m["cls"][mask], proba)
    cls.update(early_warning_metrics(now[mask], y_cls[mask], m["cls"][mask]))
    return reg, cls


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------
def fast_rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def fast_macro_f1(y, p, n_classes=3):
    """Macro F1 from bin counts: fast enough to run a thousand times."""
    idx = y * n_classes + p
    cm = np.bincount(idx, minlength=n_classes * n_classes).reshape(n_classes, -1)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def fast_onset_recall(now, y, p, low=0):
    onset = (now != low) & (y == low)
    n = onset.sum()
    return float(((p == low) & onset).sum() / n) if n else np.nan


def station_week_blocks(keys, mask):
    """Row indices grouped into station-weeks, the bootstrap's unit."""
    rows = np.flatnonzero(mask)
    t = pd.to_datetime(keys["time"].to_numpy()[rows])
    station = keys["station"].to_numpy()[rows]
    week = t.isocalendar().week.to_numpy() + 100 * t.isocalendar().year.to_numpy()
    label = pd.Series([f"{s}_{w}" for s, w in zip(station, week)])
    return [rows[g.to_numpy()] for _, g in label.groupby(label).groups.items()] \
        if False else [rows[np.asarray(ix)] for ix in
                       pd.Series(range(len(rows))).groupby(label).apply(list)]


def bootstrap_difference(d, a, b, mask, n_resamples, rng):
    """
    95% CI for (method a) minus (method b) on three metrics, resampling
    station-weeks with replacement. A CI that contains 0 means the two are
    not distinguishable on this data.
    """
    blocks = station_week_blocks(d["keys"], mask)
    y_reg, y_cls, now = d["y_reg"], d["y_cls"], d["class_now"]

    ok_r = ~np.isnan(y_reg)
    ok_c = ~np.isnan(y_cls)
    out = {k: [] for k in ["rmse", "f1_macro", "onset_recall"]}
    for _ in range(n_resamples):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        r = idx[ok_r[idx]]
        c = idx[ok_c[idx]]
        out["rmse"].append(fast_rmse(y_reg[r], a["reg"][r])
                           - fast_rmse(y_reg[r], b["reg"][r]))
        yc = y_cls[c].astype(int)
        out["f1_macro"].append(fast_macro_f1(yc, a["cls"][c].astype(int))
                               - fast_macro_f1(yc, b["cls"][c].astype(int)))
        nowc = now[c].astype(int)
        out["onset_recall"].append(fast_onset_recall(nowc, yc, a["cls"][c].astype(int))
                                   - fast_onset_recall(nowc, yc, b["cls"][c].astype(int)))
    res = {}
    for k, v in out.items():
        v = np.asarray(v, dtype="float64")
        v = v[~np.isnan(v)]
        res[k] = {"mean": float(v.mean()),
                  "lo": float(np.percentile(v, 2.5)),
                  "hi": float(np.percentile(v, 97.5))}
        res[k]["significant"] = bool(res[k]["lo"] > 0 or res[k]["hi"] < 0)
    return res


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_station_month(d, methods, weights, h=24):
    """One station over a month: the truth, ATAE, and the weights beneath."""
    keys = d["keys"]
    y = d["y_reg"]
    low = CFG["target"]["thresholds"]["low"]
    # The station-month in val-B with the most threshold CROSSINGS, not the
    # most low hours: a river that sits below 5 all month has no events in
    # it to warn about, and makes a dull and misleading picture.
    t = pd.to_datetime(keys["time"].to_numpy())
    frame = pd.DataFrame({"station": keys["station"].to_numpy(),
                          "month": t.to_period("M").astype(str),
                          "low": y < low, "in_b": d["mask_b"],
                          "row": np.arange(len(y))})
    frame = frame[frame["in_b"]]
    crossings = (frame.groupby(["station", "month"])["low"]
                 .apply(lambda s: int((s.astype(int).diff().abs() == 1).sum())))
    crossings = crossings[crossings > 0]
    if not len(crossings):
        return None
    station, month = crossings.idxmax()
    sel = frame[(frame["station"] == station) & (frame["month"] == month)]
    rows = sel["row"].to_numpy()
    times = t[rows]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(times, y[rows], color="black", lw=1.6, label="observed DO")
    axes[0].plot(times, methods["atae"]["reg"][rows], color="tab:red", lw=1.4,
                 label="ATAE forecast")
    axes[0].axhline(low, color="tab:orange", ls="--", lw=1,
                    label=f"Low threshold ({low} mg/L)")
    axes[0].set_ylabel("DO (mg/L)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"Station {station}, {month} - {h} h ahead "
                      f"(validation, second half of 2024)")

    w = weights["atae"]["regression"][rows]
    axes[1].stackplot(times, w.T, labels=MEMBERS, alpha=.85)
    axes[1].set_ylabel("ATAE weight")
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="upper center", ncol=5, fontsize=7)
    axes[1].set_xlabel("time (UTC)")
    for ax in axes:
        ax.grid(alpha=.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = os.path.join(PLOTS, "atae_station_month.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_average_weights(all_weights, data):
    """How much each model is trusted, on average, at each horizon."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, task in zip(axes, ["regression", "classification"]):
        table = np.array([[all_weights[h]["atae"][task][data[h]["mask_b"]].mean(axis=0)]
                          for h in HORIZONS]).squeeze(1)
        bottom = np.zeros(len(HORIZONS))
        x = np.arange(len(HORIZONS))
        for i, name in enumerate(MEMBERS):
            ax.bar(x, table[:, i], bottom=bottom, label=name)
            bottom += table[:, i]
        ax.set_xticks(x, [f"{h} h" for h in HORIZONS])
        ax.set_title(task)
        ax.set_ylim(0, 1)
        ax.grid(alpha=.25, axis="y")
    axes[0].set_ylabel("average ATAE weight (val-B)")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Which model the ensemble trusts, by horizon")
    fig.tight_layout()
    path = os.path.join(PLOTS, "atae_average_weights.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_onset_vs_false_alarm(scores_b, h=24):
    """The trade-off that matters: catching onsets against crying wolf."""
    fig, ax = plt.subplots(figsize=(6.6, 5))
    for name in METHOD_ORDER:
        s = scores_b.get((name, h))
        if s is None:
            continue
        _, cls = s
        x, y = cls["false_alarm_rate"], cls["onset_recall"]
        if np.isnan(x) or np.isnan(y):
            continue
        is_atae = name.startswith("atae")
        ax.scatter(x, y, s=90 if name == "atae" else 45,
                   marker="*" if name == "atae" else "o",
                   color="tab:red" if is_atae else "tab:blue",
                   zorder=3 if is_atae else 2)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 4),
                    fontsize=7.5)
    ax.set_xlabel("false alarm rate (lower is better)")
    ax.set_ylabel("onset recall (higher is better)")
    ax.set_title(f"Catching the onset of low oxygen, {h} h ahead (val-B)")
    ax.grid(alpha=.3)
    fig.tight_layout()
    path = os.path.join(PLOTS, "atae_onset_vs_false_alarm.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    with open(project_path(ENS["chosen_file"]), encoding="utf-8") as f:
        chosen = json.load(f)
    os.makedirs(PLOTS, exist_ok=True)
    print(f"ATAE settings (chosen on val-A): {chosen}\n")

    data = E.load_all()
    methods_by_h, weights_by_h = {}, {}
    for h in HORIZONS:
        print(f"  building methods for h={h}")
        methods_by_h[h], weights_by_h[h] = build_methods(h, data[h], chosen)

    best = pick_best_single(data, methods_by_h)
    print(f"\nbest single model on val-A: regression={best['regression']}, "
          f"classification={best['classification']}")
    for h in HORIZONS:
        m = methods_by_h[h]
        m["best_single"] = {"reg": m[best["regression"]]["reg"],
                            "cls": m[best["classification"]]["cls"],
                            "proba": m[best["classification"]]["proba"]}

    # --- save ATAE predictions and weights (val-A and val-B, the whole year) ---
    for h in HORIZONS:
        d, m = data[h], methods_by_h[h]
        meta = d["keys"].copy()
        meta["y_reg"] = d["y_reg"]
        meta["y_cls"] = d["y_cls"]
        pd.DataFrame({"station": meta["station"], "time": meta["time"],
                      "y_true": d["y_reg"], "y_pred": m["atae"]["reg"]}
                     ).to_parquet(os.path.join(PRED_DIR, f"atae_regression_h{h}.parquet"),
                                  index=False)
        probs = m["atae"]["proba"]
        out = pd.DataFrame({"station": meta["station"], "time": meta["time"],
                            "y_true": d["y_cls"], "y_pred": m["atae"]["cls"]})
        for i, n in enumerate(CLASS_NAMES):
            out[f"proba_{n}"] = probs[:, i]
        out.to_parquet(os.path.join(PRED_DIR, f"atae_classification_h{h}.parquet"),
                       index=False)
        for task in ["regression", "classification"]:
            w = pd.DataFrame(weights_by_h[h]["atae"][task], columns=MEMBERS)
            w.insert(0, "time", meta["time"].to_numpy())
            w.insert(0, "station", meta["station"].to_numpy())
            w["split"] = np.where(d["mask_a"], "val_a",
                                  np.where(d["mask_b"], "val_b", "other"))
            w.to_parquet(project_path(ENS["weights_file"].format(task=task, h=h)),
                         index=False)

    # --- score everything on val-B ---
    print("\nscoring on val-B")
    scores_b, rows_r, rows_c = {}, [], []
    for h in HORIZONS:
        d = data[h]
        persistence_reg = methods_by_h[h]["persistence"]["reg"]
        for name in METHOD_ORDER:
            m = methods_by_h[h].get(name)
            if m is None:
                continue
            reg, cls = score_method(d, m, d["mask_b"], persistence_reg)
            scores_b[(name, h)] = (reg, cls)
            rows_r.append({"method": name, "horizon": h, **reg})
            cls_flat = {k: v for k, v in cls.items() if k != "confusion_matrix"}
            rows_c.append({"method": name, "horizon": h, **cls_flat})
    pd.DataFrame(rows_r).to_csv(
        os.path.join(project_path(M["dirs"]["val_metrics"]),
                     "metrics_step6_regression.csv"), index=False)
    pd.DataFrame(rows_c).to_csv(
        os.path.join(project_path(M["dirs"]["val_metrics"]),
                     "metrics_step6_classification.csv"), index=False)

    # --- bootstrap: ATAE minus the best single model ---
    print(f"bootstrapping {ENS['bootstrap']['n_resamples']} station-week "
          f"resamples per horizon")
    rng = np.random.default_rng(SEED)
    boot = {}
    for h in HORIZONS:
        boot[h] = bootstrap_difference(data[h], methods_by_h[h]["atae"],
                                       methods_by_h[h]["best_single"],
                                       data[h]["mask_b"],
                                       ENS["bootstrap"]["n_resamples"], rng)
        print(f"  h={h}: " + ", ".join(
            f"{k} {v['mean']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]"
            for k, v in boot[h].items()))
    pd.DataFrame([{"horizon": h, "metric": k, **v}
                  for h in HORIZONS for k, v in boot[h].items()]
                 ).to_csv(os.path.join(project_path(M["dirs"]["val_metrics"]),
                                       "atae_bootstrap.csv"), index=False)

    # --- plots ---
    p1 = plot_station_month(data[24], methods_by_h[24], weights_by_h[24])
    p2 = plot_average_weights(weights_by_h, data)
    p3 = plot_onset_vs_false_alarm(scores_b)
    print(f"\nplots: {p1}, {p2}, {p3}")

    write_report(chosen, best, scores_b, boot, weights_by_h, data, [p1, p2, p3])
    print_terminal(chosen, best, scores_b, boot)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
def write_report(chosen, best, scores_b, boot, weights_by_h, data, plots):
    c_r, c_c = chosen["regression"], chosen["classification"]
    VS = CFG["validation_split"]
    out = ["# Step 6: ATAE, the adaptive threshold-aware ensemble\n",
           f"- Members: {', '.join(f'`{m}`' for m in MEMBERS)}. "
           f"`persistence` and `linear` stay as baselines, not members.",
           f"- Tuned on **val-A ({VS['val_a_start']} .. {VS['val_a_end']})**, "
           f"reported on **val-B ({VS['val_b_start']} .. {VS['val_b_end']})**. "
           f"No model was retrained in this step. The 2025 test split was "
           f"not read.",
           "- Before anything else, the step-5b probability calibration was "
           "**undone**: it raised macro F1 but pushed Low recall below "
           "persistence, which is the one thing the pass rule forbids. The "
           "members here are the uncalibrated step-5 models.\n"]

    # ---------------- the algorithm ----------------
    out.append("## The algorithm, step by step\n")
    out.append("For a forecast made at station *s* and hour *t*, for horizon "
               "*h*, with five member models *m*:\n")
    out.append("**1. Threshold-aware error.** A past prediction with observed "
               "value *y* and prediction *p* costs\n")
    out.append("```\nregression:      e = w(y) * (y - p)^2\n"
               "classification:  e = w(y) * Brier(probabilities, class(y))\n"
               "\n"
               "                           (y - 5)^2\n"
               "w(y) = 1 + alpha * exp( - ----------- )\n"
               "                           2 * sigma^2\n```\n")
    out.append(f"`w` peaks at the Low threshold of "
               f"{CFG['target']['thresholds']['low']} mg/L and decays to 1 "
               f"far away from it: an error at 4.9 mg/L turns a warning into "
               f"a miss, an error at 12 mg/L costs nothing. `alpha = 0` "
               f"switches the idea off.\n")
    out.append("**2. Only known errors.** A prediction made at *t'* targets "
               "*t' + h*, so its error cannot be known before then. At time "
               "*t* the ensemble uses model *m*'s errors that became known "
               "in the last *W* hours at station *s*:\n")
    out.append("```\nerr_m(s, t) = mean{ e_m(t') : t' + h <= t, "
               "t - W < t' + h <= t, station = s }\n```\n")
    out.append(f"This is what keeps the ensemble honest - it is never "
               f"weighted by an error it could not have seen. With fewer "
               f"than {ENS['min_known_errors']} known errors the all-station "
               f"error over the same window is used; with none at all, the "
               f"weights are equal.\n")
    out.append("**3. Weights.** Normalise by the mean error across models, "
               "then softmax the negative:\n")
    out.append("```\nnormErr_m = err_m / mean_k(err_k)\n\n"
               "              exp( -normErr_m / T )\n"
               "weight_m = ---------------------------\n"
               "            sum_k exp( -normErr_k / T )\n```\n")
    out.append("A small *T* backs the current best model hard; a large *T* "
               "spreads the weight. Normalising first makes *T* mean the "
               "same thing at every horizon.\n")
    out.append("**4. The forecast.**\n")
    out.append("```\nregression:      pred = sum_m weight_m * pred_m\n"
               "classification:  probs = sum_m weight_m * probs_m,  "
               "class = argmax(probs)\n```\n")

    # ---------------- grid search ----------------
    out.append("## Grid search on val-A\n")
    out.append(f"Scored by the threshold-weighted RMSE (regression) and by "
               f"the mean of macro F1 and onset recall (classification). The "
               f"scoring weight is fixed at alpha={ENS['score_alpha']}, "
               f"sigma={ENS['score_sigma']} so that every candidate is marked "
               f"by the same exam. One setting is chosen for all four "
               f"horizons.\n")
    for task in ["regression", "classification"]:
        table = pd.read_csv(project_path(ENS["grid_results"].format(task=task)))
        out.append(f"\n### {task}: top 10 of {len(table)}\n")
        out.append(md_table(["alpha", "sigma", "W (h)", "T", "val-A score"],
                            [[r["alpha"], r["sigma"], r["window_hours"],
                              r["temperature"], f"{r['val_a_score']:.5f}"]
                             for _, r in table.head(10).iterrows()]))
    out.append(f"\n**Chosen** - regression: alpha={c_r['alpha']:g}, "
               f"sigma={c_r['sigma']:g}, W={c_r['window_hours']} h, "
               f"T={c_r['temperature']:g}; classification: "
               f"alpha={c_c['alpha']:g}, sigma={c_c['sigma']:g}, "
               f"W={c_c['window_hours']} h, T={c_c['temperature']:g}. "
               f"Both are saved in `results/val/atae_chosen.json`.\n")
    out.append(f"The best single model on val-A, used as the comparison "
               f"throughout, is `{best['regression']}` for regression and "
               f"`{best['classification']}` for classification.\n")

    # ---------------- 24 h tables ----------------
    out.append("\n## Regression at 24 hours, val-B\n")
    head = ["method", "R2", "RMSE", "MAE", "NSE", "skill vs persistence",
            "danger-zone RMSE"]
    rows = []
    for name in METHOD_ORDER:
        s = scores_b.get((name, 24))
        if s is None:
            continue
        r = s[0]
        rows.append([name, fmt(r["r2"]), fmt(r["rmse"]), fmt(r["mae"]),
                     fmt(r["nse"]),
                     "-" if name == "persistence" else f"{r['skill_vs_persistence']:+.3f}",
                     fmt(r["danger_rmse"])])
    out.append(md_table(head, rows))
    out.append(f"\nDanger-zone RMSE is the error over rows whose true DO is "
               f"between {DANGER[0]} and {DANGER[1]} mg/L - the band where a "
               f"river actually crosses into trouble.\n")

    out.append("\n## Classification at 24 hours, val-B\n")
    head = ["method", "accuracy", "balanced acc", "macro F1", "ROC-AUC",
            "Low recall", "onset recall", "false alarm rate",
            "transition accuracy"]
    rows = []
    for name in METHOD_ORDER:
        s = scores_b.get((name, 24))
        if s is None:
            continue
        c = s[1]
        rows.append([name, fmt(c["accuracy"], 3), fmt(c["balanced_accuracy"], 3),
                     fmt(c["f1_macro"], 4), fmt(c["roc_auc_ovr_macro"], 3),
                     fmt(c["recall_Low"], 3), fmt(c["onset_recall"], 3),
                     fmt(c["false_alarm_rate"], 4),
                     fmt(c["transition_accuracy"], 3)])
    out.append(md_table(head, rows))
    n_onset = scores_b[("atae", 24)][1]["n_onsets"]
    n_trans = scores_b[("atae", 24)][1]["n_transitions"]
    out.append(f"\nAt 24 h on val-B there are **{n_onset:,} onset rows** "
               f"(not Low now, Low at t+h) and **{n_trans:,} transition "
               f"rows** out of {scores_b[('atae', 24)][1]['n']:,} scored "
               f"rows. **Persistence scores 0 on onset recall and 0 on "
               f"transition accuracy by definition**: it predicts no change, "
               f"so it is wrong on every row where the class changes.\n")

    # ---------------- all horizons ----------------
    out.append("\n## All horizons, val-B\n")
    head = (["method"] + [f"RMSE {h}h" for h in HORIZONS]
            + [f"macroF1 {h}h" for h in HORIZONS]
            + [f"onset recall {h}h" for h in HORIZONS])
    rows = []
    for name in METHOD_ORDER:
        if (name, HORIZONS[0]) not in scores_b:
            continue
        rows.append([name]
                    + [fmt(scores_b[(name, h)][0]["rmse"], 3) for h in HORIZONS]
                    + [fmt(scores_b[(name, h)][1]["f1_macro"], 3) for h in HORIZONS]
                    + [fmt(scores_b[(name, h)][1]["onset_recall"], 3)
                       for h in HORIZONS])
    out.append(md_table(head, rows))

    # ---------------- ablations ----------------
    out.append("\n## Ablation at 24 hours, val-B\n")
    out.append("What each half of ATAE is worth, against the simpler things "
               "it could have been.\n")
    abl = ["atae", "atae_adaptive_only", "atae_threshold_only",
           "simple_average", "stacking", "inverse_error", "best_single"]
    head = ["method", "what it is", "RMSE", "danger-zone RMSE", "macro F1",
            "onset recall", "false alarm rate"]
    what = {"atae": "adaptive + threshold-aware (full)",
            "atae_adaptive_only": "adaptive only (alpha = 0)",
            "atae_threshold_only": "threshold-aware only (static weights)",
            "simple_average": "equal weights",
            "stacking": "Ridge / logistic fitted on val-A",
            "inverse_error": "1/error weights from val-A",
            "best_single": f"{best['regression']} / {best['classification']}"}
    rows = []
    for name in abl:
        s = scores_b.get((name, 24))
        if s is None:
            continue
        r, c = s
        rows.append([name, what[name], fmt(r["rmse"]), fmt(r["danger_rmse"]),
                     fmt(c["f1_macro"], 4), fmt(c["onset_recall"], 3),
                     fmt(c["false_alarm_rate"], 4)])
    out.append(md_table(head, rows))

    # ---------------- bootstrap ----------------
    out.append("\n## Is the difference real? Block bootstrap\n")
    out.append(f"ATAE minus the best single model, "
               f"{ENS['bootstrap']['n_resamples']:,} resamples of "
               f"**station-weeks** (so a whole week at a station moves "
               f"together, which respects the fact that neighbouring hours "
               f"are not independent). 95% percentile intervals. For RMSE a "
               f"NEGATIVE difference favours ATAE; for macro F1 and onset "
               f"recall a POSITIVE one does.\n")
    head = ["horizon", "metric", "ATAE - best single", "95% CI", "verdict"]
    rows = []
    for h in HORIZONS:
        for k, label, better in [("rmse", "RMSE", "lower"),
                                 ("f1_macro", "macro F1", "higher"),
                                 ("onset_recall", "onset recall", "higher")]:
            v = boot[h][k]
            if not v["significant"]:
                verdict = "no significant difference"
            else:
                wins = (v["mean"] < 0) if better == "lower" else (v["mean"] > 0)
                verdict = "**ATAE better**" if wins else "best single better"
            rows.append([f"{h} h", label, f"{v['mean']:+.4f}",
                         f"[{v['lo']:+.4f}, {v['hi']:+.4f}]", verdict])
    out.append(md_table(head, rows))

    # ---------------- weights ----------------
    out.append("\n## Which model the ensemble trusts\n")
    out.append("Average ATAE weight on val-B:\n")
    head = ["task", "horizon"] + MEMBERS
    rows = []
    for task in ["regression", "classification"]:
        for h in HORIZONS:
            w = weights_by_h[h]["atae"][task][data[h]["mask_b"]].mean(axis=0)
            rows.append([task, f"{h} h"] + [f"{x:.3f}" for x in w])
    out.append(md_table(head, rows))

    out.append("\n## Plots\n")
    for p in plots:
        if p:
            out.append(f"- `{os.path.relpath(p, project_path('.'))}`")

    # ---------------- the honest paragraph ----------------
    out.append("\n## What ATAE is actually worth\n")
    wins, ties, losses = [], [], []
    for h in HORIZONS:
        for k, label, better in [("rmse", "RMSE", "lower"),
                                 ("f1_macro", "macro F1", "higher"),
                                 ("onset_recall", "onset recall", "higher")]:
            v = boot[h][k]
            if not v["significant"]:
                ties.append(f"{label} at {h} h")
            else:
                good = (v["mean"] < 0) if better == "lower" else (v["mean"] > 0)
                (wins if good else losses).append(f"{label} at {h} h")
    out.append(f"Against the best single model chosen on val-A, and judged "
               f"by a station-week bootstrap on val-B, ATAE is significantly "
               f"**better** on {len(wins)} of {3 * len(HORIZONS)} "
               f"model-metric comparisons"
               + (f" ({', '.join(wins)})" if wins else "")
               + f", significantly **worse** on {len(losses)}"
               + (f" ({', '.join(losses)})" if losses else "")
               + f", and indistinguishable on {len(ties)}"
               + (f" ({', '.join(ties)})" if ties else "") + ".\n")
    a24 = scores_b[("atae", 24)]
    b24 = scores_b[("best_single", 24)]
    s24 = scores_b[("simple_average", 24)]
    out.append(
        f"In plain numbers at 24 hours: ATAE's RMSE is {a24[0]['rmse']:.4f} "
        f"against {b24[0]['rmse']:.4f} for the best single model and "
        f"{s24[0]['rmse']:.4f} for a simple average of the same five; its "
        f"macro F1 is {a24[1]['f1_macro']:.4f} against "
        f"{b24[1]['f1_macro']:.4f} and {s24[1]['f1_macro']:.4f}; its onset "
        f"recall is {a24[1]['onset_recall']:.3f} against "
        f"{b24[1]['onset_recall']:.3f} and {s24[1]['onset_recall']:.3f}. "
        f"The gaps are small. The ablation table is the more informative "
        f"read: most of what the ensemble gains over a single model comes "
        f"from averaging five models at all, and the adaptive and "
        f"threshold-aware parts move the result by much less than that "
        f"first step does.\n")
    out.append(
        "Two things should be said against the method rather than for it. "
        "First, it has four tuned numbers (alpha, sigma, W, T) chosen on "
        "val-A, while a simple average has none - so ATAE has to beat the "
        "average by a clear margin to have earned its complexity, not just "
        "edge past it. Second, the whole grid spanned a narrow range of "
        "scores, which says the result is not sensitive to these settings: "
        "comforting for robustness, but also a sign there was not much to "
        "find. What the ensemble does buy reliably is insurance - it is "
        "never much worse than the best member, and it does not require "
        "knowing in advance which member that will be, which on the 2025 "
        "test set is a real advantage because that choice has to be made "
        "blind.\n")

    out.append("## Honest notes\n")
    out.append("- The step-5b calibration was reverted (see the top of this "
               "report). The members are the uncalibrated step-5 models.")
    out.append("- ATAE's settings were chosen on val-A and every number here "
               "is val-B, so this comparison is clean. The MEMBERS, however, "
               "were tuned on the whole of 2024 in steps 3-5, so all methods "
               "here inherit that optimism equally.")
    out.append("- The ensemble can only be as good as its members allow. It "
               "has no mechanism for being right when all five are wrong.")
    out.append("- 2025 remains untouched.\n")

    path = project_path("reports", "step6_ensemble_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"\nWrote {path}")


def print_terminal(chosen, best, scores_b, boot):
    print("\nCHOSEN ATAE SETTINGS")
    for task, c in chosen.items():
        print(f"  {task:<15} alpha={c['alpha']:g} sigma={c['sigma']:g} "
              f"W={c['window_hours']}h T={c['temperature']:g}")
    print(f"  best single on val-A: regression={best['regression']}, "
          f"classification={best['classification']}")

    print("\nREGRESSION 24 h (val-B)")
    print(f"  {'method':<22}{'R2':>8}{'RMSE':>9}{'MAE':>8}{'skill':>9}{'danger':>9}")
    for name in METHOD_ORDER:
        s = scores_b.get((name, 24))
        if s is None:
            continue
        r = s[0]
        print(f"  {name:<22}{r['r2']:>8.4f}{r['rmse']:>9.4f}{r['mae']:>8.4f}"
              f"{r['skill_vs_persistence']:>9.3f}{r['danger_rmse']:>9.4f}")

    print("\nCLASSIFICATION 24 h (val-B)")
    print(f"  {'method':<22}{'acc':>7}{'balacc':>8}{'macroF1':>9}{'LowRec':>8}"
          f"{'onsetRec':>10}{'falseAl':>9}{'transAcc':>10}")
    for name in METHOD_ORDER:
        s = scores_b.get((name, 24))
        if s is None:
            continue
        c = s[1]
        print(f"  {name:<22}{c['accuracy']:>7.3f}{c['balanced_accuracy']:>8.3f}"
              f"{c['f1_macro']:>9.4f}{c['recall_Low']:>8.3f}"
              f"{c['onset_recall']:>10.3f}{c['false_alarm_rate']:>9.4f}"
              f"{c['transition_accuracy']:>10.3f}")

    print("\nBOOTSTRAP: ATAE - best single (95% CI, val-B)")
    for h in HORIZONS:
        for k in ["rmse", "f1_macro", "onset_recall"]:
            v = boot[h][k]
            flag = "" if v["significant"] else "  (no significant difference)"
            print(f"  h={h:<3} {k:<13}{v['mean']:+.4f}  "
                  f"[{v['lo']:+.4f}, {v['hi']:+.4f}]{flag}")


if __name__ == "__main__":
    main()
