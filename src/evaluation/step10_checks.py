"""
STEP 10 - ADDITIONAL ANALYSES ONLY.

Nothing in this file may change the deployed system. The frozen system
(FROZEN.md, every threshold, `final_system: stacking`) is fixed. This module
only re-reads what step 8 already wrote, plus one pure-inference pass with
the frozen stage-1 models, and answers four questions that the limitations
section of the report used to leave open:

  10.1  the consistent stacking pipeline: 2023-only members -> meta-model
        fitted on 2024 -> 2025. That chain has no train/apply mismatch, so
        it says how much the mismatch in the headline chain is worth.
  10.2  station-dropout sensitivity: re-score the SAME 2025 predictions on
        29 stations, and on 30 stations up to 2025-11-18.
  10.3  reproducibility: the seed-to-seed spread of the deep models,
        against the gaps between methods in the 2025 tables.
  10.4  what the deep members add: the correlation of their 24 h errors
        with the trees'.

Nothing here is retrained. 10.1 re-runs the frozen stage-1 models in
inference mode only, because step 8 saved stage-1 METRICS but not stage-1
rows, and a confidence interval for a difference needs the rows.

Run from the project root:

    python -m src.evaluation.step10_checks
"""

import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.evaluation.metrics import (classification_metrics, early_warning_metrics,
                                    regression_metrics, to_class)
from src.evaluation.step6_report import (fast_macro_f1, fast_onset_recall,
                                         fast_rmse, station_week_blocks)

CFG = load_config()
FT = CFG["final_test"]
ENS = CFG["ensemble"]
HORIZONS = CFG["horizons"]
CLASS_NAMES = CFG["target"]["class_names"]
MEMBERS = ENS["members"]
DO = CFG["target"]["column"]
SEED = CFG["seed"]
RESULTS = project_path(FT["results_dir"])
PLOTS = project_path(FT["plots_dir"])
N_BOOT = ENS["bootstrap"]["n_resamples"]

# The station whose DO sensor stops reporting inside the test year, and the
# last hour on which all thirty stations were still reporting.
DROPOUT_STATION = "05474500"
DROPOUT_LAST_FULL_DAY = "2025-11-18"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def fast_false_alarm(now, y, p, low=0):
    """Share of safe hours wrongly warned about - the bootstrap's version."""
    stays_safe = (now != low) & (y != low)
    n = stays_safe.sum()
    return float(((p == low) & stays_safe).sum() / n) if n else np.nan


def pred_path(name, h, label=None):
    tail = "" if label in (None, "stage2") else f"_{label}"
    return os.path.join(RESULTS, f"pred_{name}_h{h}{tail}.parquet")


def score_rows(d, mask=None, rule="warn"):
    """RMSE / R2 / macro F1 / onset recall / false alarm on a set of rows."""
    d = d if mask is None else d[mask]
    cls = d["pred_warn" if rule == "warn" else "pred_argmax"].to_numpy()
    proba_cols = [f"proba_{c}" for c in CLASS_NAMES]
    proba = d[proba_cols].to_numpy() if all(c in d for c in proba_cols) else None
    reg = regression_metrics(d["y_true_reg"], d["pred_reg"])
    cm = classification_metrics(d["y_true_cls"], cls, proba)
    ew = early_warning_metrics(d["class_now"], d["y_true_cls"], cls)
    return {"n": len(d), "r2": reg["r2"], "rmse": reg["rmse"],
            "f1_macro": cm["f1_macro"], "onset_recall": ew["onset_recall"],
            "false_alarm_rate": ew["false_alarm_rate"],
            "n_onsets": ew["n_onsets"]}


# ---------------------------------------------------------------------------
# 10.1  the consistent stacking pipeline
# ---------------------------------------------------------------------------
def rebuild_stage1_rows():
    """
    The stage-1 chain, row by row: the frozen 2023-only members (models/ml,
    models/dl) -> the step-7 stacking meta-models fitted on their 2024
    predictions -> 2025. Inference only; no model file is written or read
    other than the ones FROZEN.md hashes.

    Step 8 scored this chain and saved the metrics, so the rebuild is
    checked against `metrics_*_stage1.csv` before anything is reported.
    """
    from src.evaluation import test_2025 as T8
    from src.models import common as C

    features = C.feature_columns()
    test = pd.read_parquet(project_path(CFG["paths"]["test"]))
    keys = test[["station", "time"]].reset_index(drop=True)
    y_reg = {h: pd.to_numeric(test[f"y_reg_{h}"], errors="coerce").to_numpy()
             for h in HORIZONS}
    y_cls = {h: test[f"y_cls_{h}"].astype("float64").to_numpy()
             for h in HORIZONS}
    class_now = to_class(test[DO])

    preds = T8.build_stage1_predictions(test, features, keys)
    thresholds = T8.load_thresholds()
    written = {}
    for h in HORIZONS:
        m = preds[h]["stacking"]
        proba = m["proba"]
        argmax = proba.argmax(1)
        tau = T8.tau_for(thresholds, "stacking", h)
        from src.models import run_step7 as R
        warn = argmax if tau is None else R.apply_warning(proba, tau)
        out = keys.copy()
        out["y_true_reg"] = y_reg[h]
        out["y_true_cls"] = y_cls[h]
        out["class_now"] = class_now
        out["pred_reg"] = m["reg"]
        out["pred_argmax"] = argmax
        out["pred_warn"] = warn
        for i, n in enumerate(CLASS_NAMES):
            out[f"proba_{n}"] = proba[:, i]
        out.to_parquet(pred_path("stacking", h, "stage1"), index=False)
        written[h] = out
    return written


def stage1_rows(rebuild=False):
    """The stage-1 stacking rows, rebuilding them once if they are missing."""
    if rebuild or not all(os.path.exists(pred_path("stacking", h, "stage1"))
                          for h in HORIZONS):
        return rebuild_stage1_rows()
    return {h: pd.read_parquet(pred_path("stacking", h, "stage1"))
            for h in HORIZONS}


def check_stage1_matches_step8(rows, tol=5e-4):
    """The rebuilt rows must reproduce the step-8 stage-1 metrics."""
    reg = pd.read_csv(os.path.join(RESULTS, "metrics_regression_stage1.csv"))
    cls = pd.read_csv(os.path.join(RESULTS, "metrics_classification_stage1.csv"))
    problems = []
    for h in HORIZONS:
        got = score_rows(rows[h])
        want_r = reg[(reg.method == "stacking") & (reg.horizon == h)].iloc[0]
        want_c = cls[(cls.method == "stacking") & (cls.horizon == h)
                     & (cls.rule == "warn")].iloc[0]
        for key, want in [("rmse", want_r["rmse"]), ("r2", want_r["r2"]),
                          ("f1_macro", want_c["f1_macro"]),
                          ("onset_recall", want_c["onset_recall"]),
                          ("false_alarm_rate", want_c["false_alarm_rate"])]:
            if abs(got[key] - float(want)) > tol:
                problems.append(f"h{h} {key}: rebuilt {got[key]:.6f} vs "
                                f"step 8 {float(want):.6f}")
    return problems


def bootstrap_pipeline_difference(a, b, rng):
    """
    95% CI for (stage 1) minus (stage 2) on the same 2025 rows, resampling
    station-weeks. Both chains predict the identical rows, so the blocks
    are shared and the difference is paired.
    """
    keys = a[["station", "time"]]
    y_reg = a["y_true_reg"].to_numpy(dtype="float64")
    y_cls = a["y_true_cls"].to_numpy(dtype="float64")
    now = a["class_now"].to_numpy(dtype="float64")
    blocks = station_week_blocks(keys, np.ones(len(a), bool))
    ar, br = a["pred_reg"].to_numpy(), b["pred_reg"].to_numpy()
    ac, bc = a["pred_warn"].to_numpy(), b["pred_warn"].to_numpy()
    ok_r = ~np.isnan(y_reg) & ~np.isnan(ar) & ~np.isnan(br)
    ok_c = ~np.isnan(y_cls) & ~np.isnan(now)

    out = {k: [] for k in ["rmse", "f1_macro", "onset_recall",
                           "false_alarm_rate"]}
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        r = idx[ok_r[idx]]
        c = idx[ok_c[idx]]
        out["rmse"].append(fast_rmse(y_reg[r], ar[r]) - fast_rmse(y_reg[r], br[r]))
        yc = y_cls[c].astype(int)
        nc = now[c].astype(int)
        pa, pb = ac[c].astype(int), bc[c].astype(int)
        out["f1_macro"].append(fast_macro_f1(yc, pa) - fast_macro_f1(yc, pb))
        out["onset_recall"].append(fast_onset_recall(nc, yc, pa)
                                   - fast_onset_recall(nc, yc, pb))
        out["false_alarm_rate"].append(fast_false_alarm(nc, yc, pa)
                                       - fast_false_alarm(nc, yc, pb))
    res = {}
    for k, v in out.items():
        v = np.asarray(v, dtype="float64")
        v = v[~np.isnan(v)]
        res[k] = {"diff": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
                  "hi": float(np.percentile(v, 97.5))}
        res[k]["significant"] = bool(res[k]["lo"] > 0 or res[k]["hi"] < 0)
    return res


def run_101(rebuild=False):
    print("\n10.1  consistent stacking pipeline (stage 1) vs the headline "
          "(stage 2)")
    rows1 = stage1_rows(rebuild=rebuild)
    problems = check_stage1_matches_step8(rows1)
    if problems:
        print("  WARNING: the rebuilt stage-1 rows do not match step 8:")
        for p in problems:
            print(f"    {p}")
    else:
        print("  rebuilt stage-1 rows reproduce the step-8 stage-1 metrics")

    rng = np.random.default_rng(SEED)
    out = []
    for h in HORIZONS:
        a = rows1[h]
        b = pd.read_parquet(pred_path("stacking", h))
        assert (a["time"].to_numpy() == b["time"].to_numpy()).all() and \
               (a["station"].to_numpy() == b["station"].to_numpy()).all(), \
               "the two chains do not cover the same rows"
        s1, s2 = score_rows(a), score_rows(b)
        ci = bootstrap_pipeline_difference(a, b, rng)
        row = {"horizon": h, "n": s1["n"], "n_onsets": s1["n_onsets"]}
        for metric in ["rmse", "r2", "f1_macro", "onset_recall",
                       "false_alarm_rate"]:
            row[f"{metric}_stage1"] = s1[metric]
            row[f"{metric}_stage2"] = s2[metric]
            if metric in ci:
                row[f"{metric}_diff"] = ci[metric]["diff"]
                row[f"{metric}_lo"] = ci[metric]["lo"]
                row[f"{metric}_hi"] = ci[metric]["hi"]
                row[f"{metric}_significant"] = ci[metric]["significant"]
        out.append(row)
        print(f"  h{h:<3} RMSE {s1['rmse']:.4f} / {s2['rmse']:.4f}   "
              f"macroF1 {s1['f1_macro']:.4f} / {s2['f1_macro']:.4f}   "
              f"onset {s1['onset_recall']:.3f} / {s2['onset_recall']:.3f}")
    d = pd.DataFrame(out)
    d.to_csv(os.path.join(RESULTS, "step10_pipeline.csv"), index=False)
    return d, problems


# ---------------------------------------------------------------------------
# 10.2  station-dropout sensitivity
# ---------------------------------------------------------------------------
def run_102():
    print("\n10.2  station-dropout sensitivity (re-scoring, no retraining)")
    cutoff = pd.Timestamp(DROPOUT_LAST_FULL_DAY) + pd.Timedelta(days=1)
    out = []
    for h in HORIZONS:
        d = pd.read_parquet(pred_path("stacking", h))
        variants = {
            "headline (30 stations, full year)": None,
            f"without {DROPOUT_STATION} (29 stations, full year)":
                (d["station"] != DROPOUT_STATION).to_numpy(),
            f"30 stations, 2025-01-01..{DROPOUT_LAST_FULL_DAY}":
                (d["time"] < cutoff).to_numpy(),
        }
        for label, mask in variants.items():
            s = score_rows(d, mask)
            out.append({"horizon": h, "subset": label, **s})
            print(f"  h{h:<3} {label:<46} RMSE {s['rmse']:.4f}  "
                  f"macroF1 {s['f1_macro']:.4f}  onset {s['onset_recall']:.3f}"
                  f"  FA {s['false_alarm_rate']:.4f}")
    d = pd.DataFrame(out)
    d.to_csv(os.path.join(RESULTS, "step10_dropout.csv"), index=False)
    return d


# ---------------------------------------------------------------------------
# 10.3  reproducibility: the seed spread of the deep models
# ---------------------------------------------------------------------------
def run_103():
    print("\n10.3  seed-to-seed spread of the deep models")
    dl = project_path(CFG["deep_learning"]["dirs"]["dl_models"])
    rows = []
    for name in ["bilstm", "cnn_lstm"]:
        path = os.path.join(dl, f"{name}_seed_spread.csv")
        if not os.path.exists(path):
            continue
        s = pd.read_csv(path)
        rows.append({
            "model": name,
            "seeds": ";".join(str(int(x)) for x in s["seed"]),
            "rmse_mean": s["rmse_24"].mean(), "rmse_std": s["rmse_24"].std(ddof=1),
            "rmse_range": s["rmse_24"].max() - s["rmse_24"].min(),
            "f1_mean": s["f1_macro_24"].mean(),
            "f1_std": s["f1_macro_24"].std(ddof=1),
            "f1_range": s["f1_macro_24"].max() - s["f1_macro_24"].min(),
        })
    spread = pd.DataFrame(rows)

    # the gaps the 2025 tables are asked to resolve, at 24 h
    reg = pd.read_csv(os.path.join(RESULTS, "metrics_regression_stage2.csv"))
    cls = pd.read_csv(os.path.join(RESULTS, "metrics_classification_stage2.csv"))
    reg24 = reg[reg.horizon == 24].set_index("method")
    cls24 = cls[(cls.horizon == 24) & (cls.rule == "warn")].set_index("method")
    gaps = []
    for label, a, b in [("stacking vs best single (lightgbm)",
                         "stacking", "lightgbm"),
                        ("stacking vs simple average",
                         "stacking", "simple_average"),
                        ("stacking vs the weakest member",
                         "stacking", "cnn_lstm"),
                        ("stacking vs persistence", "stacking", "persistence")]:
        gaps.append({"comparison": label,
                     "rmse_gap": abs(reg24.loc[a, "rmse"] - reg24.loc[b, "rmse"]),
                     "f1_gap": abs(cls24.loc[a, "f1_macro"]
                                   - cls24.loc[b, "f1_macro"])})
    gaps = pd.DataFrame(gaps)

    spread.to_csv(os.path.join(RESULTS, "step10_seed_spread.csv"), index=False)
    gaps.to_csv(os.path.join(RESULTS, "step10_method_gaps.csv"), index=False)
    for _, r in spread.iterrows():
        print(f"  {r['model']:<10} RMSE {r['rmse_mean']:.4f} "
              f"+-{r['rmse_std']:.4f}   macroF1 {r['f1_mean']:.4f} "
              f"+-{r['f1_std']:.4f}")
    for _, r in gaps.iterrows():
        print(f"  gap  {r['comparison']:<36} RMSE {r['rmse_gap']:.4f}  "
              f"macroF1 {r['f1_gap']:.4f}")
    return spread, gaps


# ---------------------------------------------------------------------------
# 10.4  what the deep members add: error correlation at 24 h
# ---------------------------------------------------------------------------
def run_104(h=24):
    print(f"\n10.4  correlation of {h} h errors between members "
          f"(analysis only)")
    errors = {}
    for name in MEMBERS:
        d = pd.read_parquet(pred_path(name, h))
        e = d["pred_reg"].to_numpy(dtype="float64") - \
            d["y_true_reg"].to_numpy(dtype="float64")
        errors[name] = e
    E = pd.DataFrame(errors)
    corr = E.corr()
    corr.to_csv(os.path.join(RESULTS, f"step10_error_correlation_h{h}.csv"))
    print(corr.round(3).to_string())

    os.makedirs(PLOTS, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(corr.to_numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(MEMBERS)))
    ax.set_xticklabels(MEMBERS, rotation=40, ha="right")
    ax.set_yticks(range(len(MEMBERS)))
    ax.set_yticklabels(MEMBERS)
    for i in range(len(MEMBERS)):
        for j in range(len(MEMBERS)):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.72 else "black", fontsize=9)
    ax.set_title(f"Correlation of {h} h forecast errors, 2025\n"
                 "(analysis only - not used to change the system)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    fig.tight_layout()
    path = os.path.join(PLOTS, f"error_correlation_h{h}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")
    return corr


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(rebuild=False):
    os.makedirs(RESULTS, exist_ok=True)
    print("STEP 10 - additional analyses. The frozen system is unchanged; "
          "nothing here feeds back into it.")
    pipeline, problems = run_101(rebuild=rebuild)
    dropout = run_102()
    spread, gaps = run_103()
    corr = run_104()
    summary = {
        "frozen_system": CFG["final_system"],
        "stage1_rebuild_problems": problems,
        "files": ["step10_pipeline.csv", "step10_dropout.csv",
                  "step10_seed_spread.csv", "step10_method_gaps.csv",
                  "step10_error_correlation_h24.csv",
                  "plots/error_correlation_h24.png"],
    }
    with open(os.path.join(RESULTS, "step10_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {RESULTS} (step10_*)")
    print("Next: python -m src.evaluation.final_report")
    return pipeline, dropout, spread, gaps, corr


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="re-run the stage-1 inference pass even if the rows "
                         "are already on disk")
    main(**vars(ap.parse_args()))
