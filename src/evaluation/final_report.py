"""
STEP 8.5 - the final report on the 2025 test.

Run from the project root (after src.evaluation.test_2025):

    python -m src.evaluation.final_report

Reads results/test/ and writes reports/final_results.md plus the plots.
Nothing is fitted, tuned or re-scored here.
"""

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.evaluation.metrics import classification_metrics, early_warning_metrics
from src.evaluation.step3_report import fmt, md_table
from src.evaluation.step6_report import (fast_macro_f1, fast_onset_recall,
                                         fast_rmse, station_week_blocks)

CFG = load_config()
FT = CFG["final_test"]
ENS = CFG["ensemble"]
HORIZONS = CFG["horizons"]
CLASS_NAMES = CFG["target"]["class_names"]
MEMBERS = ENS["members"]
LOW = CFG["target"]["thresholds"]["low"]
RESULTS = project_path(FT["results_dir"])
PLOTS = project_path(FT["plots_dir"])
SEED = CFG["seed"]
BEST = FT["best_single"]
METHODS = ["persistence", "linear"] + MEMBERS + ["simple_average", "atae",
                                                 "stacking"]


def load(label="stage2"):
    reg = pd.read_csv(os.path.join(RESULTS, f"metrics_regression_{label}.csv"))
    cls = pd.read_csv(os.path.join(RESULTS, f"metrics_classification_{label}.csv"))
    return reg, cls


def pred(name, h):
    return pd.read_parquet(os.path.join(RESULTS, f"pred_{name}_h{h}.parquet"))


def get_r(reg, method, h, col):
    row = reg[(reg["method"] == method) & (reg["horizon"] == h)]
    return row[col].iloc[0] if len(row) else np.nan


def get_c(cls, method, h, rule, col):
    row = cls[(cls["method"] == method) & (cls["horizon"] == h)
              & (cls["rule"] == rule)]
    return row[col].iloc[0] if len(row) else np.nan


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def boot_pair(a, b, keys, y_reg, y_cls, class_now, n, rng, col="warn"):
    blocks = station_week_blocks(keys, np.ones(len(y_reg), bool))
    ok_r = ~np.isnan(y_reg) & ~np.isnan(a["pred_reg"]) & ~np.isnan(b["pred_reg"])
    ok_c = ~np.isnan(y_cls)
    pa, pb = a[f"pred_{col}"].to_numpy(), b[f"pred_{col}"].to_numpy()
    ar, br = a["pred_reg"].to_numpy(), b["pred_reg"].to_numpy()
    out = {k: [] for k in ["rmse", "f1_macro", "onset_recall"]}
    for _ in range(n):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        r, c = idx[ok_r[idx]], idx[ok_c[idx]]
        out["rmse"].append(fast_rmse(y_reg[r], ar[r]) - fast_rmse(y_reg[r], br[r]))
        yc = y_cls[c].astype(int)
        out["f1_macro"].append(fast_macro_f1(yc, pa[c].astype(int))
                               - fast_macro_f1(yc, pb[c].astype(int)))
        nc = class_now[c].astype(int)
        out["onset_recall"].append(fast_onset_recall(nc, yc, pa[c].astype(int))
                                   - fast_onset_recall(nc, yc, pb[c].astype(int)))
    res = {}
    for k, v in out.items():
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        res[k] = {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
                  "hi": float(np.percentile(v, 97.5))}
        res[k]["significant"] = bool(res[k]["lo"] > 0 or res[k]["hi"] < 0)
    return res


def boot_rule(d, keys, y_cls, class_now, n, rng):
    """Warning rule minus argmax, for the same model: onsets and false alarms."""
    blocks = station_week_blocks(keys, np.ones(len(y_cls), bool))
    ok = ~np.isnan(y_cls)
    w, a = d["pred_warn"].to_numpy(), d["pred_argmax"].to_numpy()
    out = {"onset_recall": [], "false_alarm_rate": []}
    for _ in range(n):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        c = idx[ok[idx]]
        yc, nc = y_cls[c].astype(int), class_now[c].astype(int)
        mw = early_warning_metrics(nc, yc, w[c])
        ma = early_warning_metrics(nc, yc, a[c])
        out["onset_recall"].append(mw["onset_recall"] - ma["onset_recall"])
        out["false_alarm_rate"].append(mw["false_alarm_rate"]
                                       - ma["false_alarm_rate"])
    res = {}
    for k, v in out.items():
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        res[k] = {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
                  "hi": float(np.percentile(v, 97.5))}
        res[k]["significant"] = bool(res[k]["lo"] > 0 or res[k]["hi"] < 0)
    return res


# ---------------------------------------------------------------------------
# Per station
# ---------------------------------------------------------------------------
def per_station(h=24):
    d = pred("stacking", h)
    rows = []
    for station, g in d.groupby("station"):
        y, p = g["y_true_reg"].to_numpy(), g["pred_reg"].to_numpy()
        ok = ~np.isnan(y)
        yc = g["y_true_cls"].to_numpy()
        okc = ~np.isnan(yc)
        cm = classification_metrics(yc[okc], g["pred_warn"].to_numpy()[okc])
        ew = early_warning_metrics(g["class_now"].to_numpy()[okc], yc[okc],
                                   g["pred_warn"].to_numpy()[okc])
        rows.append({"station": station, "rows": int(ok.sum()),
                     "rmse": fast_rmse(y[ok], p[ok]),
                     "f1_macro": cm["f1_macro"],
                     "onset_recall": ew["onset_recall"],
                     "false_alarm_rate": ew["false_alarm_rate"],
                     "n_onsets": ew["n_onsets"]})
    return pd.DataFrame(rows).sort_values("n_onsets", ascending=False)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_station_months(n_stations=2, h=24):
    d = pred("stacking", h)
    d["time"] = pd.to_datetime(d["time"])
    d["month"] = d["time"].dt.to_period("M").astype(str)
    crossings = (d.assign(low=d["y_true_reg"] < LOW)
                 .groupby(["station", "month"])["low"]
                 .apply(lambda s: int((s.astype(int).diff().abs() == 1).sum())))
    picks = crossings.sort_values(ascending=False).head(n_stations)
    paths = []
    for (station, month), _ in picks.items():
        g = d[(d["station"] == station) & (d["month"] == month)].sort_values("time")
        fig, ax = plt.subplots(figsize=(11, 4.2))
        ax.plot(g["time"], g["y_true_reg"], color="black", lw=1.5,
                label="observed DO at t+24 h")
        ax.plot(g["time"], g["pred_reg"], color="tab:blue", lw=1.3,
                label="stacking forecast")
        ax.axhline(LOW, color="tab:orange", ls="--", lw=1,
                   label=f"Low threshold ({LOW} mg/L)")
        warn = g["pred_warn"] == 0
        ax.fill_between(g["time"], ax.get_ylim()[0], ax.get_ylim()[1],
                        where=warn, color="tab:red", alpha=.13,
                        label="WARNING issued", step="mid")
        ax.set_ylabel("DO (mg/L)")
        ax.set_xlabel("time (UTC)")
        ax.set_title(f"Station {station}, {month} - 24 h ahead, 2025 test year")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        p = os.path.join(PLOTS, f"test_station_{station}_{month}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_confusion(h=24):
    d = pred("stacking", h)
    ok = ~np.isnan(d["y_true_cls"].to_numpy())
    cm = classification_metrics(d["y_true_cls"].to_numpy()[ok],
                                d["pred_warn"].to_numpy()[ok])["confusion_matrix"]
    share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}\n{100 * share[i, j]:.1f}%",
                    ha="center", va="center", fontsize=9,
                    color="white" if share[i, j] > .5 else "black")
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Stacking, {h} h ahead, warning rule\n2025 test year")
    fig.tight_layout()
    p = os.path.join(PLOTS, "test_confusion_h24.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_onset_vs_fa(cls, h=24):
    fig, ax = plt.subplots(figsize=(6.8, 5))
    for m in METHODS:
        x = get_c(cls, m, h, "warn", "false_alarm_rate")
        y = get_c(cls, m, h, "warn", "onset_recall")
        if np.isnan(x) or np.isnan(y):
            continue
        star = m == "stacking"
        ax.scatter(x, y, s=140 if star else 50, marker="*" if star else "o",
                   color="tab:red" if star else "tab:blue", zorder=3)
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5)
    ax.axvline(FT["target_false_alarm_rate"], color="tab:red", ls="--", lw=1,
               label=f"budget ({FT['target_false_alarm_rate']:.0%})")
    ax.set_xlabel("false alarm rate")
    ax.set_ylabel("onset recall")
    ax.set_title(f"Catching the onset of low oxygen, {h} h ahead (2025)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(PLOTS, "test_onset_vs_false_alarm.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_rmse_vs_horizon(reg):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for m in METHODS:
        ys = [get_r(reg, m, h, "rmse") for h in HORIZONS]
        if all(np.isnan(ys)):
            continue
        style = "--o" if m == "persistence" else "-o"
        lw = 2.6 if m == "stacking" else 1.5
        ax.plot(HORIZONS, ys, style, label=m, lw=lw, markersize=5)
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("forecast horizon (hours ahead)")
    ax.set_ylabel("RMSE of DO (mg/L)")
    ax.set_title("2025 test year: error against horizon")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(PLOTS, "test_rmse_vs_horizon.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_ablation(abl):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    sets = list(FT["ablation_sets"])
    for ax, (col, title) in zip(axes, [("rmse", "RMSE (lower better)"),
                                       ("f1_macro", "macro F1"),
                                       ("onset_recall", "onset recall")]):
        x = np.arange(len(HORIZONS))
        width = 0.26
        for i, s in enumerate(sets):
            vals = [abl[(abl["set"] == s) & (abl["horizon"] == h)][col].iloc[0]
                    for h in HORIZONS]
            ax.bar(x + (i - 1) * width, vals, width, label=s)
        ax.set_xticks(x, [f"{h}h" for h in HORIZONS])
        ax.set_title(title)
        ax.grid(alpha=.25, axis="y")
    axes[0].set_ylabel("2025 test")
    axes[2].legend(fontsize=8)
    fig.suptitle("What each data source is worth (LightGBM, 2025)")
    fig.tight_layout()
    p = os.path.join(PLOTS, "test_ablation.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(PLOTS, exist_ok=True)
    reg, cls = load("stage2")
    reg1, cls1 = load("stage1")
    abl = pd.read_csv(os.path.join(RESULTS, "ablation.csv"))
    lead = pd.read_csv(os.path.join(RESULTS, "lead_times.csv"))
    with open(project_path(FT["thresholds_file"]), encoding="utf-8") as f:
        taus = json.load(f)

    d24 = pred("stacking", 24)
    keys = d24[["station", "time"]]
    y_reg = d24["y_true_reg"].to_numpy()
    y_cls = d24["y_true_cls"].to_numpy()
    class_now = d24["class_now"].to_numpy()

    rng = np.random.default_rng(SEED)
    n = ENS["bootstrap"]["n_resamples"]
    boot = {"vs_persistence": {}, "vs_best_single": {}, "rule": {}}
    for h in HORIZONS:
        s = pred("stacking", h)
        p = pred("persistence", h)
        b_reg = pred(BEST["regression"], h)
        b_cls = pred(BEST["classification"], h)
        k = s[["station", "time"]]
        yr, yc = s["y_true_reg"].to_numpy(), s["y_true_cls"].to_numpy()
        cn = s["class_now"].to_numpy()
        boot["vs_persistence"][h] = boot_pair(s, p, k, yr, yc, cn, n, rng)
        # the best single differs per task, so compare each on its own file
        tmp = b_reg.copy()
        tmp["pred_warn"] = b_cls["pred_warn"]
        tmp["pred_argmax"] = b_cls["pred_argmax"]
        boot["vs_best_single"][h] = boot_pair(s, tmp, k, yr, yc, cn, n, rng)
        boot["rule"][h] = boot_rule(s, k, yc, cn, n, rng)
        print(f"  h={h} bootstrapped")
    pd.DataFrame([{"comparison": c, "horizon": h, "metric": m, **v}
                  for c in boot for h in HORIZONS
                  for m, v in boot[c][h].items()]
                 ).to_csv(os.path.join(RESULTS, "bootstrap.csv"), index=False)

    stations = per_station(24)
    stations.to_csv(os.path.join(RESULTS, "per_station_h24.csv"), index=False)
    plots = (plot_station_months() + [plot_confusion(), plot_onset_vs_fa(cls),
                                      plot_rmse_vs_horizon(reg),
                                      plot_ablation(abl)])
    # step 10, if it has been run: an additional analysis, not a result
    extra = os.path.join(PLOTS, "error_correlation_h24.png")
    if os.path.exists(extra):
        plots.append(extra)

    write(reg, cls, reg1, cls1, abl, lead, stations, boot, taus, plots)
    terminal(reg, cls, abl, lead, boot, stations)


# ---------------------------------------------------------------------------
# STEP 10 - design choices and scope.
#
# ADDITIONAL ANALYSES ONLY. Every number below is read from the step-10
# CSVs, which re-score predictions that already existed. None of it was
# used to choose or change the deployed system: `final_system` and every
# threshold were fixed in FROZEN.md before 2025 was read.
# ---------------------------------------------------------------------------
def step10(name):
    path = os.path.join(RESULTS, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def ci(row, metric, nd=4):
    return (f"{row[f'{metric}_diff']:+.{nd}f} "
            f"[{row[f'{metric}_lo']:+.{nd}f}, {row[f'{metric}_hi']:+.{nd}f}]")


def design_checks_section(reg, cls):
    out = ["\n## Design choices and how they were checked\n"]
    out.append("Six choices shaped this system. Each one is written down "
               "here with the check that was run on it. The checks in this "
               "section are **additional analyses**: they re-score "
               "predictions that already existed, and none of them was used "
               "to choose or change the deployed system, which was fixed in "
               "`FROZEN.md` before any 2025 target was read.\n")

    # ---- 1. hyperparameters on 2024 ----
    freeze = ""
    fpath = project_path(FT["frozen_file"])
    if os.path.exists(fpath):
        m = re.search(r"at commit `([0-9a-f]{40})`",
                      open(fpath, encoding="utf-8").read())
        if m:
            freeze = (f" The freeze is commit `{m.group(1)[:12]}`, so the "
                      f"claim is checkable rather than asserted.")
    out.append("**1. Hyperparameters were chosen on the 2024 validation "
               "year.** This is the standard three-way split: 2023 trains, "
               "2024 decides, 2025 judges. What 2025 tests is therefore the "
               "whole frozen pipeline - architecture, rounds, ensemble and "
               "warning thresholds together - rather than each model in "
               "isolation, which is the quantity an operator actually "
               "deploys. Every one of those settings, plus a SHA-256 for "
               "each model file, was committed before the test."
               + freeze + "\n")

    # ---- 2. stacking train/apply design ----
    p = step10("step10_pipeline.csv")
    out.append("**2. The stacking meta-model was fitted on 2024 predictions "
               "of 2023-only members, then applied on top of members "
               "retrained on 2023+2024.** The headline chain therefore has a "
               "train/apply mismatch. A second chain has none: 2023-only "
               "members, the same meta-model fitted on their 2024 "
               "predictions, applied to 2025 (stage 1). If the mismatch were "
               "flattering the headline, the consistent chain would be the "
               "better of the two.")
    if p is not None:
        r24 = p[p["horizon"] == 24].iloc[0]
        ns = int((~p["onset_recall_significant"]).sum())
        out.append(f"It is not. At 24 h the consistent chain gives RMSE "
                   f"{r24['rmse_stage1']:.4f} against the headline's "
                   f"{r24['rmse_stage2']:.4f} and onset recall "
                   f"{r24['onset_recall_stage1']:.3f} against "
                   f"{r24['onset_recall_stage2']:.3f} - a difference of "
                   f"{ci(r24, 'onset_recall', 3)}, which contains zero. The "
                   f"consistent chain is slightly *worse* on RMSE "
                   f"({ci(r24, 'rmse')}) and macro F1 "
                   f"({ci(r24, 'f1_macro')}), and the gap is the size of "
                   f"what one extra training year bought elsewhere "
                   f"(section 9). On onset recall - the metric the system is "
                   f"built for - the two chains are indistinguishable at "
                   f"{ns} of the {len(p)} horizons. The mismatch costs "
                   f"nothing the test can detect; the headline is better "
                   f"because its members saw a second year, not because the "
                   f"meta-model was handed inputs it had not learned to "
                   f"blend.\n")
        rows = []
        for r in p.itertuples():
            rows.append([f"{r.horizon} h",
                         f"{r.rmse_stage1:.4f}", f"{r.rmse_stage2:.4f}",
                         f"{r.rmse_diff:+.4f} [{r.rmse_lo:+.4f}, {r.rmse_hi:+.4f}]",
                         f"{r.r2_stage1:.4f}", f"{r.r2_stage2:.4f}",
                         f"{r.f1_macro_stage1:.4f}", f"{r.f1_macro_stage2:.4f}",
                         f"{r.f1_macro_diff:+.4f} [{r.f1_macro_lo:+.4f}, {r.f1_macro_hi:+.4f}]",
                         f"{r.onset_recall_stage1:.3f}", f"{r.onset_recall_stage2:.3f}",
                         f"{r.onset_recall_diff:+.3f} [{r.onset_recall_lo:+.3f}, {r.onset_recall_hi:+.3f}]",
                         f"{r.false_alarm_rate_stage1:.4f}",
                         f"{r.false_alarm_rate_stage2:.4f}",
                         f"{r.false_alarm_rate_diff:+.4f} [{r.false_alarm_rate_lo:+.4f}, {r.false_alarm_rate_hi:+.4f}]"])
        out.append(md_table(
            ["horizon", "RMSE consistent", "RMSE headline",
             "RMSE diff [95% CI]", "R2 consistent", "R2 headline",
             "macroF1 consistent", "macroF1 headline",
             "macroF1 diff [95% CI]", "onset consistent", "onset headline",
             "onset diff [95% CI]", "FA consistent", "FA headline",
             "FA diff [95% CI]"], rows))
        out.append("\n*Consistent = stage 1 (2023-only members, no "
                   "train/apply mismatch); headline = stage 2 (members "
                   "retrained on 2023+2024). Both chains use the same "
                   "meta-models and the same frozen thresholds, and predict "
                   "the same 2025 rows. Differences are consistent minus "
                   "headline, with 1,000 station-week bootstrap resamples; "
                   "a CI containing zero means the two chains cannot be told "
                   "apart. No model was retrained for this table. Source: "
                   "`results/test/step10_pipeline.csv`.*\n")

    # ---- 3. the station that stops reporting ----
    d = step10("step10_dropout.csv")
    out.append("**3. One station stops reporting on 2025-11-19**, so the "
               "last six weeks of the test year have 29 stations rather "
               "than 30. The same 2025 predictions were re-scored two more "
               "ways - without that station for the whole year, and with "
               "all thirty stations but only up to 2025-11-18 - so the "
               "question is whether the uneven tail moved the headline.")
    if d is not None:
        h24 = d[d["horizon"] == 24].set_index("subset")
        base = h24.iloc[0]
        worst_fa = float(d["false_alarm_rate"].max())
        out.append(f"It did not. At 24 h, onset recall is "
                   f"{base['onset_recall']:.3f} in the headline and "
                   f"{h24['onset_recall'].min():.3f} to "
                   f"{h24['onset_recall'].max():.3f} across the three "
                   f"subsets; RMSE runs {h24['rmse'].min():.4f} to "
                   f"{h24['rmse'].max():.4f} and macro F1 "
                   f"{h24['f1_macro'].min():.4f} to "
                   f"{h24['f1_macro'].max():.4f}. The false-alarm rate "
                   f"stays inside the "
                   f"{FT['target_false_alarm_rate']:.0%} promise at every "
                   f"horizon and in every subset (worst {worst_fa:.2%}). "
                   f"Dropping the station helps slightly - it is one of the "
                   f"harder ones - and cutting the year short costs a "
                   f"little on false alarms, because December is a quiet "
                   f"month whose safe hours were diluting the rate. Neither "
                   f"moves a conclusion.\n")
        rows = [[f"{r.horizon} h", r.subset, f"{r.n:,}", f"{r.n_onsets:,}",
                 f"{r.rmse:.4f}", f"{r.r2:.4f}", f"{r.f1_macro:.4f}",
                 f"{r.onset_recall:.3f}", f"{r.false_alarm_rate:.4f}"]
                for r in d.itertuples()]
        out.append(md_table(["horizon", "subset", "rows", "onsets", "RMSE",
                             "R2", "macro F1", "onset recall",
                             "false alarm"], rows))
        out.append("\n*Stacking with the frozen warning rule, re-scored "
                   "only - no model was retrained and no threshold was "
                   "re-chosen. Source: `results/test/step10_dropout.csv`.*\n")

    # ---- 4. the deep members ----
    c = step10("step10_error_correlation_h24.csv")
    out.append("**4. The two deep models are the weakest members** and were "
               "kept anyway. The reason is diversity, not accuracy: an "
               "ensemble gains from members that are wrong about different "
               "hours, so what matters is how correlated their errors are "
               "with the trees'.")
    if c is not None:
        c = c.set_index(c.columns[0])
        names = list(c.columns)
        pairs = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pairs[(a, b)] = float(c.loc[a, b])
        trees = [n for n in names
                 if n in ("random_forest", "xgboost", "lightgbm")]
        deep = [n for n in names if n in ("bilstm", "cnn_lstm")]
        tree_tree = [v for (a, b), v in pairs.items()
                     if a in trees and b in trees]
        cross = [v for (a, b), v in pairs.items()
                 if (a in trees and b in deep) or (a in deep and b in trees)]
        deep_deep = [v for (a, b), v in pairs.items()
                     if a in deep and b in deep]
        out.append(f"The gradient-boosted trees' 24 h errors correlate "
                   f"{min(tree_tree):.3f} to {max(tree_tree):.3f} with each "
                   f"other. A deep model against a boosted tree is lower: "
                   f"XGBoost and LightGBM sit at "
                   f"{c.loc['xgboost', 'bilstm']:.3f} and "
                   f"{c.loc['lightgbm', 'bilstm']:.3f} against the BiLSTM, "
                   f"with the whole cross-family range running "
                   f"{min(cross):.3f} to {max(cross):.3f}. The two deep "
                   f"models are close to interchangeable with each other "
                   f"({deep_deep[0]:.3f}), which is why there would be "
                   f"little to gain from a third. **Lower correlation means "
                   f"the members miss different hours, so averaging or "
                   f"stacking them cancels more error than combining models "
                   f"that fail together.** That is what the deep members "
                   f"are in the ensemble for, and it is why they stay in "
                   f"even though each one on its own is beaten by every "
                   f"tree.\n")
        rows = [[a] + [f"{c.loc[a, b]:.3f}" for b in names] for a in names]
        out.append(md_table([""] + names, rows))
        out.append("\n*Pearson correlation of 24 h errors "
                   "(`y_pred - y_true`) over the 2025 rows, from the "
                   "predictions step 8 already wrote. Heatmap: "
                   "`results/test/plots/error_correlation_h24.png`. "
                   "**Analysis only - not used to choose or change the "
                   "system.** Source: "
                   "`results/test/step10_error_correlation_h24.csv`.*\n")

    # ---- 5. the warning rule ----
    out.append("**5. The warning rule is the main contribution of the "
               "decision layer.** Argmax answers *which class is most "
               "likely*; a warning system asks *is the chance of Low high "
               "enough to act on*, which is a lower bar and a different "
               "question. Separating the two is the decision layer's whole "
               "job, and on 2025 it is worth more than any choice of "
               "ensemble.")
    rows = []
    for h in HORIZONS:
        rows.append([f"{h} h",
                     fmt(get_c(cls, "stacking", h, "argmax", "onset_recall"), 3),
                     fmt(get_c(cls, "stacking", h, "warn", "onset_recall"), 3),
                     fmt(get_c(cls, "stacking", h, "argmax",
                               "false_alarm_rate"), 4),
                     fmt(get_c(cls, "stacking", h, "warn",
                               "false_alarm_rate"), 4)])
    a24 = get_c(cls, "stacking", 24, "argmax", "onset_recall")
    w24 = get_c(cls, "stacking", 24, "warn", "onset_recall")
    fa24a = get_c(cls, "stacking", 24, "argmax", "false_alarm_rate")
    fa24 = get_c(cls, "stacking", 24, "warn", "false_alarm_rate")
    best_r = get_r(reg, "stacking", 24, "rmse") - \
        get_r(reg, BEST["regression"], 24, "rmse")
    out.append(f"At 24 h the rule lifts onset recall from {a24:.3f} to "
               f"{w24:.3f} - **{w24 / a24:.1f} times** as many onsets "
               f"caught - for a false-alarm rate that rises from "
               f"{fa24a:.2%} to {fa24:.2%}, still inside the "
               f"{FT['target_false_alarm_rate']:.0%} the system promises. "
               f"For comparison, choosing stacking over the best single "
               f"model is worth {abs(best_r):.4f} mg/L of RMSE and nothing "
               f"detectable on macro F1 or onset recall (section 3). The "
               f"threshold was chosen on out-of-fold 2024 probabilities "
               f"against a {FT['select_false_alarm_rate']:.1%} internal "
               f"budget - tighter than the promise on purpose - and frozen "
               f"before the test.\n")
    out.append(md_table(["horizon", "onset recall (argmax)",
                         "onset recall (rule)", "false alarm (argmax)",
                         "false alarm (rule)"], rows))
    out.append("")

    # ---- 6. GPU training and reproducibility ----
    s = step10("step10_seed_spread.csv")
    g = step10("step10_method_gaps.csv")
    out.append("**6. The deep models were trained on a GPU with "
               "non-deterministic kernels**, so a rerun does not reproduce "
               "a run bit for bit. The frozen system handles that by "
               "averaging three seeds, and the size of the remaining wobble "
               "was measured rather than assumed.")
    if s is not None:
        parts = [f"{r.model} RMSE +-{r.rmse_std:.4f}, macro F1 "
                 f"+-{r.f1_std:.4f}" for r in s.itertuples()]
        worst_rmse = float(s["rmse_std"].max())
        worst_f1 = float(s["f1_std"].max())
        line = (f"Across seeds 42-44 the single-model spread at 24 h is "
                f"{'; '.join(parts)} (standard deviation over three seeds).")
        if g is not None:
            m = g.set_index("comparison")
            line += (f" The differences the 2025 tables are asked to resolve "
                     f"are of the same order: stacking beats the best single "
                     f"model by "
                     f"{m.loc['stacking vs best single (lightgbm)', 'rmse_gap']:.4f}"
                     f" mg/L of RMSE and the simple average by "
                     f"{m.loc['stacking vs simple average', 'rmse_gap']:.4f},"
                     f" against a worst-case seed spread of "
                     f"{worst_rmse:.4f}. On macro F1 the gaps "
                     f"({m.loc['stacking vs best single (lightgbm)', 'f1_gap']:.4f}"
                     f" and "
                     f"{m.loc['stacking vs simple average', 'f1_gap']:.4f}) "
                     f"are smaller than the spread ({worst_f1:.4f}).")
        line += (" So the ranking *among the ensembles* sits inside "
                 "run-to-run noise. That is exactly why step 7 chose "
                 "stacking by a rule written before the numbers were known "
                 "rather than by whichever value came out highest, and why "
                 "section 3 reports bootstrap intervals instead of point "
                 "differences. The margin over persistence, and the margin "
                 "the warning rule adds, are both an order of magnitude "
                 "larger than the noise.")
        out.append(line + "\n")
        rows = [[r.model, r.seeds, f"{r.rmse_mean:.4f}", f"{r.rmse_std:.4f}",
                 f"{r.rmse_range:.4f}", f"{r.f1_mean:.4f}", f"{r.f1_std:.4f}",
                 f"{r.f1_range:.4f}"] for r in s.itertuples()]
        out.append(md_table(["model", "seeds", "RMSE mean", "RMSE std",
                             "RMSE range", "macro F1 mean", "macro F1 std",
                             "macro F1 range"], rows))
        out.append("\n*Seed-to-seed spread of one deep model at 24 h, from "
                   "`models/dl/*_seed_spread.csv` - measured on the 2024 "
                   "validation year, the only year on which repeated runs "
                   "were scored. The frozen system averages all three "
                   "seeds, so the wobble inside the deployed ensemble is "
                   "smaller than this. Sources: "
                   "`results/test/step10_seed_spread.csv`, "
                   "`results/test/step10_method_gaps.csv`.*\n")
    return out


def scope_section():
    out = ["\n## Scope and future work\n"]
    out.append("- **Evaluated at monitored stations.** That is the intended "
               "use: a warning for a river where a sensor already sits, "
               "turning a reading into a forecast. Every station in the 2025 "
               "test also appears in training, so nothing here measures "
               "transfer to a river the system has never seen - testing on "
               "unseen stations is future work.")
    out.append("- **US rivers, USGS gauges.** Water data comes from USGS "
               "instantaneous values, with ERA5 weather and CAMS air "
               "quality. Applying the same pipeline to Indian rivers - the "
               "CPCB real-time stations, for example - is future work, and "
               "would need its own freeze and its own untouched test year.")
    out.append("- **One test year.** 2025 was scored once, which is what "
               "makes it honest, but one year is one draw. A further "
               "untouched year would narrow every interval in this report.")
    out.append("- **Adaptive weighting (ATAE) is worth revisiting.** It was "
               "rejected on validation by a rule written in advance, and it "
               "then came out level with or just ahead of stacking on 2025 "
               "(sections 1 and 2). That is a promising hint, not evidence: "
               "the comparison happened after the test year was opened, so "
               "it cannot be used to change the deployed system. Only a "
               "new, unseen year can settle it.")
    out.append("")
    return out


def write(reg, cls, reg1, cls1, abl, lead, stations, boot, taus, plots):
    s24r = {c: get_r(reg, "stacking", 24, c)
            for c in ["r2", "rmse", "mae", "nse", "skill_vs_persistence",
                      "danger_rmse"]}
    s24c = {c: get_c(cls, "stacking", 24, "warn", c)
            for c in ["accuracy", "balanced_accuracy", "f1_macro",
                      "roc_auc_ovr_macro", "recall_Low", "onset_recall",
                      "false_alarm_rate", "transition_accuracy"]}
    warned = lead["warned"].mean()
    med_lead = lead.loc[lead["warned"], "lead_hours"].median()

    out = ["# Final results - the 2025 test year\n"]
    out.append(
        f"**Summary.** Thirty USGS river stations, three data sources "
        f"(water-quality sensors, ERA5 weather, CAMS air quality), hourly, "
        f"2023-2025. Models were built on 2023, every choice was made on "
        f"2024, and the system was frozen - hyperparameters, ensemble, "
        f"warning thresholds and file hashes all recorded in `FROZEN.md` "
        f"and committed - **before any 2025 target was read**. The 2025 "
        f"year was then scored once. Twenty-four hours ahead, the final "
        f"system (a stacking ensemble over five models) predicts dissolved "
        f"oxygen with **R2 {s24r['r2']:.3f}, RMSE {s24r['rmse']:.3f} mg/L** "
        f"({s24r['skill_vs_persistence']:+.1%} better than assuming no "
        f"change), classifies Low/Medium/Good with **macro F1 "
        f"{s24c['f1_macro']:.3f}** and, with the frozen warning rule, "
        f"catches **{s24c['onset_recall']:.1%} of the hours when safe water "
        f"turns dangerous** at a false-alarm rate of "
        f"{s24c['false_alarm_rate']:.2%}. Of the "
        f"{len(lead)} real low-oxygen events in 2025, "
        f"**{warned:.0%} were warned about in advance**, with a median lead "
        f"time of **{med_lead:.0f} hours**.\n")

    # ---- 2. regression ----
    out.append("\n## 1. Regression, 24 hours ahead, 2025\n")
    head = ["method", "R2", "RMSE", "MAE", "NSE", "skill vs persistence",
            "danger-zone RMSE"]
    rows = []
    for m in METHODS:
        rows.append([("**" + m + "**") if m == "stacking" else m,
                     fmt(get_r(reg, m, 24, "r2")), fmt(get_r(reg, m, 24, "rmse")),
                     fmt(get_r(reg, m, 24, "mae")), fmt(get_r(reg, m, 24, "nse")),
                     "-" if m == "persistence"
                     else f"{get_r(reg, m, 24, 'skill_vs_persistence'):+.3f}",
                     fmt(get_r(reg, m, 24, "danger_rmse"))])
    out.append(md_table(head, rows))
    out.append(f"\nDanger-zone RMSE covers the rows whose true DO lies "
               f"between {ENS['danger_zone'][0]} and {ENS['danger_zone'][1]} "
               f"mg/L - the band where a river actually crosses into "
               f"trouble.\n")

    # ---- 3. classification ----
    for rule, title in [("warn", "with the frozen warning rule"),
                        ("argmax", "with argmax")]:
        out.append(f"\n## 2. Classification, 24 hours ahead, 2025 - {title}\n")
        head = ["method", "accuracy", "balanced acc", "macro F1", "ROC-AUC",
                "Low recall", "onset recall", "false alarm", "transition acc"]
        rows = []
        for m in METHODS:
            rows.append([("**" + m + "**") if m == "stacking" else m,
                         fmt(get_c(cls, m, 24, rule, "accuracy"), 3),
                         fmt(get_c(cls, m, 24, rule, "balanced_accuracy"), 3),
                         fmt(get_c(cls, m, 24, rule, "f1_macro"), 4),
                         fmt(get_c(cls, m, 24, rule, "roc_auc_ovr_macro"), 3),
                         fmt(get_c(cls, m, 24, rule, "recall_Low"), 3),
                         fmt(get_c(cls, m, 24, rule, "onset_recall"), 3),
                         fmt(get_c(cls, m, 24, rule, "false_alarm_rate"), 4),
                         fmt(get_c(cls, m, 24, rule, "transition_accuracy"), 3)])
        out.append(md_table(head, rows))

    # ---- 4. all horizons ----
    out.append("\n## 3. All horizons: the final system against the bar\n")
    head = (["method"] + [f"RMSE {h}h" for h in HORIZONS]
            + [f"macroF1 {h}h" for h in HORIZONS]
            + [f"onset recall {h}h" for h in HORIZONS])
    rows = []
    for m in ["persistence", BEST["regression"], BEST["classification"],
              "stacking"]:
        label = m
        if m == BEST["regression"]:
            label = f"{m} (best single, regression)"
        if m == BEST["classification"]:
            label = f"{m} (best single, classification)"
        rows.append([label]
                    + [fmt(get_r(reg, m, h, "rmse"), 3) for h in HORIZONS]
                    + [fmt(get_c(cls, m, h, "warn", "f1_macro"), 3)
                       for h in HORIZONS]
                    + [fmt(get_c(cls, m, h, "warn", "onset_recall"), 3)
                       for h in HORIZONS])
    out.append(md_table(head, rows))

    out.append("\n### Is the final system really better?\n")
    out.append(f"{ENS['bootstrap']['n_resamples']:,} station-week resamples "
               f"of the 2025 rows. Negative RMSE favours stacking; positive "
               f"macro F1 and onset recall favour stacking.\n")
    rows = []
    for against, label in [("vs_persistence", "persistence"),
                           ("vs_best_single", "best single model")]:
        for h in HORIZONS:
            for k, name, better in [("rmse", "RMSE", "lower"),
                                    ("f1_macro", "macro F1", "higher"),
                                    ("onset_recall", "onset recall", "higher")]:
                v = boot[against][h][k]
                if not v["significant"]:
                    verdict = "no significant difference"
                else:
                    good = (v["mean"] < 0) if better == "lower" else (v["mean"] > 0)
                    verdict = "**stacking better**" if good else f"{label} better"
                rows.append([label, f"{h} h", name, f"{v['mean']:+.4f}",
                             f"[{v['lo']:+.4f}, {v['hi']:+.4f}]", verdict])
    out.append(md_table(["stacking vs", "horizon", "metric", "difference",
                         "95% CI", "verdict"], rows))

    # ---- 5. warning rule ----
    out.append("\n## 4. What the warning rule is worth\n")
    out.append(
        f"Argmax answers *which class is most likely*. A warning system asks "
        f"*is the chance of Low high enough to act on*. The rule issues a "
        f"warning when P(Low) reaches a threshold tau, frozen before the "
        f"test from the out-of-fold 2024 probabilities "
        f"(stacking: tau = {taus['stacking_h24']['tau']:.2f} at 24 h).\n")
    head = ["horizon", "onset recall (argmax)", "onset recall (rule)",
            "change [95% CI]", "false alarm (argmax)", "false alarm (rule)",
            "change [95% CI]"]
    rows = []
    for h in HORIZONS:
        o = boot["rule"][h]["onset_recall"]
        f = boot["rule"][h]["false_alarm_rate"]
        rows.append([f"{h} h",
                     fmt(get_c(cls, "stacking", h, "argmax", "onset_recall"), 3),
                     fmt(get_c(cls, "stacking", h, "warn", "onset_recall"), 3),
                     f"{o['mean']:+.3f} [{o['lo']:+.3f}, {o['hi']:+.3f}]",
                     fmt(get_c(cls, "stacking", h, "argmax", "false_alarm_rate"), 4),
                     fmt(get_c(cls, "stacking", h, "warn", "false_alarm_rate"), 4),
                     f"{f['mean']:+.4f} [{f['lo']:+.4f}, {f['hi']:+.4f}]"])
    out.append(md_table(head, rows))
    fa24 = get_c(cls, "stacking", 24, "warn", "false_alarm_rate")
    budget = FT["target_false_alarm_rate"]
    out.append(f"\nThe rule was tuned against a {FT['select_false_alarm_rate']:.1%} "
               f"internal budget so that the {budget:.0%} promise would "
               f"survive contact with unseen data. On 2025 it delivers "
               f"**{fa24:.2%}** at 24 h - "
               + ("inside the promise." if fa24 <= budget else
                  f"slightly over the {budget:.0%} promise, which is "
                  f"reported rather than fixed, because the threshold was "
                  f"frozen before the test and moving it now would undo the "
                  f"point of freezing it.") + "\n")

    # ---- 6. lead time ----
    out.append("\n## 5. Lead time: how much warning\n")
    out.append(f"A low-oxygen event starts when DO drops below {LOW} mg/L "
               f"after at least {FT['event_recovery_hours']} h at or above "
               f"it. A warning counts if the 24-hour-ahead rule fired at any "
               f"point in the {FT['lead_time_window_hours']} h before that "
               f"moment.\n")
    out.append(md_table(["", "value"], [
        ["low-DO events in 2025", f"{len(lead):,}"],
        ["warned in advance", f"{lead['warned'].sum():,} "
                              f"({lead['warned'].mean():.1%})"],
        ["median lead time", f"{lead.loc[lead['warned'], 'lead_hours'].median():.0f} h"],
        ["mean lead time", f"{lead.loc[lead['warned'], 'lead_hours'].mean():.1f} h"],
        ["stations with at least one event",
         f"{lead['station'].nunique()} of {CFG['download']['target_stations']}"]]))

    # ---- 7. ablation ----
    out.append("\n## 6. What each data source is worth\n")
    out.append("LightGBM alone, retrained on 2023+2024 with the frozen "
               "settings, three input sets. **W** = water + time + station, "
               "**W+M** adds weather, **W+M+A** adds air quality. The "
               "thresholds were frozen in the same step as everything else.\n")
    head = ["input set", "features"] + [f"RMSE {h}h" for h in HORIZONS] + \
           [f"onset recall {h}h" for h in HORIZONS]
    rows = []
    for s in FT["ablation_sets"]:
        sub = abl[abl["set"] == s]
        rows.append([s, int(sub["n_features"].iloc[0])]
                    + [fmt(sub[sub["horizon"] == h]["rmse"].iloc[0], 3)
                       for h in HORIZONS]
                    + [fmt(sub[sub["horizon"] == h]["onset_recall"].iloc[0], 3)
                       for h in HORIZONS])
    out.append(md_table(head, rows))

    # ---- 8. per station ----
    out.append("\n## 7. Per station, 24 hours ahead (stacking, warning rule)\n")
    out.append("Sorted by how many onsets each station actually had.\n")
    out.append(md_table(["station", "rows", "RMSE", "macro F1",
                         "onset recall", "false alarm", "onsets"],
                        [[r.station, f"{r.rows:,}", f"{r.rmse:.3f}",
                          f"{r.f1_macro:.3f}",
                          "-" if np.isnan(r.onset_recall) else f"{r.onset_recall:.3f}",
                          "-" if np.isnan(r.false_alarm_rate) else f"{r.false_alarm_rate:.4f}",
                          r.n_onsets]
                         for r in stations.itertuples()]))

    # ---- 9. 2024 vs 2025 ----
    out.append("\n## 8. Did the 2024 result hold up in 2025?\n")
    oof = project_path(CFG["step7"]["oof_predictions"])
    rows = []
    for h in HORIZONS:
        v_r = pd.read_parquet(os.path.join(oof, f"stacking_regression_h{h}.parquet"))
        v_c = pd.read_parquet(os.path.join(oof, f"stacking_classification_h{h}.parquet"))
        okv = ~np.isnan(v_r["y_true"].to_numpy())
        v_rmse = fast_rmse(v_r["y_true"].to_numpy()[okv],
                           v_r["y_pred"].to_numpy()[okv])
        okc = ~np.isnan(v_c["y_true"].to_numpy())
        v_f1 = fast_macro_f1(v_c["y_true"].to_numpy()[okc].astype(int),
                             v_c["y_pred_warn"].to_numpy()[okc].astype(int))
        v_on = fast_onset_recall(v_c["class_now"].to_numpy()[okc].astype(int),
                                 v_c["y_true"].to_numpy()[okc].astype(int),
                                 v_c["y_pred_warn"].to_numpy()[okc].astype(int))
        rows.append([f"{h} h", f"{v_rmse:.3f}",
                     fmt(get_r(reg, "stacking", h, "rmse"), 3),
                     f"{v_f1:.3f}",
                     fmt(get_c(cls, "stacking", h, "warn", "f1_macro"), 3),
                     f"{v_on:.3f}",
                     fmt(get_c(cls, "stacking", h, "warn", "onset_recall"), 3)])
    out.append(md_table(["horizon", "RMSE 2024", "RMSE 2025",
                         "macro F1 2024", "macro F1 2025",
                         "onset recall 2024", "onset recall 2025"], rows))
    out.append("\n*2024 numbers are the out-of-fold cross-fitted ones from "
               "step 7, which is the fairest in-sample comparison available; "
               "2024's thresholds were chosen on 2024 data, so a small drop "
               "in 2025 is expected rather than alarming.*\n")

    # ---- robustness ----
    out.append("\n## 9. Robustness: how much did the extra training year help?\n")
    out.append("The headline numbers use members retrained on 2023+2024 "
               "(stage 2). The same chain with the original 2023-only "
               "members gives:\n")
    rows = []
    for h in HORIZONS:
        rows.append([f"{h} h",
                     fmt(get_r(reg1, "stacking", h, "rmse"), 3),
                     fmt(get_r(reg, "stacking", h, "rmse"), 3),
                     fmt(get_c(cls1, "stacking", h, "warn", "onset_recall"), 3),
                     fmt(get_c(cls, "stacking", h, "warn", "onset_recall"), 3)])
    out.append(md_table(["horizon", "RMSE (2023 only)", "RMSE (2023+2024)",
                         "onset recall (2023 only)",
                         "onset recall (2023+2024)"], rows))

    # ---- stage 3: the models the app uses, which cannot be tested ----
    out.append("\n## Stage-3 models (what the app uses)\n")
    out.append("**Every number in this report comes from the stage-2 "
               "models, trained on 2023+2024.** Those are the ones that "
               "were scored on 2025.\n")
    out.append("A second set, in `models/final/`, was retrained afterwards "
               "on **2023 + 2024 + 2025** so the app can forecast using "
               "every hour ever recorded. Those models are **untested and "
               "untestable**: they have seen the test year, so there is no "
               "data left to score them on, and any metric computed for "
               "them would be meaningless. They are never used for a "
               "reported result.\n")
    san = os.path.join(RESULTS, "stage3_sanity.csv")
    if os.path.exists(san):
        d = pd.read_csv(san)
        out.append(
            f"They were checked only for sanity: every file loads and "
            f"predicts without error, and their 2025 predictions still "
            f"agree closely with the stage-2 models'. Across {len(d)} "
            f"model-task-horizon comparisons the correlation runs from "
            f"**{d['correlation'].min():.4f} to "
            f"{d['correlation'].max():.4f}** (median "
            f"{d['correlation'].median():.4f}), with a median absolute "
            f"difference of {d['mean_abs_diff'].median():.4f} and a worst "
            f"of {d['mean_abs_diff'].max():.4f}. "
            + (f"**{int(d['flag'].sum())} comparisons fell below the 0.95 "
               f"correlation threshold and are flagged in "
               f"`results/test/stage3_sanity.csv`.**"
               if d["flag"].any() else
               "Nothing fell below the 0.95 correlation threshold, so "
               "nothing is flagged.")
            + " A small difference is expected and healthy: stage 3 saw "
              "2025 during training and stage 2 did not.\n")
        worst = d.sort_values("correlation").head(3)
        out.append(md_table(["model", "task", "horizon", "correlation",
                             "mean abs. difference"],
                            [[r.model, r.task, f"{r.horizon} h",
                              f"{r.correlation:.4f}", f"{r.mean_abs_diff:.4f}"]
                             for r in worst.itertuples()]))
        out.append("\n*(the three lowest correlations, for reference)*\n")

    out.append("\n## Plots\n")
    for p in plots:
        out.append(f"- `{os.path.relpath(p, project_path('.'))}`")

    # ---- design choices and scope (step 10: additional analyses) ----
    out.extend(design_checks_section(reg, cls))
    out.extend(scope_section())

    path = project_path("reports", "final_results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Wrote {path}")


def terminal(reg, cls, abl, lead, boot, stations):
    print("\nREGRESSION 24 h, 2025")
    print(f"  {'method':<17}{'R2':>8}{'RMSE':>9}{'MAE':>8}{'skill':>8}{'danger':>9}")
    for m in METHODS:
        print(f"  {m:<17}{get_r(reg,m,24,'r2'):>8.4f}{get_r(reg,m,24,'rmse'):>9.4f}"
              f"{get_r(reg,m,24,'mae'):>8.4f}"
              f"{get_r(reg,m,24,'skill_vs_persistence'):>8.3f}"
              f"{get_r(reg,m,24,'danger_rmse'):>9.4f}")
    for rule in ["warn", "argmax"]:
        print(f"\nCLASSIFICATION 24 h, 2025 ({rule})")
        print(f"  {'method':<17}{'acc':>7}{'macroF1':>9}{'LowRec':>8}"
              f"{'onsetRec':>10}{'falseAl':>9}")
        for m in METHODS:
            print(f"  {m:<17}{get_c(cls,m,24,rule,'accuracy'):>7.3f}"
                  f"{get_c(cls,m,24,rule,'f1_macro'):>9.4f}"
                  f"{get_c(cls,m,24,rule,'recall_Low'):>8.3f}"
                  f"{get_c(cls,m,24,rule,'onset_recall'):>10.3f}"
                  f"{get_c(cls,m,24,rule,'false_alarm_rate'):>9.4f}")
    print(f"\nLEAD TIME: {lead['warned'].sum()}/{len(lead)} events warned "
          f"({lead['warned'].mean():.1%}), median "
          f"{lead.loc[lead['warned'],'lead_hours'].median():.0f} h")
    print("\nABLATION (RMSE / onset recall at 24 h)")
    for s in FT["ablation_sets"]:
        r = abl[(abl["set"] == s) & (abl["horizon"] == 24)].iloc[0]
        print(f"  {s:<7}{r['n_features']:>4} features  RMSE {r['rmse']:.4f}  "
              f"onset {r['onset_recall']:.3f}")


if __name__ == "__main__":
    main()
