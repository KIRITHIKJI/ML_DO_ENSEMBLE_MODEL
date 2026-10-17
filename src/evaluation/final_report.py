"""
STEP 8.5 - the final report on the 2025 test.

Run from the project root (after src.evaluation.test_2025):

    python -m src.evaluation.final_report

Reads results/test/ and writes reports/final_results.md plus the plots.
Nothing is fitted, tuned or re-scored here.
"""

import json
import os

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

    write(reg, cls, reg1, cls1, abl, lead, stations, boot, taus, plots)
    terminal(reg, cls, abl, lead, boot, stations)


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

    # ---- limitations ----
    out.append("\n## Limitations\n")
    out.append("- **The members' hyperparameters were chosen on 2024.** The "
               "2025 test is honest about the *system*, but the models "
               "inside it were tuned on the validation year, so 2025 is a "
               "fair test of the pipeline rather than of each model.")
    out.append("- **The stacking meta-model was fitted on 2024 predictions "
               "of 2023-only members**, then applied on top of members "
               "retrained on 2023+2024. The inputs it sees at test time come "
               "from slightly better-trained models than the ones it learned "
               "to blend. Section 9 shows the effect is small.")
    out.append("- **One station stops reporting on 2025-11-19**, so the last "
               "six weeks of the test year have 29 stations rather than 30.")
    out.append("- **US rivers only**, 30 stations, three years. Nothing here "
               "says how the system behaves on a river it has never seen - "
               "every station in the test appears in training.")
    out.append("- **The deep models are the weakest members.** The BiLSTM "
               "and CNN-LSTM were tuned in step 5 and still trail the "
               "gradient-boosted trees; they earn their place in the "
               "ensemble by being different, not by being better.")
    out.append("- **Two ensemble ideas were tested and rejected.** ATAE "
               "(adaptive threshold-aware weighting, step 6) matched a plain "
               "average and was not worth its four tuned parameters. CAE "
               "(context-aware meta-model, step 7) failed its pre-set "
               "decision rule outright - worse onset recall and worse RMSE "
               "than stacking. Both are still scored here for the record.")
    out.append("- **The gains are concentrated in the decision rule, not the "
               "ensemble.** Most of the onset-detection performance comes "
               "from choosing a probability threshold properly; the "
               "difference between combining five models cleverly and "
               "averaging them is small.\n")

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
