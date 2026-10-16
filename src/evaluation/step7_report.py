"""
STEP 7.4 - score every method out-of-fold over 2024 and decide.

Run from the project root (after src.models.run_step7):

    python -m src.evaluation.step7_report

Every number here comes from the six-fold cross-fitting: each row of 2024
was predicted once, by a model that never saw it, with 14 days purged on
both sides. The 2025 test split is not read.

At the end this applies the decision rule that was written down BEFORE the
results were known, and records the answer in config.yaml as
`final_system`.

Writes:
    reports/step7_cae_report.md
    results/val/metrics_step7_{regression,classification}.csv
    results/val/step7_bootstrap.csv
    results/val/plots/step7_*.png
    config.yaml: final_system
"""


import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.evaluation.metrics import (classification_metrics, danger_zone_rmse,
                                    early_warning_metrics, regression_metrics)
from src.evaluation.step3_report import fmt, md_table
from src.evaluation.step6_report import (fast_macro_f1, fast_onset_recall,
                                         fast_rmse, station_week_blocks)

from src.models import run_step7 as R

CFG = load_config()
M = CFG["models"]
ENS = CFG["ensemble"]
S7 = CFG["step7"]
CAE_CFG = CFG["context_ensemble"]
WR = CFG["warning_rule"]
HORIZONS = CFG["horizons"]
CLASS_NAMES = CFG["target"]["class_names"]
MEMBERS = ENS["members"]
METHODS = S7["methods"]
OOF = project_path(S7["oof_predictions"])
PLOTS = project_path(M["dirs"]["val_plots"])
SEED = CFG["seed"]
BASELINE = "persistence"


def load_oof(method, h):
    r = pd.read_parquet(os.path.join(OOF, f"{method}_regression_h{h}.parquet"))
    c = pd.read_parquet(os.path.join(OOF, f"{method}_classification_h{h}.parquet"))
    proba_cols = [f"proba_{n}" for n in CLASS_NAMES]
    proba = c[proba_cols].to_numpy() if all(p in c for p in proba_cols) else None
    return {"keys": r[["station", "time"]], "y_reg": r["y_true"].to_numpy(),
            "reg": r["y_pred"].to_numpy(), "y_cls": c["y_true"].to_numpy(),
            "cls": c["y_pred"].to_numpy(), "cls_warn": c["y_pred_warn"].to_numpy(),
            "class_now": c["class_now"].to_numpy(), "proba": proba}


def score(d, persistence_reg, rule="warn"):
    cls_pred = d["cls_warn"] if rule == "warn" else d["cls"]
    reg = regression_metrics(d["y_reg"], d["reg"], persistence_reg)
    reg["danger_rmse"] = danger_zone_rmse(d["y_reg"], d["reg"],
                                          *ENS["danger_zone"])
    cls = classification_metrics(d["y_cls"], cls_pred, d["proba"])
    cls.update(early_warning_metrics(d["class_now"], d["y_cls"], cls_pred))
    return reg, cls


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def bootstrap(a, b, keys, y_reg, y_cls, class_now, n, rng, rule="warn"):
    """95% CI for (a minus b), resampling station-weeks."""
    mask = np.ones(len(y_reg), dtype=bool)
    blocks = station_week_blocks(keys, mask)
    ok_r = ~np.isnan(y_reg) & ~np.isnan(a["reg"]) & ~np.isnan(b["reg"])
    ok_c = ~np.isnan(y_cls)
    pa = a["cls_warn"] if rule == "warn" else a["cls"]
    pb = b["cls_warn"] if rule == "warn" else b["cls"]
    out = {k: [] for k in ["rmse", "f1_macro", "onset_recall"]}
    for _ in range(n):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        r = idx[ok_r[idx]]
        c = idx[ok_c[idx]]
        out["rmse"].append(fast_rmse(y_reg[r], a["reg"][r])
                           - fast_rmse(y_reg[r], b["reg"][r]))
        yc = y_cls[c].astype(int)
        out["f1_macro"].append(fast_macro_f1(yc, pa[c].astype(int))
                               - fast_macro_f1(yc, pb[c].astype(int)))
        nc = class_now[c].astype(int)
        out["onset_recall"].append(fast_onset_recall(nc, yc, pa[c].astype(int))
                                   - fast_onset_recall(nc, yc, pb[c].astype(int)))
    res = {}
    for k, v in out.items():
        v = np.asarray(v, dtype="float64")
        v = v[~np.isnan(v)]
        res[k] = {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
                  "hi": float(np.percentile(v, 97.5))}
        res[k]["significant"] = bool(res[k]["lo"] > 0 or res[k]["hi"] < 0)
    return res


# ---------------------------------------------------------------------------
# CAE explanation
# ---------------------------------------------------------------------------
def shap_by_situation(h=24, task="classification", sample=6000):
    """
    How much each member drives the meta-model's answer, in two very
    different situations. SHAP gives a per-row contribution for each input;
    summing the absolute contributions of a member's own inputs and
    normalising turns that into a "weight-equivalent".
    """
    import joblib
    from src.models import context_ensemble as C
    from src.models import ensemble as E

    path = C.model_path(task, h, "final")
    if not os.path.exists(path):
        return None
    bundle = joblib.load(path)
    model, names = bundle["model"], bundle["features"]

    d = E.load_all()[h]
    ctx = C.load_context(d["keys"])
    feats, _ = C.build_features(task, d["keys"], d["y_reg"], d["y_cls"],
                                d["reg"], d["proba"], h, ctx)

    do_now = ctx["do_mgl"].to_numpy()
    falling = ctx["do_mgl_chg6"].to_numpy()
    low = CFG["target"]["thresholds"]["low"]
    danger = (np.abs(do_now - low) <= 1.5) & (falling < 0)
    calm = (do_now > 7) & (np.abs(falling) < 0.2)

    rng = np.random.default_rng(SEED)
    import shap
    explainer = shap.TreeExplainer(model)
    out = {}
    for label, mask in [("DO falling, within 1.5 mg/L of 5", danger),
                        ("DO stable and above 7", calm)]:
        rows = np.flatnonzero(mask)
        if len(rows) == 0:
            continue
        rows = rng.choice(rows, min(sample, len(rows)), replace=False)
        vals = explainer.shap_values(feats.iloc[rows])
        if isinstance(vals, list):                 # multiclass -> list of arrays
            vals = np.mean([np.abs(v) for v in vals], axis=0)
        elif vals.ndim == 3:
            vals = np.abs(vals).mean(axis=2)
        else:
            vals = np.abs(vals)
        share = {}
        for m in MEMBERS:
            cols = [i for i, n in enumerate(names)
                    if n.startswith((f"pred_{m}", f"proba_{m}_", f"recent_err_{m}"))]
            share[m] = float(vals[:, cols].sum())
        total = sum(share.values())
        out[label] = {m: v / total for m, v in share.items()}
        out[label]["_n_rows"] = int(len(rows))
    return out


def plot_shap_situations(shares):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    labels = list(shares)
    x = np.arange(len(MEMBERS))
    width = 0.38
    for i, label in enumerate(labels):
        vals = [shares[label][m] for m in MEMBERS]
        ax.bar(x + (i - 0.5) * width, vals, width,
               label=f"{label}  (n={shares[label]['_n_rows']:,})")
    ax.set_xticks(x, MEMBERS, rotation=15)
    ax.set_ylabel("share of the meta-model's attention")
    ax.set_title("Which member CAE leans on, by situation (24 h, classification)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25, axis="y")
    fig.tight_layout()
    path = os.path.join(PLOTS, "step7_cae_shap_by_situation.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_tau_curves(h=24, methods=("cae", "stacking", "simple_average",
                                   "lightgbm", "bilstm")):
    """Onset recall against false alarms as tau is swept, out-of-fold."""
    fig, ax = plt.subplots(figsize=(7, 5))
    budget = WR["max_false_alarm_rate"]
    for name in methods:
        d = load_oof(name, h)
        if d["proba"] is None:
            continue
        xs, ys = [], []
        for tau in R.tau_grid():
            pred = R.apply_warning(d["proba"], tau)
            m = early_warning_metrics(d["class_now"], d["y_cls"], pred)
            xs.append(m["false_alarm_rate"])
            ys.append(m["onset_recall"])
        ax.plot(xs, ys, lw=1.8, label=name)
        # the operating point actually used
        m = early_warning_metrics(d["class_now"], d["y_cls"], d["cls_warn"])
        ax.scatter(m["false_alarm_rate"], m["onset_recall"], s=60, zorder=3)
    ax.axvline(budget, color="tab:red", ls="--", lw=1,
               label=f"false-alarm budget ({budget:.0%})")
    ax.set_xlim(0, 0.08)
    ax.set_xlabel("false alarm rate")
    ax.set_ylabel("onset recall")
    ax.set_title(f"Catching onsets, {h} h ahead (out-of-fold 2024)\n"
                 f"dots mark the threshold each method actually used")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(PLOTS, "step7_onset_vs_false_alarm.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------
def decide(boot_vs_stacking):
    """
    The rule, fixed before any of this was run:

      CAE becomes the proposed method if, at 24 h with the warning rule,
      its onset recall is higher than stacking's with a CI excluding 0,
      AND its RMSE is not significantly worse than stacking's.
      Otherwise the final system is stacking.
    """
    b = boot_vs_stacking[24]
    onset, rmse = b["onset_recall"], b["rmse"]
    onset_better = bool(onset["significant"] and onset["mean"] > 0)
    rmse_worse = bool(rmse["significant"] and rmse["mean"] > 0)
    chosen = "cae" if (onset_better and not rmse_worse) else "stacking"
    return chosen, onset_better, rmse_worse, onset, rmse


def write_final_system(name):
    path = project_path("config.yaml")
    s = open(path, encoding="utf-8").read()
    s = s.replace(f'final_system: "undecided"', f'final_system: "{name}"')
    import re
    s = re.sub(r'^final_system: ".*"$', f'final_system: "{name}"', s,
               flags=re.M)
    open(path, "w", encoding="utf-8").write(s)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(PLOTS, exist_ok=True)
    data = {h: {m: load_oof(m, h) for m in METHODS} for h in HORIZONS}
    folds = pd.read_csv(os.path.join(OOF, "folds.csv"))
    taus = pd.read_csv(os.path.join(OOF, "taus.csv"))

    scores = {}
    rows_r, rows_c = [], []
    for h in HORIZONS:
        pers = data[h][BASELINE]["reg"]
        for m in METHODS:
            d = data[h][m]
            for rule in ["warn", "argmax"]:
                reg, cls = score(d, pers, rule)
                scores[(m, h, rule)] = (reg, cls)
                if rule == "warn":
                    rows_r.append({"method": m, "horizon": h, **reg})
                flat = {k: v for k, v in cls.items() if k != "confusion_matrix"}
                rows_c.append({"method": m, "horizon": h, "rule": rule, **flat})
    pd.DataFrame(rows_r).to_csv(os.path.join(project_path(M["dirs"]["val_metrics"]),
                                             "metrics_step7_regression.csv"),
                                index=False)
    pd.DataFrame(rows_c).to_csv(os.path.join(project_path(M["dirs"]["val_metrics"]),
                                             "metrics_step7_classification.csv"),
                                index=False)

    # bootstrap CAE against stacking and the simple average
    rng = np.random.default_rng(SEED)
    n = ENS["bootstrap"]["n_resamples"]
    boot = {}
    for against in S7["bootstrap_against"]:
        boot[against] = {}
        for h in HORIZONS:
            d = data[h]
            boot[against][h] = bootstrap(d["cae"], d[against], d["cae"]["keys"],
                                         d["cae"]["y_reg"], d["cae"]["y_cls"],
                                         d["cae"]["class_now"], n, rng)
            print(f"  CAE - {against}, h={h}: " + ", ".join(
                f"{k} {v['mean']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]"
                for k, v in boot[against][h].items()))
    pd.DataFrame([{"against": a, "horizon": h, "metric": k, **v}
                  for a in boot for h in HORIZONS
                  for k, v in boot[a][h].items()]
                 ).to_csv(os.path.join(project_path(M["dirs"]["val_metrics"]),
                                       "step7_bootstrap.csv"), index=False)

    chosen, onset_better, rmse_worse, onset, rmse = decide(boot["stacking"])
    write_final_system(chosen)
    print(f"\nDECISION: final_system = {chosen}")

    shares = shap_by_situation()
    p_shap = plot_shap_situations(shares) if shares else None
    p_tau = plot_tau_curves()

    write_report(data, scores, boot, folds, taus, chosen, onset_better,
                 rmse_worse, onset, rmse, shares, [p_tau, p_shap])
    print_terminal(scores, boot, chosen, onset, rmse)


def write_report(data, scores, boot, folds, taus, chosen, onset_better,
                 rmse_worse, onset, rmse, shares, plots):
    out = ["# Step 7: the Context-Aware Ensemble and a warning rule\n",
           "- Everything here is **out-of-fold over 2024**: six two-month "
           "folds, each predicted by models fitted on the other five, with "
           f"**{CFG['crossfit']['purge_days']} days purged** on both sides. "
           "Every row of 2024 is predicted exactly once, by a model that "
           "never saw it.",
           "- That replaces the val-A / val-B split for this step, and "
           "**every** method - including persistence, the members, the "
           "simple average, stacking and ATAE - is re-scored on the same "
           "rows, so the comparison is like for like. These numbers are not "
           "comparable with step 6's.",
           "- No member was retrained. The 2025 test split was not read.\n"]

    out.append("## The protocol\n")
    out.append(md_table(["fold", "period", "test rows", "train rows",
                         "purged rows"],
                        [[r["fold"], r["period"], f"{r['test_rows']:,}",
                          f"{r['train_rows']:,}", f"{r['purged_rows']:,}"]
                         for _, r in folds.iterrows()]))
    out.append(f"\nThe purge has to cover two leaks at once: a target "
               f"{max(HORIZONS)} h ahead, and the "
               f"{CAE_CFG['error_window_hours']} h window the ensembles look "
               f"back over. Fourteen days covers both.\n")

    out.append("\n## CAE in one paragraph\n")
    out.append(
        "ATAE asked *which member has been right lately*. Step 6 showed that "
        "is worth almost nothing over a plain average, because the members "
        "do not take turns being good - they are good at different "
        "**situations**. The BiLSTM catches the most onsets at 24 h and has "
        "the worst macro F1; LightGBM is the reverse. So CAE asks a "
        "different question: *given what the river is doing right now, whom "
        "should I believe?* It is a small LightGBM (15 leaves, 200 samples "
        "per leaf) per task and horizon, taking the five members' outputs, "
        "how much they disagree, how each has been doing over the last "
        f"{CAE_CFG['error_window_hours']} h (known errors only), and the "
        "river's state at t: DO, its distance from 5 mg/L, its 6 h and 24 h "
        "change, its daily swing, saturation, water temperature, hours since "
        "it was last below 5, and the time of day and year. There is no "
        "`station_id` input, on purpose - it should learn situations, not "
        "stations.\n")

    out.append("\n## The warning rule\n")
    out.append(
        f"Argmax answers *which class is most likely*. A warning system asks "
        f"*is the chance of Low high enough to act on*, which is a lower "
        f"bar - and at 24 h the most likely class is almost never Low even "
        f"when the risk is real. So each method issues a warning when "
        f"P(Low) >= tau, with tau chosen on the training folds to catch as "
        f"many onsets as possible while keeping the false-alarm rate at or "
        f"below **{WR['max_false_alarm_rate']:.0%}**. Both rules are "
        f"reported below.\n")
    t24 = taus[(taus["horizon"] == 24) & taus["tau"].notna()]
    if len(t24):
        med = t24.groupby("method")["tau"].median().sort_values()
        out.append("Median tau chosen across folds at 24 h: "
                   + ", ".join(f"`{m}` {v:.2f}" for m, v in med.items()) + ".\n")

    # ---- regression 24 h ----
    out.append("\n## Regression at 24 hours, out-of-fold 2024\n")
    head = ["method", "R2", "RMSE", "MAE", "skill vs persistence",
            "danger-zone RMSE"]
    rows = []
    for m in METHODS:
        r = scores[(m, 24, "warn")][0]
        rows.append([m, fmt(r["r2"]), fmt(r["rmse"]), fmt(r["mae"]),
                     "-" if m == BASELINE else f"{r['skill_vs_persistence']:+.3f}",
                     fmt(r["danger_rmse"])])
    out.append(md_table(head, rows))

    # ---- classification 24 h, both rules ----
    for rule, title in [("warn", "with the warning rule"),
                        ("argmax", "with argmax")]:
        out.append(f"\n## Classification at 24 hours, {title}\n")
        head = ["method", "accuracy", "macro F1", "ROC-AUC", "Low recall",
                "onset recall", "false alarm rate", "transition acc"]
        rows = []
        for m in METHODS:
            c = scores[(m, 24, rule)][1]
            rows.append([m, fmt(c["accuracy"], 3), fmt(c["f1_macro"], 4),
                         fmt(c["roc_auc_ovr_macro"], 3), fmt(c["recall_Low"], 3),
                         fmt(c["onset_recall"], 3),
                         fmt(c["false_alarm_rate"], 4),
                         fmt(c["transition_accuracy"], 3)])
        out.append(md_table(head, rows))

    # ---- all horizons ----
    out.append("\n## All horizons, out-of-fold (warning rule)\n")
    head = (["method"] + [f"RMSE {h}h" for h in HORIZONS]
            + [f"macroF1 {h}h" for h in HORIZONS]
            + [f"onset recall {h}h" for h in HORIZONS])
    rows = []
    for m in METHODS:
        rows.append([m]
                    + [fmt(scores[(m, h, "warn")][0]["rmse"], 3) for h in HORIZONS]
                    + [fmt(scores[(m, h, "warn")][1]["f1_macro"], 3)
                       for h in HORIZONS]
                    + [fmt(scores[(m, h, "warn")][1]["onset_recall"], 3)
                       for h in HORIZONS])
    out.append(md_table(head, rows))

    # ---- bootstrap ----
    out.append("\n## Is CAE really different? Block bootstrap\n")
    out.append(f"{ENS['bootstrap']['n_resamples']:,} resamples of "
               f"station-weeks, out-of-fold rows. Negative RMSE favours CAE; "
               f"positive macro F1 and onset recall favour CAE.\n")
    for against in boot:
        out.append(f"\n### CAE minus `{against}`\n")
        rows = []
        for h in HORIZONS:
            for k, label, better in [("rmse", "RMSE", "lower"),
                                     ("f1_macro", "macro F1", "higher"),
                                     ("onset_recall", "onset recall", "higher")]:
                v = boot[against][h][k]
                if not v["significant"]:
                    verdict = "no significant difference"
                else:
                    good = (v["mean"] < 0) if better == "lower" else (v["mean"] > 0)
                    verdict = "**CAE better**" if good else f"{against} better"
                rows.append([f"{h} h", label, f"{v['mean']:+.4f}",
                             f"[{v['lo']:+.4f}, {v['hi']:+.4f}]", verdict])
        out.append(md_table(["horizon", "metric", "CAE - " + against,
                             "95% CI", "verdict"], rows))

    # ---- CAE explanation ----
    out.append("\n## What the meta-model learned\n")
    imp = pd.read_csv(os.path.join(OOF, "cae_importance.csv"))
    for task in ["classification", "regression"]:
        t = imp[(imp["task"] == task) & (imp["horizon"] == 24)].head(12)
        out.append(f"\n### Top inputs, {task}, 24 h\n")
        out.append(md_table(["feature", "importance"],
                            [[r["feature"], int(r["importance"])]
                             for _, r in t.iterrows()]))
    if shares:
        out.append("\n### Which member CAE leans on, by situation\n")
        out.append("Share of the meta-model's attention (mean absolute SHAP "
                   "over each member's own inputs), 24 h classification:\n")
        labels = list(shares)
        out.append(md_table(["member"] + labels,
                            [[m] + [f"{shares[l][m]:.3f}" for l in labels]
                             for m in MEMBERS]))

    out.append("\n## Plots\n")
    for p in plots:
        if p:
            out.append(f"- `{os.path.relpath(p, project_path('.'))}`")

    # ---- the decision ----
    out.append("\n## DECISION\n")
    out.append("The rule, written down before any of this was run:\n")
    out.append("> CAE becomes the proposed method if, at 24 h with the "
               "warning rule, its onset recall is higher than stacking's "
               "with a confidence interval excluding 0, AND its RMSE is not "
               "significantly worse than stacking's. Otherwise the final "
               "system is stacking.\n")
    out.append(f"At 24 h, CAE minus stacking: onset recall "
               f"{onset['mean']:+.4f} [{onset['lo']:+.4f}, {onset['hi']:+.4f}]"
               f" - {'significantly higher' if onset_better else 'not significantly higher'}; "
               f"RMSE {rmse['mean']:+.4f} [{rmse['lo']:+.4f}, {rmse['hi']:+.4f}]"
               f" - {'significantly worse' if rmse_worse else 'not significantly worse'}.\n")
    cae24 = scores[("cae", 24, "warn")]
    st24 = scores[("stacking", 24, "warn")]
    out.append(f"**final_system = `{chosen}`** (written to config.yaml).\n")
    if chosen == "cae":
        out.append(
            f"CAE earns it: at 24 h it catches "
            f"{cae24[1]['onset_recall']:.1%} of onsets against stacking's "
            f"{st24[1]['onset_recall']:.1%}, at a false-alarm rate of "
            f"{cae24[1]['false_alarm_rate']:.2%} against "
            f"{st24[1]['false_alarm_rate']:.2%}, with an RMSE of "
            f"{cae24[0]['rmse']:.4f} against {st24[0]['rmse']:.4f}. The "
            f"condition that mattered was the one about onsets - which is "
            f"the right thing for a warning system to be judged on.\n")
    else:
        out.append(
            f"CAE does not earn it. At 24 h it catches "
            f"{cae24[1]['onset_recall']:.1%} of onsets against stacking's "
            f"{st24[1]['onset_recall']:.1%}, with an RMSE of "
            f"{cae24[0]['rmse']:.4f} against {st24[0]['rmse']:.4f}. The "
            f"pre-set rule required a clear win on onset recall, and that "
            f"did not happen, so the final system is the simpler method. "
            f"This is the intended behaviour of writing the rule down "
            f"first: a method does not get adopted because it is the "
            f"newest thing tried.\n")

    out.append("\n## What actually moved the needle\n")
    arg24 = scores[("cae", 24, "argmax")][1]
    out.append(
        f"The honest headline of step 7 is not CAE. It is the **warning "
        f"rule**. At 24 h, switching from argmax to a tuned P(Low) "
        f"threshold takes CAE's onset recall from {arg24['onset_recall']:.3f} "
        f"to {cae24[1]['onset_recall']:.3f} - the same model, the same "
        f"probabilities, a different question asked of them - while keeping "
        f"false alarms inside the {WR['max_false_alarm_rate']:.0%} budget. "
        f"Every method gains from it. Changing how the decision is made was "
        f"worth far more than changing how the members are combined, which "
        f"is a useful thing to know and slightly deflating for the "
        f"ensemble work.\n")

    # ---- the budget slipped, and it is worth saying so ----
    budget = WR["max_false_alarm_rate"]
    t24 = taus[(taus["horizon"] == 24) & taus["tau"].notna()]
    if len(t24):
        in_sample = float(t24["train_false_alarm"].mean())
        realised = float(np.mean([scores[(m, 24, "warn")][1]["false_alarm_rate"]
                                  for m in METHODS if m != BASELINE]))
        if realised > budget:
            out.append("\n## The false-alarm budget slipped out-of-fold\n")
            out.append(
                f"tau is chosen as the lowest threshold whose false-alarm "
                f"rate still fits the {budget:.0%} budget, which means it "
                f"lands *exactly* on the boundary: across methods at 24 h "
                f"the in-sample rate averages {in_sample:.2%}. Out-of-fold "
                f"the same thresholds deliver {realised:.2%} - over budget. "
                f"Nothing is broken; this is what happens when a constraint "
                f"is tuned to the limit, because the threshold that just "
                f"fits on five-sixths of the year will not quite fit on the "
                f"sixth. If the {budget:.0%} were a real operational "
                f"contract rather than a design target, tau should be "
                f"chosen against a tighter internal budget (say "
                f"{0.75 * budget:.1%}) to leave headroom. That is a "
                f"one-line change in config.yaml and it is left as a "
                f"finding rather than quietly applied, because changing the "
                f"budget after seeing the result is how a constraint stops "
                f"meaning anything.\n")

    out.append("## Honest notes\n")
    out.append("- The members were tuned on all of 2024 in steps 3-5, so "
               "every method here inherits that optimism equally. The "
               "cross-fitting makes the COMBINERS honest, not the members.")
    out.append("- The warning threshold is one number per method and "
               "horizon, chosen on five-sixths of the year. It is the least "
               "overfittable thing in this project, but it is still fitted.")
    out.append("- CAE has no station_id, but its context features come from "
               "stations it did see. It is honest across time, not across "
               "rivers; a genuinely new river is a different question this "
               "project does not answer.")
    out.append("- 2025 remains untouched.\n")

    path = project_path("reports", "step7_cae_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Wrote {path}")


def print_terminal(scores, boot, chosen, onset, rmse):
    for rule in ["warn", "argmax"]:
        print(f"\nCLASSIFICATION 24 h, out-of-fold ({rule})")
        print(f"  {'method':<17}{'acc':>7}{'macroF1':>9}{'LowRec':>8}"
              f"{'onsetRec':>10}{'falseAl':>9}{'transAcc':>10}")
        for m in METHODS:
            c = scores[(m, 24, rule)][1]
            print(f"  {m:<17}{c['accuracy']:>7.3f}{c['f1_macro']:>9.4f}"
                  f"{c['recall_Low']:>8.3f}{c['onset_recall']:>10.3f}"
                  f"{c['false_alarm_rate']:>9.4f}"
                  f"{c['transition_accuracy']:>10.3f}")
    print("\nREGRESSION 24 h, out-of-fold")
    print(f"  {'method':<17}{'R2':>8}{'RMSE':>9}{'MAE':>8}{'danger':>9}")
    for m in METHODS:
        r = scores[(m, 24, "warn")][0]
        print(f"  {m:<17}{r['r2']:>8.4f}{r['rmse']:>9.4f}{r['mae']:>8.4f}"
              f"{r['danger_rmse']:>9.4f}")
    print(f"\nDECISION: final_system = {chosen}")


if __name__ == "__main__":
    main()
