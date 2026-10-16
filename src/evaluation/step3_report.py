"""
STEP 3.5 - Compare every step-3 model on the validation year and report.

Run from the project root (after baselines.py and train_ml.py):

    python -m src.evaluation.step3_report

Everything here is measured on VALIDATION (2024). The 2025 test split is not
read at all.

Applies the pass criteria that were written into config.yaml BEFORE any
results were seen:

  regression      skill_vs_persistence > 0, i.e. a lower RMSE than the naive
                  "DO will stay as it is now" forecast
  classification  a higher macro F1 than persistence AND a Low-class recall
                  at least as high as persistence - a model that scores
                  better overall by catching fewer low-oxygen hours is not
                  an improvement for an early-warning system
  a model goes forward to the ensemble if it passes at >= 3 of the 4 horizons

Writes:
    reports/step3_ml_report.md
    results/val/pass_table.csv
    results/val/plots/confusion_<model>_h24.png
    results/val/plots/rmse_vs_horizon.png
"""

import os

import matplotlib
matplotlib.use("Agg")           # no screen needed, just write png files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config, project_path

CFG = load_config()
P = CFG["paths"]
M = CFG["models"]
PC = CFG["pass_criteria"]
HORIZONS = CFG["horizons"]
CLASS_NAMES = CFG["target"]["class_names"]
BASELINE = "persistence"
MODEL_ORDER = ["persistence", "linear", "random_forest", "xgboost",
               "lightgbm", "bilstm", "cnn_lstm"]
# This report is about step 3 only. Step 4 adds the deep models to the same
# metrics files, so they are filtered out here and shown in step4_ml_report.
STEP3_MODELS = ["persistence", "linear", "random_forest", "xgboost", "lightgbm"]


def metrics_path(task):
    return os.path.join(project_path(M["dirs"]["val_metrics"]),
                        f"metrics_{task}.csv")


def load_metrics(task, keep=None):
    df = pd.read_csv(metrics_path(task))
    if keep is not None:
        df = df[df["model"].isin(keep)]
    df["order"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return df.sort_values(["horizon", "order"]).drop(columns="order")


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def fmt(x, nd=4):
    return "-" if pd.isna(x) else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# Pass criteria
# ---------------------------------------------------------------------------
def pass_flags(reg, cls):
    """
    One row per model / task / horizon with a True/False and the reason.
    The baseline itself is marked "-": it is the bar, not a candidate.
    """
    rows = []
    for h in HORIZONS:
        r_h = reg[reg["horizon"] == h].set_index("model")
        c_h = cls[cls["horizon"] == h].set_index("model")

        for model in r_h.index:
            if model == BASELINE:
                continue
            skill = r_h.loc[model, "skill_vs_persistence"]
            ok = bool(skill > PC["regression"]["min_skill_vs_persistence"])
            rows.append({"model": model, "task": "regression", "horizon": h,
                         "passed": ok,
                         "detail": f"skill {skill:+.3f}"})

        if BASELINE in c_h.index:
            base_f1 = c_h.loc[BASELINE, PC["classification"]["beat_persistence_on"]]
            base_rec = c_h.loc[BASELINE, PC["classification"]["not_worse_on"]]
            for model in c_h.index:
                if model == BASELINE:
                    continue
                f1 = c_h.loc[model, PC["classification"]["beat_persistence_on"]]
                rec = c_h.loc[model, PC["classification"]["not_worse_on"]]
                beats = bool(f1 > base_f1)
                keeps = bool(rec >= base_rec - PC["classification"]["recall_tolerance"])
                why = []
                if not beats:
                    why.append(f"macroF1 {f1:.3f} <= persistence {base_f1:.3f}")
                if not keeps:
                    why.append(f"Low recall {rec:.3f} < persistence {base_rec:.3f}")
                rows.append({"model": model, "task": "classification",
                             "horizon": h, "passed": beats and keeps,
                             "detail": "; ".join(why) if why else
                                       f"macroF1 {f1:.3f}, Low recall {rec:.3f}"})
    return pd.DataFrame(rows)


def goes_forward(flags):
    """Which model/task combinations pass at enough horizons."""
    need = PC["min_horizons_passed"]
    g = (flags.groupby(["model", "task"])["passed"].sum().astype(int)
         .reset_index().rename(columns={"passed": "horizons_passed"}))
    g["of"] = len(HORIZONS)
    g["goes_forward"] = g["horizons_passed"] >= need
    return g.sort_values(["task", "horizons_passed"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_confusions(cls, h=24):
    """One confusion-matrix picture per classifier, at the given horizon."""
    folder = project_path(M["dirs"]["val_plots"])
    os.makedirs(folder, exist_ok=True)
    made = []
    for _, row in cls[cls["horizon"] == h].iterrows():
        if pd.isna(row.get("confusion_matrix")):
            continue
        cm = np.array([int(x) for x in str(row["confusion_matrix"]).split(";")])
        cm = cm.reshape(len(CLASS_NAMES), len(CLASS_NAMES))
        share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:,}\n{100 * share[i, j]:.1f}%",
                        ha="center", va="center", fontsize=9,
                        color="white" if share[i, j] > .5 else "black")
        ax.set_xticks(range(3), CLASS_NAMES)
        ax.set_yticks(range(3), CLASS_NAMES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{row['model']} - {h} h ahead\nvalidation 2024 "
                     f"(rows add up to 100%)", fontsize=10)
        fig.tight_layout()
        path = os.path.join(folder, f"confusion_{row['model']}_h{h}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        made.append(path)
    return made


def plot_rmse_vs_horizon(reg):
    """RMSE against forecast horizon, one line per model."""
    folder = project_path(M["dirs"]["val_plots"])
    os.makedirs(folder, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for model in MODEL_ORDER:
        d = reg[reg["model"] == model].sort_values("horizon")
        if not len(d):
            continue
        style = "--o" if model == BASELINE else "-o"
        ax.plot(d["horizon"], d["rmse"], style, label=model, linewidth=2,
                markersize=5)
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("forecast horizon (hours ahead)")
    ax.set_ylabel("RMSE of DO (mg/L)   lower is better")
    ax.set_title("Validation 2024: error against horizon")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(folder, "rmse_vs_horizon.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    reg = load_metrics("regression", STEP3_MODELS)
    cls = load_metrics("classification", STEP3_MODELS)
    flags = pass_flags(reg, cls)
    forward = goes_forward(flags)

    pass_csv = os.path.join(project_path(M["dirs"]["val_metrics"]),
                            "pass_table.csv")
    flags.to_csv(pass_csv, index=False)
    confusions = plot_confusions(cls, h=24)
    rmse_plot = plot_rmse_vs_horizon(reg)

    passed_at = {(r.model, r.task, r.horizon): r.passed
                 for r in flags.itertuples()}

    def mark(model, task, h):
        if model == BASELINE:
            return "(baseline)"
        return "PASS" if passed_at.get((model, task, h)) else "fail"

    # ---------------- the report ----------------
    out = ["# Step 3: baselines and ML models (validation only)\n",
           "- Trained on **2023**, scored on **2024**. The 2025 test split "
           "was not read by any script in this step.",
           f"- Horizons: {HORIZONS} hours ahead. Classes: "
           f"{CLASS_NAMES[0]} < {CFG['target']['thresholds']['low']} mg/L, "
           f"{CLASS_NAMES[1]} "
           f"{CFG['target']['thresholds']['low']}-{CFG['target']['thresholds']['good']}, "
           f"{CLASS_NAMES[2]} > {CFG['target']['thresholds']['good']}.",
           f"- Seed {CFG['seed']} everywhere. Settings are in config.yaml "
           f"under `models:`; they are starting values, not tuned.",
           "",
           "## Pass criteria (fixed in config.yaml before any results were seen)\n",
           f"- **Regression:** `skill_vs_persistence > "
           f"{PC['regression']['min_skill_vs_persistence']}`, i.e. a lower "
           f"RMSE than the naive \"DO stays as it is\" forecast.",
           f"- **Classification:** macro F1 **higher** than persistence AND "
           f"Low-class recall **at least as high** as persistence. Catching "
           f"the low-oxygen hours is the point of the system, so a better "
           f"overall score with fewer Low hours caught does not count.",
           f"- A model goes forward to the ensemble if it passes at "
           f"**{PC['min_horizons_passed']} of {len(HORIZONS)}** horizons.\n"]

    # ---- regression, 24 h ----
    out.append("## Regression, 24 hours ahead\n")
    head = ["model", "R2", "RMSE", "MAE", "NSE", "skill vs persistence", "pass?"]
    rows = []
    for _, r in reg[reg["horizon"] == 24].iterrows():
        rows.append([r["model"], fmt(r["r2"]), fmt(r["rmse"]), fmt(r["mae"]),
                     fmt(r["nse"]),
                     "-" if r["model"] == BASELINE else f"{r['skill_vs_persistence']:+.3f}",
                     mark(r["model"], "regression", 24)])
    out.append(md_table(head, rows))

    # ---- classification, 24 h ----
    out.append("\n## Classification, 24 hours ahead\n")
    head = ["model", "accuracy", "balanced acc", "macro F1", "ROC-AUC",
            "Low recall", "Low precision", "pass?"]
    rows = []
    for _, r in cls[cls["horizon"] == 24].iterrows():
        rows.append([r["model"], fmt(r["accuracy"], 3),
                     fmt(r["balanced_accuracy"], 3), fmt(r["f1_macro"], 3),
                     fmt(r["roc_auc_ovr_macro"], 3), fmt(r["recall_Low"], 3),
                     fmt(r["precision_Low"], 3),
                     mark(r["model"], "classification", 24)])
    out.append(md_table(head, rows))

    # ---- compact, all horizons ----
    out.append("\n## All horizons, compact\n")
    out.append("### Regression: RMSE (mg/L) and skill vs persistence\n")
    head = ["model"] + [f"{h} h RMSE" for h in HORIZONS] + \
           [f"{h} h skill" for h in HORIZONS] + ["passed"]
    rows = []
    for model in MODEL_ORDER:
        d = reg[reg["model"] == model].set_index("horizon")
        if not len(d):
            continue
        n_pass = int(flags[(flags["model"] == model) &
                           (flags["task"] == "regression")]["passed"].sum())
        rows.append([model]
                    + [fmt(d.loc[h, "rmse"], 3) if h in d.index else "-"
                       for h in HORIZONS]
                    + [("-" if model == BASELINE else
                        f"{d.loc[h, 'skill_vs_persistence']:+.3f}")
                       if h in d.index else "-" for h in HORIZONS]
                    + ["(baseline)" if model == BASELINE
                       else f"{n_pass}/{len(HORIZONS)}"])
    out.append(md_table(head, rows))

    out.append("\n### Classification: macro F1 and Low recall\n")
    head = ["model"] + [f"{h} h macroF1" for h in HORIZONS] + \
           [f"{h} h Low recall" for h in HORIZONS] + ["passed"]
    rows = []
    for model in MODEL_ORDER:
        d = cls[cls["model"] == model].set_index("horizon")
        if not len(d):
            continue
        n_pass = int(flags[(flags["model"] == model) &
                           (flags["task"] == "classification")]["passed"].sum())
        rows.append([model]
                    + [fmt(d.loc[h, "f1_macro"], 3) if h in d.index else "-"
                       for h in HORIZONS]
                    + [fmt(d.loc[h, "recall_Low"], 3) if h in d.index else "-"
                       for h in HORIZONS]
                    + ["(baseline)" if model == BASELINE
                       else f"{n_pass}/{len(HORIZONS)}"])
    out.append(md_table(head, rows))

    # ---- who goes forward ----
    out.append("\n## Who goes forward to the ensemble\n")
    out.append(md_table(["model", "task", "horizons passed", "goes forward?"],
                        [[r.model, r.task, f"{r.horizons_passed}/{r.of}",
                          "YES" if r.goes_forward else "no"]
                         for r in forward.itertuples()]))

    failures = flags[~flags["passed"]]
    out.append("\n### Where models fail, and why\n")
    if len(failures):
        out.append(md_table(["model", "task", "horizon", "reason"],
                            [[r.model, r.task, f"{r.horizon} h", r.detail]
                             for r in failures.itertuples()]))
    else:
        out.append("Nothing failed.\n")

    # ---- training time ----
    out.append("\n## Training time (seconds, one fit per horizon)\n")
    both = pd.concat([reg.assign(task="regression"),
                      cls.assign(task="classification")])
    piv = (both[both["model"] != BASELINE]
           .pivot_table(index="model", columns=["task", "horizon"],
                        values="train_seconds", aggfunc="first"))
    rows = [[m] + [fmt(piv.loc[m, (task, h)], 1) if (task, h) in piv.columns else "-"
                   for task in ["regression", "classification"] for h in HORIZONS]
            + [fmt(piv.loc[m].sum() / 60, 1)]
            for m in piv.index]
    head = (["model"] + [f"reg {h}h" for h in HORIZONS]
            + [f"cls {h}h" for h in HORIZONS] + ["total (min)"])
    out.append(md_table(head, rows))

    # ---- plots and files ----
    out.append("\n## Plots\n")
    out.append(f"- `{os.path.relpath(rmse_plot, project_path('.'))}` - RMSE "
               f"against horizon, all models")
    for c in confusions:
        out.append(f"- `{os.path.relpath(c, project_path('.'))}`")

    out.append("\n## Honest notes\n")
    out.append("- **Persistence is strong at 24 h and weak at 6 h.** DO has a "
               "daily cycle, so \"the same as now\" is a good guess exactly "
               "one day ahead and a poor one six hours ahead. That is why the "
               "skill numbers jump about between horizons, and why beating "
               "persistence at 24 h is the hardest test here.")
    out.append("- **XGBoost and LightGBM pick their number of trees by "
               "watching the 2024 validation set** (early stopping, "
               f"{M['early_stopping_rounds']} rounds). That is a legitimate "
               "use of validation, but it makes their validation scores "
               "slightly optimistic next to the Random Forest, which never "
               "looks at 2024. Only the 2025 test set settles it.")
    out.append("- Feature importances are each library's own measure "
               "(impurity decrease for the forest, gain for the boosters), so "
               "they are comparable within a model and not across models.")
    out.append("- Nothing here was tuned. These are the starting settings "
               "from config.yaml.\n")

    path = project_path("reports", "step3_ml_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Wrote {path}")
    print(f"Wrote {pass_csv}")
    print(f"Wrote {rmse_plot} and {len(confusions)} confusion plots\n")

    # ---------------- terminal ----------------
    print("REGRESSION, 24 h")
    print(reg[reg["horizon"] == 24][["model", "r2", "rmse", "mae", "nse",
                                     "skill_vs_persistence"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCLASSIFICATION, 24 h")
    print(cls[cls["horizon"] == 24][["model", "accuracy", "balanced_accuracy",
                                     "f1_macro", "roc_auc_ovr_macro",
                                     "recall_Low", "precision_Low"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nGOES FORWARD TO THE ENSEMBLE")
    print(forward.to_string(index=False))


if __name__ == "__main__":
    main()
