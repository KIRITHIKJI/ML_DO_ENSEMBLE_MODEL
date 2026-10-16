"""
STEP 4.5 - The step-4 report: every model so far, side by side.

Run from the project root (after src.models.train_dl):

    python -m src.evaluation.step4_report

Reuses the tables, pass criteria and plots from step3_report, but over ALL
models now - persistence, linear, Random Forest, XGBoost, LightGBM, BiLSTM
and CNN-LSTM - and adds a table of how the two deep models actually trained
(epochs, best epoch, time per epoch, peak GPU memory).

Still validation only: the 2025 test split is not read.

Writes:
    reports/step4_dl_report.md
    results/val/pass_table.csv              (now including the deep models)
    results/val/plots/rmse_vs_horizon.png   (redrawn with every model)
    results/val/plots/confusion_<model>_h24.png
"""

import json
import os

import pandas as pd

from src.config import load_config, project_path
from src.evaluation.step3_report import (fmt, goes_forward, load_metrics,
                                         md_table, pass_flags,
                                         plot_confusions, plot_rmse_vs_horizon,
                                         BASELINE, MODEL_ORDER)

CFG = load_config()
M = CFG["models"]
DL = CFG["deep_learning"]
PC = CFG["pass_criteria"]
HORIZONS = CFG["horizons"]
CLASS_NAMES = CFG["target"]["class_names"]
DL_MODELS = ["bilstm", "cnn_lstm"]


def training_summary():
    """How the deep models trained, written by train_dl.py."""
    path = os.path.join(project_path(DL["dirs"]["dl_models"]),
                        "training_summary.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    reg = load_metrics("regression")           # no filter: every model
    cls = load_metrics("classification")
    flags = pass_flags(reg, cls)
    forward = goes_forward(flags)
    summary = training_summary()

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

    import torch
    device_line = (f"{torch.cuda.get_device_name(0)}"
                   if torch.cuda.is_available() else "CPU only")

    out = ["# Step 4: BiLSTM and CNN-LSTM (validation only)\n",
           "- Trained on **2023**, scored on **2024**. The 2025 test split "
           "was not read by any script in this step.",
           f"- torch {torch.__version__}, CUDA {torch.version.cuda}, "
           f"device: **{device_line}**",
           f"- Both networks are multi-task: one model produces all "
           f"{len(HORIZONS)} horizons for both tasks at once.",
           "- The regression head predicts the **change** in DO in scaled "
           "units; DO(t) is added back before scoring, so the numbers below "
           "are directly comparable with step 3.",
           "- Input is the raw hourly history (last 48 h of "
           f"{len(DL['water_vars'])} water variables, their missing flags, "
           "DO saturation, 8 weather and 8 air variables, time sin/cos), not "
           "the 211 engineered features the tree models use.",
           "- Rows are exactly the rows of `data/processed/val.parquet`, so "
           "every model in the tables is scored on the same hours.\n"]

    # ---- how the deep models trained ----
    out.append("## How the deep models trained\n")
    if summary:
        rows = []
        for name in DL_MODELS:
            if name not in summary:
                continue
            s = summary[name]
            total = s["epochs"] * s["seconds_per_epoch"]
            rows.append([name, s["epochs"], s["best_epoch"],
                         f"{s['seconds_per_epoch'] / 60:.1f}",
                         f"{total / 60:.1f}", f"{s['peak_gpu_gb']:.2f}"])
        out.append(md_table(["model", "epochs run", "best epoch",
                             "min / epoch", "total min", "peak GPU GB"], rows))
        out.append("\nTraining curves: "
                   + ", ".join(f"`results/val/plots/{n}_training_curve.png`"
                               for n in DL_MODELS if n in summary) + "\n")
    else:
        out.append("No training summary found - run "
                   "`python -m src.models.train_dl` first.\n")

    # ---- regression 24 h ----
    out.append("## Regression, 24 hours ahead - all models\n")
    head = ["model", "R2", "RMSE", "MAE", "NSE", "skill vs persistence", "pass?"]
    rows = []
    for _, r in reg[reg["horizon"] == 24].iterrows():
        rows.append([r["model"], fmt(r["r2"]), fmt(r["rmse"]), fmt(r["mae"]),
                     fmt(r["nse"]),
                     "-" if r["model"] == BASELINE
                     else f"{r['skill_vs_persistence']:+.3f}",
                     mark(r["model"], "regression", 24)])
    out.append(md_table(head, rows))

    # ---- classification 24 h ----
    out.append("\n## Classification, 24 hours ahead - all models\n")
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

    # ---- compact over all horizons ----
    out.append("\n## All horizons, all models\n")
    out.append("### Regression: RMSE (mg/L) and skill vs persistence\n")
    head = (["model"] + [f"{h} h RMSE" for h in HORIZONS]
            + [f"{h} h skill" for h in HORIZONS] + ["passed"])
    rows = []
    for model in MODEL_ORDER:
        d = reg[reg["model"] == model].set_index("horizon")
        if not len(d):
            continue
        n_pass = int(flags[(flags["model"] == model)
                           & (flags["task"] == "regression")]["passed"].sum())
        rows.append([model]
                    + [fmt(d.loc[h, "rmse"], 3) if h in d.index else "-"
                       for h in HORIZONS]
                    + [("-" if model == BASELINE
                        else f"{d.loc[h, 'skill_vs_persistence']:+.3f}")
                       if h in d.index else "-" for h in HORIZONS]
                    + ["(baseline)" if model == BASELINE
                       else f"{n_pass}/{len(HORIZONS)}"])
    out.append(md_table(head, rows))

    out.append("\n### Classification: macro F1 and Low recall\n")
    head = (["model"] + [f"{h} h macroF1" for h in HORIZONS]
            + [f"{h} h Low recall" for h in HORIZONS] + ["passed"])
    rows = []
    for model in MODEL_ORDER:
        d = cls[cls["model"] == model].set_index("horizon")
        if not len(d):
            continue
        n_pass = int(flags[(flags["model"] == model)
                           & (flags["task"] == "classification")]["passed"].sum())
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
    out.append(f"A model needs to pass at {PC['min_horizons_passed']} of "
               f"{len(HORIZONS)} horizons.\n")
    out.append(md_table(["model", "task", "horizons passed", "goes forward?"],
                        [[r.model, r.task, f"{r.horizons_passed}/{r.of}",
                          "YES" if r.goes_forward else "no"]
                         for r in forward.itertuples()]))

    failures = flags[~flags["passed"]]
    out.append("\n### Where models fail, and why\n")
    out.append(md_table(["model", "task", "horizon", "reason"],
                        [[r.model, r.task, f"{r.horizon} h", r.detail]
                         for r in failures.itertuples()])
               if len(failures) else "Nothing failed.\n")

    # ---- honest notes ----
    out.append("\n## Honest notes\n")
    out.append("- **GPU LSTM training is not bit-for-bit reproducible.** All "
               "the seeds are fixed (python, numpy, torch, cuda) and "
               "`cudnn.benchmark` is off, but cuDNN's recurrent kernels have "
               "no deterministic implementation, so "
               "`torch.use_deterministic_algorithms(True, warn_only=True)` "
               "warns rather than enforces. Two runs of the same command can "
               "differ in the last decimal. Everything else in the project "
               "is exactly reproducible.")
    out.append("- **The deep models watch 2024 too.** Early stopping and the "
               "learning-rate schedule both read the validation loss, the "
               "same arrangement as XGBoost and LightGBM in step 3. The "
               "Random Forest is the only model that never sees 2024, so it "
               "is the most conservative number in these tables.")
    out.append("- The deep models see raw hourly history, the tree models see "
               "211 engineered features. That is the comparison on purpose: "
               "it asks whether a network can learn from the raw series what "
               "the feature engineering spells out.")
    out.append("- Nothing here was tuned. These are the starting settings in "
               "config.yaml.\n")

    path = project_path("reports", "step4_dl_report.md")
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
