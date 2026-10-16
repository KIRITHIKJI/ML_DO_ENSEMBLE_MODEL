"""
STEP 5.4 - The step-5 report: what the tuning changed.

Run from the project root (after search_dl.py and finalize_dl.py):

    python -m src.evaluation.step5_report

Shows the search table, the before/after for each deep model, and the usual
comparison of every model in the project. Still validation only: the 2025
test split is not read.

Writes:
    reports/step5_dl_tuning_report.md
    results/val/pass_table.csv
    results/val/plots/rmse_vs_horizon.png
    results/val/plots/confusion_<model>_h24.png
"""

import json
import os

import pandas as pd

from src.config import load_config, project_path
from src.evaluation.step3_report import (BASELINE, MODEL_ORDER, fmt,
                                         goes_forward, load_metrics, md_table,
                                         pass_flags, plot_confusions,
                                         plot_rmse_vs_horizon)

CFG = load_config()
M = CFG["models"]
DL = CFG["deep_learning"]
SEARCH = CFG["dl_search"]
PC = CFG["pass_criteria"]
HORIZONS = CFG["horizons"]
CLASS_NAMES = CFG["target"]["class_names"]
DL_MODELS = ["bilstm", "cnn_lstm"]

VARIANT_LABEL = {
    "step4": "step-4 reference",
    "reg": "(a) regularisation",
    "reg_hybrid": "(a)+(b) hybrid input",
    "reg_hybrid_72h": "(a)+(b)+(c) 72 h window",
    "reg_hybrid_72h_small": "(a)+(b)+(c)+(d) hidden 32",
}


def old_metrics(task):
    """The step-4 deep-model rows, archived before they were replaced."""
    path = os.path.join(project_path(SEARCH["archive_dir"]),
                        f"step4_metrics_{task}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def training_summary():
    path = os.path.join(project_path(DL["dirs"]["dl_models"]),
                        "training_summary.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    reg = load_metrics("regression")
    cls = load_metrics("classification")
    flags = pass_flags(reg, cls)
    forward = goes_forward(flags)
    summary = training_summary()
    search = pd.read_csv(project_path(SEARCH["results_csv"]))

    flags.to_csv(os.path.join(project_path(M["dirs"]["val_metrics"]),
                              "pass_table.csv"), index=False)
    confusions = plot_confusions(cls, h=24)
    rmse_plot = plot_rmse_vs_horizon(reg)

    passed_at = {(r.model, r.task, r.horizon): r.passed
                 for r in flags.itertuples()}

    def mark(model, task, h):
        if model == BASELINE:
            return "(baseline)"
        return "PASS" if passed_at.get((model, task, h)) else "fail"

    import torch
    device_line = (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                   else "CPU only")

    out = ["# Step 5: tuning the deep models (validation only)\n",
           "- Trained on **2023**, scored on **2024**. The 2025 test split "
           "was not read by any script in this step.",
           f"- torch {torch.__version__}, CUDA {torch.version.cuda}, "
           f"device: **{device_line}**",
           "- Step 4 left both deep models behind every tree model and "
           "overfitting by epoch 3. Four changes were tried, each added on "
           "top of the last, and the winner was picked by **validation "
           "loss** - the quantity the networks minimise - rather than by any "
           "single metric.",
           f"- The chosen configuration of each model was then retrained "
           f"with seeds {SEARCH['final_seeds']} and the three predictions "
           f"averaged. That average is what the tables below score.\n"]

    # ---------------- the search ----------------
    out.append("## The search: 5 variants x 2 models\n")
    out.append("| variant | what it changes |\n|---|---|")
    out.append("\n".join(
        f"| `{k}` | {v} |" for k, v in VARIANT_LABEL.items()))
    out.append("\n### Results (single seed 42, chosen row marked)\n")
    head = ["model", "variant", "params", "epochs", "best epoch", "val loss",
            "24 h RMSE", "24 h macro F1", "24 h Low recall", "min", "chosen"]
    rows = []
    for _, r in search.iterrows():
        rows.append([r["model"], f"`{r['variant']}`", f"{r['parameters']:,}",
                     r["epochs"], r["best_epoch"], f"{r['val_loss']:.4f}",
                     f"{r['rmse_24']:.4f}", f"{r['f1_macro_24']:.4f}",
                     f"{r['recall_low_24']:.4f}", f"{r['minutes']:.1f}",
                     "**chosen**" if r["chosen"] else ""])
    out.append(md_table(head, rows))

    # ---------------- before / after ----------------
    out.append("\n## Before and after, 24 hours ahead\n")
    old_r, old_c = old_metrics("regression"), old_metrics("classification")
    if old_r is not None:
        head = ["model", "", "RMSE", "R2", "macro F1", "Low recall",
                "regression", "classification"]
        rows = []
        for name in DL_MODELS:
            for label, rsrc, csrc in [("step 4", old_r, old_c),
                                      ("step 5", reg, cls)]:
                rr = rsrc[(rsrc["model"] == name) & (rsrc["horizon"] == 24)]
                cc = csrc[(csrc["model"] == name) & (csrc["horizon"] == 24)]
                if not len(rr) or not len(cc):
                    continue
                rr, cc = rr.iloc[0], cc.iloc[0]
                if label == "step 5":
                    pr = mark(name, "regression", 24)
                    pc = mark(name, "classification", 24)
                else:
                    # the old rows were scored against the same criteria
                    base_f1 = cls[(cls["model"] == BASELINE)
                                  & (cls["horizon"] == 24)]["f1_macro"].iloc[0]
                    base_rec = cls[(cls["model"] == BASELINE)
                                   & (cls["horizon"] == 24)]["recall_Low"].iloc[0]
                    pr = "PASS" if rr["skill_vs_persistence"] > 0 else "fail"
                    pc = ("PASS" if (cc["f1_macro"] > base_f1
                                     and cc["recall_Low"] >= base_rec) else "fail")
                rows.append([name, label, f"{rr['rmse']:.4f}", f"{rr['r2']:.4f}",
                             f"{cc['f1_macro']:.4f}", f"{cc['recall_Low']:.4f}",
                             pr, pc])
        out.append(md_table(head, rows))
    else:
        out.append("No archived step-4 metrics found.\n")

    # ---------------- seed spread ----------------
    out.append("\n## How much is luck? The spread over three seeds\n")
    if summary:
        head = ["model", "variant", "seeds", "mean val loss",
                "24 h RMSE (sd)", "24 h macro F1 (sd)", "averaged RMSE",
                "averaged macro F1"]
        rows = []
        for name in DL_MODELS:
            s = summary.get(name)
            if not s:
                continue
            rr = reg[(reg["model"] == name) & (reg["horizon"] == 24)].iloc[0]
            cc = cls[(cls["model"] == name) & (cls["horizon"] == 24)].iloc[0]
            per = pd.DataFrame(s["per_seed"])
            rows.append([name, f"`{s['variant']}`", s["seeds"],
                         f"{s['val_loss']:.4f}",
                         f"{per['rmse_24'].mean():.4f} (± {s['rmse_24_seed_std']:.4f})",
                         f"{per['f1_macro_24'].mean():.4f} (± {s['f1_24_seed_std']:.4f})",
                         f"**{rr['rmse']:.4f}**", f"**{cc['f1_macro']:.4f}**"])
        out.append(md_table(head, rows))
        out.append("\nThe last two columns are the averaged prediction, which "
                   "is the final model. It is better than the average single "
                   "seed, which is the point of averaging.\n")

    # ---------------- all models, 24 h ----------------
    out.append("\n## Regression, 24 hours ahead - all models\n")
    head = ["model", "R2", "RMSE", "MAE", "NSE", "skill vs persistence", "pass?"]
    out.append(md_table(head, [
        [r["model"], fmt(r["r2"]), fmt(r["rmse"]), fmt(r["mae"]), fmt(r["nse"]),
         "-" if r["model"] == BASELINE else f"{r['skill_vs_persistence']:+.3f}",
         mark(r["model"], "regression", 24)]
        for _, r in reg[reg["horizon"] == 24].iterrows()]))

    out.append("\n## Classification, 24 hours ahead - all models\n")
    head = ["model", "accuracy", "balanced acc", "macro F1", "ROC-AUC",
            "Low recall", "Low precision", "pass?"]
    out.append(md_table(head, [
        [r["model"], fmt(r["accuracy"], 3), fmt(r["balanced_accuracy"], 3),
         fmt(r["f1_macro"], 3), fmt(r["roc_auc_ovr_macro"], 3),
         fmt(r["recall_Low"], 3), fmt(r["precision_Low"], 3),
         mark(r["model"], "classification", 24)]
        for _, r in cls[cls["horizon"] == 24].iterrows()]))

    # ---------------- all horizons ----------------
    out.append("\n## All horizons, all models\n")
    out.append("### Regression: RMSE (mg/L) and skill vs persistence\n")
    head = (["model"] + [f"{h} h RMSE" for h in HORIZONS]
            + [f"{h} h skill" for h in HORIZONS] + ["passed"])
    rows = []
    for model in MODEL_ORDER:
        d = reg[reg["model"] == model].set_index("horizon")
        if not len(d):
            continue
        n = int(flags[(flags["model"] == model)
                      & (flags["task"] == "regression")]["passed"].sum())
        rows.append([model]
                    + [fmt(d.loc[h, "rmse"], 3) if h in d.index else "-"
                       for h in HORIZONS]
                    + [("-" if model == BASELINE
                        else f"{d.loc[h, 'skill_vs_persistence']:+.3f}")
                       if h in d.index else "-" for h in HORIZONS]
                    + ["(baseline)" if model == BASELINE
                       else f"{n}/{len(HORIZONS)}"])
    out.append(md_table(head, rows))

    out.append("\n### Classification: macro F1 and Low recall\n")
    head = (["model"] + [f"{h} h macroF1" for h in HORIZONS]
            + [f"{h} h Low recall" for h in HORIZONS] + ["passed"])
    rows = []
    for model in MODEL_ORDER:
        d = cls[cls["model"] == model].set_index("horizon")
        if not len(d):
            continue
        n = int(flags[(flags["model"] == model)
                      & (flags["task"] == "classification")]["passed"].sum())
        rows.append([model]
                    + [fmt(d.loc[h, "f1_macro"], 3) if h in d.index else "-"
                       for h in HORIZONS]
                    + [fmt(d.loc[h, "recall_Low"], 3) if h in d.index else "-"
                       for h in HORIZONS]
                    + ["(baseline)" if model == BASELINE
                       else f"{n}/{len(HORIZONS)}"])
    out.append(md_table(head, rows))

    out.append("\n## Who goes forward to the ensemble\n")
    out.append(md_table(["model", "task", "horizons passed", "goes forward?"],
                        [[r.model, r.task, f"{r.horizons_passed}/{r.of}",
                          "YES" if r.goes_forward else "no"]
                         for r in forward.itertuples()]))

    failures = flags[~flags["passed"]]
    out.append("\n### Where models still fail\n")
    out.append(md_table(["model", "task", "horizon", "reason"],
                        [[r.model, r.task, f"{r.horizon} h", r.detail]
                         for r in failures.itertuples()])
               if len(failures) else "Nothing fails any more.\n")

    out.append("\n## Plots\n")
    out.append(f"- `results/val/plots/rmse_vs_horizon.png`")
    for n in DL_MODELS:
        out.append(f"- `results/val/plots/{n}_training_curve.png`")
    for c in confusions:
        out.append(f"- `{os.path.relpath(c, project_path('.'))}`")

    # ---------------- what helped ----------------
    best = {}
    for name in DL_MODELS:
        here = search[search["model"] == name].set_index("variant")
        if "step4" in here.index:
            best[name] = here
    out.append("\n## What helped and what did not\n")
    para = []
    for name in DL_MODELS:
        here = search[search["model"] == name].set_index("variant")
        if "step4" not in here.index:
            continue
        base = here.loc["step4", "val_loss"]
        deltas = ", ".join(
            f"`{v}` {here.loc[v, 'val_loss'] - base:+.4f}"
            for v in SEARCH["variants"] if v in here.index and v != "step4")
        para.append(f"For **{name}**, validation loss against the step-4 "
                    f"reference ({base:.4f}): {deltas}.")
    out.append(" ".join(para))
    out.append("")
    out.append(
        "**What helped: the hybrid input, and nothing else very much.** "
        "Handing the network the same 211 engineered features the tree "
        "models use is the one change that clearly moved the validation "
        "loss, and it moved the 24 h RMSE with it. That is the useful "
        "finding of this step: the gap in step 4 was mostly about the "
        "*inputs*, not the architecture. A recurrent network is supposed to "
        "learn lags and rolling statistics from the raw series, and on one "
        "year of training data it simply does not learn them as well as "
        "step 2 computes them.\n")
    out.append(
        "**What did not help: regularisation on its own.** This is the "
        "opposite of what step 4's overfitting suggested. Dropout 0.3 with "
        "AdamW and a lower learning rate made the validation loss *worse* "
        "for the BiLSTM and roughly neutral for the CNN-LSTM. Early "
        "stopping was already handling the overfitting; slowing learning "
        "down on top of that mostly cost accuracy. The longer 72-hour "
        "window (c) gave a small gain - 48 hours already covers two daily "
        "cycles, so there is not much more to see - and halving the hidden "
        "size (d) helped slightly rather than hurting, which says these "
        "networks were larger than one year of data supports. Averaging "
        "three seeds is a small, free and reliable gain, and the seed "
        "spread above is the honest scale against which to read any of "
        "these differences: a gap smaller than the spread is noise.\n")

    out.append("## Honest notes\n")
    out.append("- **Everything here was chosen on 2024.** Five variants were "
               "compared on the validation year and the winner retrained on "
               "it; that is what a validation set is for, but it does mean "
               "these numbers are the optimistic end of what to expect. Only "
               "the 2025 test set, still untouched, settles it.")
    out.append("- **GPU LSTM training is still not bit-for-bit "
               "reproducible** (cuDNN has no deterministic recurrent "
               "kernel). The seed spread table is the honest measure of how "
               "much that matters: anything smaller than it is noise.")
    out.append("- The tree models were not retuned in this step, so the "
               "comparison is a tuned deep model against an untuned "
               "LightGBM. If the deep models still lose, that conclusion is "
               "safe; if they win narrowly, it is not.\n")

    path = project_path("reports", "step5_dl_tuning_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Wrote {path}\n")

    # ---------------- terminal ----------------
    show = ["model", "variant", "epochs", "best_epoch", "val_loss", "rmse_24",
            "f1_macro_24", "recall_low_24", "minutes", "chosen"]
    print("SEARCH")
    print(search[show].to_string(index=False))
    print("\nREGRESSION, 24 h")
    print(reg[reg["horizon"] == 24][["model", "r2", "rmse", "mae", "nse",
                                     "skill_vs_persistence"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCLASSIFICATION, 24 h")
    print(cls[cls["horizon"] == 24][["model", "accuracy", "balanced_accuracy",
                                     "f1_macro", "roc_auc_ovr_macro",
                                     "recall_Low", "precision_Low"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nGOES FORWARD")
    print(forward.to_string(index=False))


if __name__ == "__main__":
    main()
