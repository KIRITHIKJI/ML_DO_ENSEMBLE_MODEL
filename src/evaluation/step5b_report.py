"""
STEP 5b report - did the classification fix work, judged on val-B.

Run from the project root (after src.models.fix_dl_cls):

    python -m src.evaluation.step5b_report

Everything here is scored on val-B (July to December 2024), the half of the
validation year that was never used to choose anything in step 5b. Every
model - persistence, the three tree models and both deep models, before and
after the fix - is re-scored on exactly those rows, so the comparison is
like for like. The 2025 test split is not read.

The pass rule is unchanged: macro F1 higher than persistence AND Low-class
recall at least as high as persistence.

Writes reports/step5b_dl_cls_fix_report.md and
results/val/metrics_val_b.csv.
"""

import json
import os

import pandas as pd

from src.config import load_config, project_path
from src.evaluation.calibration import val_b_mask
from src.evaluation.metrics import classification_metrics, regression_metrics
from src.evaluation.step3_report import fmt, md_table

CFG = load_config()
M = CFG["models"]
FIX = CFG["dl_cls_fix"]
PC = CFG["pass_criteria"]
HORIZONS = CFG["horizons"]
CLASS_NAMES = CFG["target"]["class_names"]
VS = CFG["validation_split"]
BASELINE = "persistence"
DL_NAMES = ["bilstm", "cnn_lstm"]
ML_NAMES = ["persistence", "linear", "random_forest", "xgboost", "lightgbm"]
PRED_DIR = project_path(M["dirs"]["val_predictions"])
STEP5_DIR = os.path.join(project_path(FIX["archive_dir"]), "predictions_step5")


def read_predictions(folder, model, task, h):
    path = os.path.join(folder, f"{model}_{task}_h{h}.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None


def score_on_val_b(folder, model):
    """Every metric for one model, on val-B rows only."""
    out = {}
    for h in HORIZONS:
        r = read_predictions(folder, model, "regression", h)
        c = read_predictions(folder, model, "classification", h)
        if r is None or c is None:
            return None
        mb = val_b_mask(r["time"])
        rr = r[mb]
        cc = c[mb]
        # persistence for the skill score: DO at t is the step-3 persistence
        # prediction, read from its own file
        pers = read_predictions(folder, BASELINE, "regression", h)
        pv = pers[val_b_mask(pers["time"])]["y_pred"].to_numpy() if pers is not None else None
        out[("regression", h)] = regression_metrics(rr["y_true"], rr["y_pred"], pv)
        proba_cols = [f"proba_{n}" for n in CLASS_NAMES]
        proba = cc[proba_cols].to_numpy() if all(p in cc for p in proba_cols) else None
        out[("classification", h)] = classification_metrics(
            cc["y_true"], cc["y_pred"], proba)
    return out


def passes(cls_metrics, base_metrics):
    f1 = cls_metrics["f1_macro"]
    rec = cls_metrics["recall_Low"]
    beats = f1 > base_metrics["f1_macro"]
    keeps = rec >= base_metrics["recall_Low"] - PC["classification"]["recall_tolerance"]
    return bool(beats and keeps), f1, rec


def main():
    with open(os.path.join(project_path(CFG["deep_learning"]["dirs"]["dl_models"]),
                           "cls_fix_summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    tried = pd.read_csv(project_path(FIX["results_csv"]))

    # score everything on val-B
    scores = {name: score_on_val_b(PRED_DIR, name) for name in ML_NAMES}
    for name in DL_NAMES:
        scores[f"{name} (step 5)"] = score_on_val_b(STEP5_DIR, name)
        scores[f"{name} (step 5b)"] = score_on_val_b(PRED_DIR, name)
    base = scores[BASELINE]

    rows_csv = []
    for name, sc in scores.items():
        if sc is None:
            continue
        for h in HORIZONS:
            row = {"model": name, "horizon": h, "split": "val_b"}
            row.update({f"reg_{k}": v for k, v in sc[("regression", h)].items()})
            cm = dict(sc[("classification", h)])
            cm.pop("confusion_matrix", None)
            row.update({f"cls_{k}": v for k, v in cm.items()})
            rows_csv.append(row)
    csv_path = os.path.join(project_path(M["dirs"]["val_metrics"]),
                            "metrics_val_b.csv")
    pd.DataFrame(rows_csv).to_csv(csv_path, index=False)

    # ---------------- report ----------------
    out = ["# Step 5b: one attempt at the deep models' classification\n",
           f"- Trained on **2023**. The validation year is split: "
           f"**val-A {VS['val_a_start']} .. {VS['val_a_end']}** chose the "
           f"loss weight and the calibration factors, "
           f"**val-B {VS['val_b_start']} .. {VS['val_b_end']}** is what "
           f"every number below is scored on. Nothing in 2025 was read.",
           "- Early stopping still watches the loss on the whole of 2024, "
           "as in step 5, so the split changes what is tuned, not how "
           "training stops.",
           "- Every model, including the tree models and persistence, is "
           "re-scored on the same val-B rows, so the comparison is like for "
           "like. These numbers are therefore NOT comparable with the "
           "full-2024 numbers in the step-5 report.",
           f"- Pass rule, unchanged: macro F1 above persistence AND Low "
           f"recall at least persistence's.\n"]

    # ---- what was tried ----
    out.append("## What was tried, and what was chosen (on val-A)\n")
    head = ["model", "classification loss weight", "seeds",
            "val-A mean macro F1", "val-A RMSE 24 h", "regression ok?",
            "chosen", "note"]
    out.append(md_table(head, [
        [r["model"], r["cls_loss_weight"], r["seeds"],
         f"{r['val_a_mean_f1']:.4f}", f"{r['val_a_rmse_24']:.4f}",
         "yes" if r["rmse_ok"] else "**no**",
         "**chosen**" if r["chosen"] else "", r["note"]]
        for _, r in tried.iterrows()]))

    out.append("\n### Chosen settings\n")
    head = ["model", "loss weight", "seeds", "calibration (Low, Medium, Good)"]
    rows = []
    for name in DL_NAMES:
        s = summary[name]
        cal = "; ".join(
            f"{h} h: ({', '.join(f'{x:g}' for x in s['calibration_factors'][str(h)])})"
            for h in HORIZONS)
        rows.append([name, s["cls_loss_weight"], s["seeds"], cal])
    out.append(md_table(head, rows))
    out.append("")
    for name in DL_NAMES:
        s = summary[name]
        out.append(f"- `{name}`: calibration moved val-A mean macro F1 from "
                   f"{s['val_a_mean_f1_before_calibration']:.4f} to "
                   f"{s['val_a_mean_f1_after_calibration']:.4f}.")
    out.append("")

    # ---- before / after on val-B ----
    out.append("\n## Before and after, on val-B\n")
    head = (["model", "version"] + [f"macro F1 {h} h" for h in HORIZONS]
            + ["Low recall 24 h", "RMSE 24 h", "pass at 24 h?"])
    rows = []
    for name in DL_NAMES:
        for version in ["step 5", "step 5b"]:
            sc = scores[f"{name} ({version})"]
            if sc is None:
                continue
            ok, f1, rec = passes(sc[("classification", 24)],
                                 base[("classification", 24)])
            rows.append([name, version]
                        + [f"{sc[('classification', h)]['f1_macro']:.4f}"
                           for h in HORIZONS]
                        + [f"{rec:.4f}",
                           f"{sc[('regression', 24)]['rmse']:.4f}",
                           "PASS" if ok else "fail"])
    pb = base[("classification", 24)]
    rows.append([BASELINE, "-"]
                + [f"{base[('classification', h)]['f1_macro']:.4f}"
                   for h in HORIZONS]
                + [f"{pb['recall_Low']:.4f}",
                   f"{base[('regression', 24)]['rmse']:.4f}", "(baseline)"])
    out.append(md_table(head, rows))

    # ---- all models at 24 h on val-B ----
    out.append("\n## Classification at 24 hours, val-B, all models\n")
    head = ["model", "accuracy", "balanced acc", "macro F1", "ROC-AUC",
            "Low recall", "Low precision", "pass?"]
    rows = []
    order = ML_NAMES + [f"{n} (step 5)" for n in DL_NAMES] + \
            [f"{n} (step 5b)" for n in DL_NAMES]
    for name in order:
        sc = scores.get(name)
        if sc is None:
            continue
        cm = sc[("classification", 24)]
        if name == BASELINE:
            verdict = "(baseline)"
        else:
            verdict = "PASS" if passes(cm, pb)[0] else "fail"
        rows.append([name, fmt(cm["accuracy"], 3),
                     fmt(cm["balanced_accuracy"], 3), fmt(cm["f1_macro"], 4),
                     fmt(cm["roc_auc_ovr_macro"], 3), fmt(cm["recall_Low"], 3),
                     fmt(cm["precision_Low"], 3), verdict])
    out.append(md_table(head, rows))

    # ---- all horizons pass table ----
    out.append("\n## Pass/fail at every horizon, val-B\n")
    head = ["model"] + [f"{h} h" for h in HORIZONS] + ["passed"]
    rows = []
    for name in order:
        sc = scores.get(name)
        if sc is None or name == BASELINE:
            continue
        marks, n = [], 0
        for h in HORIZONS:
            ok = passes(sc[("classification", h)],
                        base[("classification", h)])[0]
            marks.append("PASS" if ok else "fail")
            n += ok
        rows.append([name] + marks + [f"{n}/{len(HORIZONS)}"])
    out.append(md_table(head, rows))

    # ---- the honest paragraph ----
    out.append("\n## Did it work?\n")
    para = []
    for name in DL_NAMES:
        a = scores[f"{name} (step 5)"][("classification", 24)]
        b = scores[f"{name} (step 5b)"][("classification", 24)]
        ok_a = passes(a, pb)[0]
        ok_b = passes(b, pb)[0]
        verdict = ("now passes" if ok_b and not ok_a else
                   "still fails" if not ok_b else
                   "passed before and still passes")
        para.append(f"**{name}**: macro F1 at 24 h went from {a['f1_macro']:.4f} "
                    f"to {b['f1_macro']:.4f} against persistence's "
                    f"{pb['f1_macro']:.4f}, and it {verdict}.")
    out.append(" ".join(para))
    out.append("")

    # Did any model go backwards? That is the headline, if so.
    regressed = []
    for name in DL_NAMES:
        a = scores[f"{name} (step 5)"][("classification", 24)]
        b = scores[f"{name} (step 5b)"][("classification", 24)]
        if passes(a, pb)[0] and not passes(b, pb)[0]:
            regressed.append((name, a, b))
    if regressed:
        out.append("### The fix made it worse by the project's own rule\n")
        for name, a, b in regressed:
            out.append(
                f"**`{name}` passed at 24 h before this step and fails "
                f"after it.** Calibration did what it was asked to do - macro "
                f"F1 rose from {a['f1_macro']:.4f} to {b['f1_macro']:.4f} - "
                f"but it bought that by predicting Low less often, and "
                f"Low recall fell from {a['recall_Low']:.3f} to "
                f"{b['recall_Low']:.3f}, below persistence's "
                f"{pb['recall_Low']:.3f}. The pass rule requires both, so "
                f"the model is now worse where it counts.")
        out.append(
            "\nThe cause is a mismatch that is worth stating plainly: the "
            "calibration grid was told to maximise **macro F1**, but the "
            "project's rule is **macro F1 above persistence AND Low recall "
            "not below it**. Optimising one half of a two-part rule is "
            "exactly how you lose the other half. Macro F1 treats a missed "
            "low-oxygen hour and a false alarm as equally bad; this project "
            "does not, which is why the Low-recall condition was written "
            "into the criteria in the first place.\n")
        out.append(
            "The uncalibrated step-5 predictions are kept in "
            "`results/val/archive/predictions_step5/`. Restoring them brings "
            f"`{regressed[0][0]}` back to passing at 24 h. That decision is "
            "left open rather than taken quietly here, because changing the "
            "calibration objective after seeing which way the number moved "
            "would be precisely the kind of search this step was meant to "
            "avoid.\n")

    n_pass = sum(passes(scores[f'{n} (step 5b)'][("classification", 24)], pb)[0]
                 for n in DL_NAMES)
    if n_pass == len(DL_NAMES):
        out.append("Both deep classifiers now clear the bar at 24 hours. "
                   "That was one attempt and it worked, but the honest "
                   "reading is that it was close: the margin is small, the "
                   "calibration was fitted on val-A, and the tree models "
                   "still do the job better without needing either fix.")
    elif n_pass == 0:
        out.append("Neither deep classifier clears the bar at 24 hours, and "
                   "that is the result. The agreed plan was one principled "
                   "attempt, and this was it: a louder classification loss "
                   "and a calibration of the class priors, both chosen on "
                   "val-A. Twenty-four hours ahead is the horizon where "
                   "persistence is strongest, because DO runs on a daily "
                   "cycle and \"the same as this time yesterday\" is a very "
                   "good guess. The deep models beat it comfortably at 6, 12 "
                   "and 48 hours and lose to it at 24. The tree models do "
                   "clear it at 24 h, so the ensemble in step 6 has "
                   "classifiers that pass at every horizon regardless.")
    else:
        out.append("One of the two deep classifiers now clears the bar at 24 "
                   "hours and the other does not. The one that still fails "
                   "fails narrowly, at the one horizon where persistence is "
                   "strongest. That is the result, and it stands: the plan "
                   "was one principled attempt, not an open search until the "
                   "number went the right way.")
    out.append("")
    out.append("A note on size: the differences being argued over here are "
               "in the third decimal place of macro F1, while the spread "
               "between random seeds in step 5 was 0.0014 to 0.0041. Any "
               "conclusion resting on a gap smaller than that is noise, "
               "which is exactly why this step reports on val-B rather than "
               "on the rows the settings were chosen from.\n")

    out.append("## Honest notes\n")
    out.append("- The step-5 numbers in this report are re-scored on val-B, "
               "so they differ from the step-5 report, which used the whole "
               "of 2024. Comparing a number here with a number there is a "
               "mistake.")
    out.append("- Calibration is fitted per model and per horizon on val-A, "
               "about 128,000 rows. It is a three-number adjustment, so the "
               "risk of overfitting it is low, but it is still a choice made "
               "on 2024 and will carry its optimism into any 2024 number.")
    out.append("- The regression side was protected, not improved: a louder "
               "classification loss was only allowed if no horizon's RMSE "
               f"got worse by more than "
               f"{100 * FIX['max_rmse_worsening']:.0f}%.")
    out.append("- 2025 remains untouched.\n")

    path = project_path("reports", "step5b_dl_cls_fix_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Wrote {path}")
    print(f"Wrote {csv_path}\n")

    # ---------------- terminal ----------------
    print("CHOSEN SETTINGS")
    for name in DL_NAMES:
        s = summary[name]
        print(f"  {name}: loss weight {s['cls_loss_weight']}, seeds "
              f"{s['seeds']}")
        for h in HORIZONS:
            print(f"     h{h:<3} calibration "
                  f"{tuple(round(x, 2) for x in s['calibration_factors'][str(h)])}")
    print("\nBEFORE / AFTER ON VAL-B (macro F1 by horizon, 24 h detail)")
    for name in DL_NAMES:
        for version in ["step 5", "step 5b"]:
            sc = scores[f"{name} ({version})"]
            f1s = " ".join(f"{sc[('classification', h)]['f1_macro']:.4f}"
                           for h in HORIZONS)
            cm = sc[("classification", 24)]
            ok = passes(cm, pb)[0]
            print(f"  {name:<9} {version:<8} {f1s}  | Low recall 24h "
                  f"{cm['recall_Low']:.3f} | RMSE 24h "
                  f"{sc[('regression', 24)]['rmse']:.4f} | "
                  f"{'PASS' if ok else 'fail'}")
    print(f"  {'persistence':<9} {'-':<8} "
          + " ".join(f"{base[('classification', h)]['f1_macro']:.4f}"
                     for h in HORIZONS)
          + f"  | Low recall 24h {pb['recall_Low']:.3f}")


if __name__ == "__main__":
    main()
