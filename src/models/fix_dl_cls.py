"""
STEP 5b - one principled attempt to fix the deep models' classification.

Run from the project root:

    python -m src.models.fix_dl_cls

Step 5 left both deep models failing the classification criterion at 24 h by
a hair. Two changes are tried here, and then the result is accepted either
way:

  1. Make the classification part of the loss louder (weight 2.0 or 3.0
     instead of 1.0), so the network spends less of its capacity on the
     regression head.
  2. Calibrate the class probabilities: multiply them by one factor per
     class, renormalise, take the argmax. A model trained with balanced
     class weights over-predicts the rare class; this moves the boundary
     back without retraining.

To avoid choosing and reporting on the same rows, 2024 is split in half:
val-A (January to June) decides the loss weight and the calibration
factors, val-B (July to December) is only read when reporting. Early
stopping still watches the loss on the whole of 2024, exactly as in step 5,
so what changed is what gets tuned, not how training stops.

The 2025 test split is never loaded.

Writes:
    results/val/dl_cls_fix.csv              what was tried and chosen
    results/val/predictions/<model>_<task>_h<h>.parquet   (calibrated)
    results/val/archive/predictions_step5/  the step-5 versions
    models/dl/<model>_clsfix_seed<k>.pt
    models/dl/cls_fix_summary.json
"""

import json
import os
import shutil
import time

import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.evaluation.calibration import (apply_calibration, fit_calibration,
                                        val_a_mask, val_b_mask)
from src.evaluation.metrics import classification_metrics, regression_metrics
from src.models import common as C
from src.models import sequences as S
from src.models import train_dl as T
from src.models.dl_models import MODELS

CFG = load_config()
DL = CFG["deep_learning"]
FIX = CFG["dl_cls_fix"]
HORIZONS = CFG["horizons"]
CLASS_NAMES = CFG["target"]["class_names"]
DL_NAMES = list(MODELS)


# ---------------------------------------------------------------------------
# The step-5 model, read back from its saved predictions
# ---------------------------------------------------------------------------
def load_step5_predictions(name):
    """
    The step-5 model's validation predictions: DO per horizon and class
    probabilities per horizon, aligned on (station, time).
    """
    folder = project_path(C.M["dirs"]["val_predictions"])
    reg, prob, keys = {}, {}, None
    for h in HORIZONS:
        r = pd.read_parquet(os.path.join(folder, f"{name}_regression_h{h}.parquet"))
        c = pd.read_parquet(os.path.join(folder,
                                         f"{name}_classification_h{h}.parquet"))
        if keys is None:
            keys = r[["station", "time"]].copy()
        reg[h] = r["y_pred"].to_numpy()
        prob[h] = c[[f"proba_{n}" for n in CLASS_NAMES]].to_numpy()
    do_pred = np.column_stack([reg[h] for h in HORIZONS])
    cls_prob = np.stack([prob[h] for h in HORIZONS], axis=1)
    return keys, do_pred, cls_prob


def archive_step5_predictions():
    """Keep the step-5 prediction files before they are overwritten."""
    src = project_path(C.M["dirs"]["val_predictions"])
    dst = os.path.join(project_path(FIX["archive_dir"]), "predictions_step5")
    os.makedirs(dst, exist_ok=True)
    saved = 0
    for name in DL_NAMES:
        for task in ["regression", "classification"]:
            for h in HORIZONS:
                f = f"{name}_{task}_h{h}.parquet"
                target = os.path.join(dst, f)
                if os.path.exists(os.path.join(src, f)) and not os.path.exists(target):
                    shutil.copy(os.path.join(src, f), target)
                    saved += 1
    return dst, saved


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def mean_macro_f1(meta, cls_prob, mask):
    """Macro F1 averaged over the four horizons, on the given rows."""
    scores = []
    for j, h in enumerate(HORIZONS):
        m = classification_metrics(meta.loc[mask, f"y_cls_{h}"],
                                   cls_prob[mask, j, :].argmax(1))
        scores.append(m["f1_macro"])
    return float(np.mean(scores)), scores


def rmse_per_horizon(meta, do_pred, mask):
    return [regression_metrics(meta.loc[mask, f"y_reg_{h}"],
                               do_pred[mask, j])["rmse"]
            for j, h in enumerate(HORIZONS)]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    device = T.pick_device()
    T.describe_device(device)
    base = FIX["base_variant"]
    settings0 = T.variant_settings(base)
    print(f"\nBase architecture: {base} (chosen in step 5)")
    print(f"val-A {CFG['validation_split']['val_a_start']} .. "
          f"{CFG['validation_split']['val_a_end']}  (choose)")
    print(f"val-B {CFG['validation_split']['val_b_start']} .. "
          f"{CFG['validation_split']['val_b_end']}  (report)\n")

    train_data, val_data, do_scale = T.build_data(settings0, save=False)
    reg_tr, cls_tr = T.build_targets(train_data, do_scale)
    reg_va, cls_va = T.build_targets(val_data, do_scale)
    meta = val_data.meta
    in_a = val_a_mask(meta["time"])
    in_b = val_b_mask(meta["time"])
    print(f"  val-A {in_a.sum():,} rows | val-B {in_b.sum():,} rows\n")

    archive_dir, n_archived = archive_step5_predictions()
    print(f"Archived {n_archived} step-5 prediction files to {archive_dir}\n")

    rows, summary = [], {}
    for name in DL_NAMES:
        print(f"=== {name} ===")
        # --- the step-5 model (weight 1.0), read back from disk ---
        keys, do5, prob5 = load_step5_predictions(name)
        assert (keys["time"].to_numpy() == meta["time"].to_numpy()).all(), \
            "the saved predictions do not line up with the sequence rows"
        f1_a_5, _ = mean_macro_f1(meta, prob5, in_a)
        rmse_a_5 = rmse_per_horizon(meta, do5, in_a)
        print(f"  weight 1.0 (step 5): val-A mean macro F1 {f1_a_5:.4f}")
        rows.append({"model": name, "cls_loss_weight": 1.0, "seeds": "42,43,44",
                     "val_a_mean_f1": round(f1_a_5, 5),
                     "val_a_rmse_24": round(rmse_a_5[HORIZONS.index(24)], 5),
                     "rmse_ok": True, "chosen": False, "note": "step 5 model"})

        # --- the candidates ---
        candidates = {1.0: (do5, prob5)}
        for w in FIX["cls_loss_weights"]:
            settings = dict(settings0)
            settings["classification_loss_weight"] = w
            print(f"  training with classification loss weight {w} (seed 42)")
            t0 = time.perf_counter()
            model, hist, best, batch_size, loss_fn = T.train_model(
                name, settings, train_data, val_data, reg_tr, cls_tr, reg_va,
                cls_va, do_scale, device, CFG["seed"], verbose=False)
            _, do_w, prob_w = T.predict(model, val_data, reg_va, cls_va,
                                        loss_fn, device, batch_size, do_scale)
            del model
            f1_a, _ = mean_macro_f1(meta, prob_w, in_a)
            rmse_a = rmse_per_horizon(meta, do_w, in_a)
            # the regression side must not pay for it
            worst = max((new - old) / old for new, old in zip(rmse_a, rmse_a_5))
            ok = worst <= FIX["max_rmse_worsening"]
            print(f"    val-A mean macro F1 {f1_a:.4f} "
                  f"({f1_a - f1_a_5:+.4f}) | worst RMSE change "
                  f"{100 * worst:+.1f}% -> regression {'ok' if ok else 'HURT'}"
                  f" | {(time.perf_counter() - t0) / 60:.1f} min")
            rows.append({"model": name, "cls_loss_weight": w, "seeds": "42",
                         "val_a_mean_f1": round(f1_a, 5),
                         "val_a_rmse_24": round(rmse_a[HORIZONS.index(24)], 5),
                         "rmse_ok": bool(ok), "chosen": False,
                         "note": f"worst RMSE change {100 * worst:+.1f}%"})
            if ok:
                candidates[w] = (do_w, prob_w)

        # --- pick the weight on val-A ---
        scored = {w: mean_macro_f1(meta, p, in_a)[0]
                  for w, (d, p) in candidates.items()}
        best_w = max(scored, key=scored.get)
        print(f"  chosen classification loss weight: {best_w} "
              f"(val-A mean macro F1 {scored[best_w]:.4f})")
        for r in rows:
            if r["model"] == name and r["cls_loss_weight"] == best_w:
                r["chosen"] = True

        # --- if it is not 1.0, retrain it with three seeds and average ---
        if best_w == 1.0:
            do_pred, cls_prob = do5, prob5
            seeds_used = "42,43,44 (from step 5)"
            print("  keeping the step-5 model; no retraining needed")
        else:
            settings = dict(settings0)
            settings["classification_loss_weight"] = best_w
            preds, probs = [], []
            for seed in FIX["seeds"]:
                print(f"  retraining weight {best_w}, seed {seed}")
                model, hist, best, batch_size, loss_fn = T.train_model(
                    name, settings, train_data, val_data, reg_tr, cls_tr,
                    reg_va, cls_va, do_scale, device, seed, verbose=False)
                _, d, p = T.predict(model, val_data, reg_va, cls_va, loss_fn,
                                    device, batch_size, do_scale)
                preds.append(d)
                probs.append(p)
                import torch
                folder = project_path(DL["dirs"]["dl_models"])
                torch.save({"state_dict": model.state_dict(),
                            "channels": S.CHANNELS,
                            "seq_len": settings["sequence_length"],
                            "do_scale": do_scale, "variant": base,
                            "cls_loss_weight": best_w, "seed": seed,
                            "arch": T.architecture_overrides(settings, name)},
                           os.path.join(folder, f"{name}_clsfix_seed{seed}.pt"))
                del model
            do_pred = np.mean(preds, axis=0)
            cls_prob = np.mean(probs, axis=0)
            seeds_used = ",".join(map(str, FIX["seeds"]))

        # --- calibration, fitted on val-A only, per horizon ---
        factors, cal_before, cal_after = {}, [], []
        calibrated = cls_prob.copy()
        for j, h in enumerate(HORIZONS):
            f, after, before = fit_calibration(meta.loc[in_a, f"y_cls_{h}"],
                                               cls_prob[in_a, j, :])
            factors[h] = f
            cal_before.append(before)
            cal_after.append(after)
            calibrated[:, j, :] = apply_calibration(cls_prob[:, j, :], f)
            print(f"  h{h:<3} calibration {tuple(round(x, 2) for x in f)} "
                  f"-> val-A macro F1 {before:.4f} to {after:.4f}")

        # --- save the final predictions over the whole of 2024 ---
        for j, h in enumerate(HORIZONS):
            C.save_predictions(name, "regression", h, meta, do_pred[:, j])
            proba = calibrated[:, j, :]
            C.save_predictions(name, "classification", h, meta,
                               proba.argmax(1), proba)

        summary[name] = {
            "base_variant": base, "cls_loss_weight": best_w,
            "seeds": seeds_used,
            "calibration_factors": {str(h): list(factors[h]) for h in HORIZONS},
            "val_a_mean_f1_before_calibration": round(float(np.mean(cal_before)), 5),
            "val_a_mean_f1_after_calibration": round(float(np.mean(cal_after)), 5),
        }
        f1_b, per_h = mean_macro_f1(meta, calibrated, in_b)
        print(f"  val-B mean macro F1 after everything: {f1_b:.4f} "
              f"(per horizon {[round(x, 4) for x in per_h]})\n")

    path = project_path(FIX["results_csv"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    with open(os.path.join(project_path(DL["dirs"]["dl_models"]),
                           "cls_fix_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {path}")
    print("Next: python -m src.evaluation.step5b_report")


if __name__ == "__main__":
    main()
