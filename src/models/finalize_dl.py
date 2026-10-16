"""
STEP 5.3 - Retrain the chosen configuration with three seeds and average.

Run from the project root (after src.models.search_dl):

    python -m src.models.finalize_dl

For each deep model, this reads results/val/dl_search.csv, takes the variant
with the lowest validation loss, and trains it again with seeds 42, 43 and
44. The final prediction is the AVERAGE of the three:

    regression      the mean predicted DO
    classification  the mean class probabilities, then argmax

Averaging a few seeds is the cheapest reliable improvement there is for a
neural network, and it also tells us how much of the step-4 result was luck:
the per-seed spread is reported next to the average.

The old step-4 metric rows are copied to results/val/archive/ first, then
replaced, so the report can show before and after.

The 2025 test split is never loaded.
"""

import json
import os
import shutil
import time

import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.evaluation.metrics import classification_metrics, regression_metrics
from src.models import common as C
from src.models import sequences as S
from src.models import train_dl as T
from src.models.baselines import metric_row
from src.models.dl_models import MODELS

CFG = load_config()
DL = CFG["deep_learning"]
SEARCH = CFG["dl_search"]
HORIZONS = CFG["horizons"]
DL_NAMES = list(MODELS)


def archive_old_metrics():
    """Keep the step-4 numbers so the report can show before and after."""
    folder = project_path(SEARCH["archive_dir"])
    os.makedirs(folder, exist_ok=True)
    saved = []
    for task in ["regression", "classification"]:
        src = os.path.join(project_path(CFG["models"]["dirs"]["val_metrics"]),
                           f"metrics_{task}.csv")
        dst = os.path.join(folder, f"step4_metrics_{task}.csv")
        if os.path.exists(src) and not os.path.exists(dst):
            # only the deep-model rows: the tree models are unchanged
            df = pd.read_csv(src)
            df[df["model"].isin(DL_NAMES)].to_csv(dst, index=False)
            saved.append(dst)
    return saved


def chosen_variants():
    """The winning variant per model, from the search table."""
    path = project_path(SEARCH["results_csv"])
    if not os.path.exists(path):
        raise SystemExit("run `python -m src.models.search_dl` first")
    df = pd.read_csv(path)
    out = {}
    for name in DL_NAMES:
        here = df[df["model"] == name]
        if not len(here):
            continue
        win = here[here["chosen"]] if here["chosen"].any() else \
            here.loc[[here[SEARCH["select_by"]].idxmin()]]
        out[name] = win["variant"].iloc[0]
    return out, df


def main():
    device = T.pick_device()
    T.describe_device(device)
    archived = archive_old_metrics()
    for a in archived:
        print(f"  archived the step-4 rows to {a}")

    picks, search = chosen_variants()
    seeds = SEARCH["final_seeds"]
    print(f"\nChosen configurations: {picks}")
    print(f"Retraining each with seeds {seeds} and averaging\n")

    all_rows = {"regression": [], "classification": []}
    summary = {}
    folder = project_path(DL["dirs"]["dl_models"])
    os.makedirs(folder, exist_ok=True)

    # group models by the dataset their variant needs, so each dataset is
    # built once
    by_data = {}
    for name, variant in picks.items():
        s = T.variant_settings(variant)
        by_data.setdefault((s["sequence_length"],
                            s["use_engineered_features"]), []).append(name)

    for key, names in by_data.items():
        settings0 = T.variant_settings(picks[names[0]])
        print(f"--- data: {key[0]} h window"
              + (", with engineered features" if key[1] else "") + " ---")
        train_data, val_data, do_scale = T.build_data(settings0, save=True)
        reg_tr, cls_tr = T.build_targets(train_data, do_scale)
        reg_va, cls_va = T.build_targets(val_data, do_scale)

        for name in names:
            settings = T.variant_settings(picks[name])
            print(f"\n=== {name} ({picks[name]}) ===")
            per_seed, preds, probs, hists = [], [], [], []
            t0 = time.perf_counter()
            for seed in seeds:
                print(f"  seed {seed}")
                model, hist, best, batch_size, loss_fn = T.train_model(
                    name, settings, train_data, val_data, reg_tr, cls_tr,
                    reg_va, cls_va, do_scale, device, seed, verbose=False)
                _, do_pred, cls_prob = T.predict(model, val_data, reg_va,
                                                 cls_va, loss_fn, device,
                                                 batch_size, do_scale)
                j24 = HORIZONS.index(24)
                rm = regression_metrics(val_data.meta["y_reg_24"],
                                        do_pred[:, j24])
                cm = classification_metrics(val_data.meta["y_cls_24"],
                                            cls_prob[:, j24, :].argmax(1))
                per_seed.append({"seed": seed, "val_loss": best["val_loss"],
                                 "epochs": int(hist["epoch"].max()),
                                 "best_epoch": int(best["epoch"]),
                                 "rmse_24": rm["rmse"],
                                 "f1_macro_24": cm["f1_macro"],
                                 "recall_low_24": cm["recall_Low"],
                                 "minutes": float(hist["seconds"].sum()) / 60,
                                 "peak_gpu_gb": float(hist["peak_gpu_gb"].max())})
                preds.append(do_pred)
                probs.append(cls_prob)
                hists.append(hist)
                import torch
                torch.save({"state_dict": model.state_dict(),
                            "channels": S.CHANNELS,
                            "seq_len": settings["sequence_length"],
                            "do_scale": do_scale, "variant": picks[name],
                            "seed": seed,
                            "arch": T.architecture_overrides(settings, name)},
                           os.path.join(folder, f"{name}_seed{seed}.pt"))
                print(f"    val_loss {best['val_loss']:.4f} | RMSE24 "
                      f"{rm['rmse']:.4f} | macroF1_24 {cm['f1_macro']:.4f} "
                      f"| {per_seed[-1]['minutes']:.1f} min")
                del model

            # the average of the seeds IS the final model's prediction
            do_pred = np.mean(preds, axis=0)
            cls_prob = np.mean(probs, axis=0)
            spread = pd.DataFrame(per_seed)

            extra = {"variant": picks[name], "seeds": ",".join(map(str, seeds)),
                     "n_seeds": len(seeds),
                     "epochs": int(spread["epochs"].mean().round()),
                     "best_epoch": int(spread["best_epoch"].mean().round()),
                     "seconds_per_epoch": round(
                         float(np.mean([h["seconds"].mean() for h in hists])), 1),
                     "peak_gpu_gb": round(float(spread["peak_gpu_gb"].max()), 2),
                     "val_loss": round(float(spread["val_loss"].mean()), 5),
                     "rmse_24_seed_std": round(float(spread["rmse_24"].std()), 5),
                     "f1_24_seed_std": round(float(spread["f1_macro_24"].std()), 5)}
            rows = T.score(val_data.meta, do_pred, cls_prob, name,
                           (time.perf_counter() - t0), extra, save=True)
            for task in all_rows:
                all_rows[task] += rows[task]
            summary[name] = {**extra, "per_seed": per_seed}

            # the curve of the first seed, as a representative picture
            T.plot_curve(name, hists[0], suffix=f" ({picks[name]}, seed {seeds[0]})")
            spread.to_csv(os.path.join(folder, f"{name}_seed_spread.csv"),
                          index=False)

            r = [x for x in rows["regression"] if x["horizon"] == 24][0]
            c = [x for x in rows["classification"] if x["horizon"] == 24][0]
            print(f"  AVERAGED: RMSE24 {r['rmse']:.4f} (skill "
                  f"{r['skill_vs_persistence']:+.3f}) | macroF1_24 "
                  f"{c['f1_macro']:.4f} | Low recall {c['recall_Low']:.3f}")
            print(f"  seed spread: RMSE sd {extra['rmse_24_seed_std']:.4f}, "
                  f"macroF1 sd {extra['f1_24_seed_std']:.4f}")

    for task in all_rows:
        print(f"\nWrote {C.update_metrics_csv(task, all_rows[task])}")
    with open(os.path.join(folder, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nNext: python -m src.evaluation.step5_report")


if __name__ == "__main__":
    main()
