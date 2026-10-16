"""
STEP 5.2 - A small, systematic search over the deep-model settings.

Run from the project root:

    python -m src.models.search_dl

Each variant in `dl_search.variants` is run for both models, trained on 2023
and scored on 2024. The variants build on each other, so the table shows
what each change was worth on its own:

    step4                 the step-4 settings, re-run as the reference
    reg                   more dropout, AdamW + weight decay, lower lr
    reg_hybrid            + the 211 engineered features at time t
    reg_hybrid_72h        + a 72-hour window
    reg_hybrid_72h_small  + hidden size 32

The winner for each model is chosen by VALIDATION LOSS - the quantity the
networks are actually trained to minimise - and not by any single metric.
Picking by "best 24 h macro F1" would be choosing the luckiest run on one
number out of eight.

The 2025 test split is never loaded.

Writes results/val/dl_search.csv (one row per model x variant).
"""

import os
import time

import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.models import train_dl as T
from src.models.dl_models import MODELS, count_parameters

CFG = load_config()
SEARCH = CFG["dl_search"]
HORIZONS = CFG["horizons"]


def data_key(settings):
    """Variants with the same window and input set can share one dataset."""
    return (settings["sequence_length"], settings["use_engineered_features"])


def main():
    device = T.pick_device()
    T.describe_device(device)
    variants = SEARCH["variants"]
    print(f"\nSearching {len(variants)} variants x {len(MODELS)} models = "
          f"{len(variants) * len(MODELS)} runs, seed {CFG['seed']}")
    print(f"Chosen by {SEARCH['select_by']} on validation (2024)\n")

    rows = []
    cache_key, cache = None, None
    for variant in variants:
        settings = T.variant_settings(variant)
        key = data_key(settings)
        if key != cache_key:
            # free the previous dataset before building the next one: the
            # hybrid feature matrices are a few hundred MB each
            cache = None
            print(f"--- building data for {key[0]} h window"
                  + (", with engineered features" if key[1] else "") + " ---")
            cache = T.build_data(settings, save=False)
            cache_key = key
        train_data, val_data, do_scale = cache
        reg_tr, cls_tr = T.build_targets(train_data, do_scale)
        reg_va, cls_va = T.build_targets(val_data, do_scale)

        for name in MODELS:
            print(f"\n[{variant} / {name}]")
            t0 = time.perf_counter()
            model, hist, best, batch_size, loss_fn = T.train_model(
                name, settings, train_data, val_data, reg_tr, cls_tr,
                reg_va, cls_va, do_scale, device, CFG["seed"])
            val_loss, do_pred, cls_prob = T.predict(
                model, val_data, reg_va, cls_va, loss_fn, device,
                batch_size, do_scale)

            j24 = HORIZONS.index(24)
            from src.evaluation.metrics import (classification_metrics,
                                                regression_metrics)
            rm = regression_metrics(val_data.meta["y_reg_24"],
                                    do_pred[:, j24],
                                    val_data.meta["do_now"])
            cm = classification_metrics(val_data.meta["y_cls_24"],
                                        cls_prob[:, j24, :].argmax(1),
                                        cls_prob[:, j24, :])
            rows.append({
                "model": name, "variant": variant,
                "sequence_length": settings["sequence_length"],
                "engineered_features": settings["use_engineered_features"],
                "dropout": settings.get("dropout", "default"),
                "optimizer": settings.get("optimizer", "adam"),
                "learning_rate": settings["learning_rate"],
                "hidden_size": settings.get("hidden_size", "default"),
                "parameters": count_parameters(model),
                "epochs": int(hist["epoch"].max()),
                "best_epoch": int(best["epoch"]),
                "val_loss": round(float(best["val_loss"]), 5),
                "rmse_24": round(rm["rmse"], 4),
                "r2_24": round(rm["r2"], 4),
                "skill_24": round(rm["skill_vs_persistence"], 4),
                "f1_macro_24": round(cm["f1_macro"], 4),
                "recall_low_24": round(cm["recall_Low"], 4),
                "minutes": round((time.perf_counter() - t0) / 60, 2),
                "seconds_per_epoch": round(float(hist["seconds"].mean()), 1),
            })
            print(f"  -> val_loss {best['val_loss']:.4f} | RMSE24 "
                  f"{rm['rmse']:.4f} | macroF1_24 {cm['f1_macro']:.4f} | "
                  f"{rows[-1]['minutes']:.1f} min")
            del model

    df = pd.DataFrame(rows)
    # mark the winner per model: lowest validation loss
    df["chosen"] = False
    for name in MODELS:
        here = df[df["model"] == name]
        if len(here):
            df.loc[here[SEARCH["select_by"]].idxmin(), "chosen"] = True

    path = project_path(SEARCH["results_csv"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nWrote {path}\n")
    show = ["model", "variant", "parameters", "epochs", "best_epoch",
            "val_loss", "rmse_24", "f1_macro_24", "recall_low_24",
            "minutes", "chosen"]
    print(df[show].to_string(index=False))
    for name in MODELS:
        win = df[(df["model"] == name) & df["chosen"]]
        if len(win):
            print(f"\nchosen for {name}: {win['variant'].iloc[0]} "
                  f"(val_loss {win['val_loss'].iloc[0]:.4f})")
    print("\nNext: python -m src.models.finalize_dl")


if __name__ == "__main__":
    main()
