"""
STEP 8.1 / 8.4 - retrain the members with everything frozen.

Run from the project root:

    python -m src.models.stage2            # stage 2: train on 2023 + 2024
    python -m src.models.stage2 --stage 3  # stage 3: train on all three years

Stage 2 produces the models that are scored once on 2025. Stage 3 produces
the models the app will use, trained on everything; they are never scored,
because by then there is no year left to score them on.

The important rule here is that NOTHING is chosen. Every hyperparameter,
every number of boosting rounds and every epoch count comes from FROZEN.md,
which was written and committed before any 2025 target was read. In
particular there is no early stopping: early stopping needs a validation
set to watch, and the only year left to watch would be the test year.

Saves to models/stage2/ or models/final/, with the training time logged.
"""

import argparse
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from src.config import load_config, project_path, set_seeds
from src.models import common as C

CFG = load_config()
FT = CFG["final_test"]
HORIZONS = CFG["horizons"]
MEMBERS = CFG["ensemble"]["members"]
SEED = CFG["seed"]

STAGE_YEARS = {2: ["train", "val"], 3: ["train", "val", "test"]}
STAGE_DIR = {2: FT["stage2_dir"], 3: FT["stage3_dir"]}


def frozen_rounds():
    """Read the frozen round counts back out of the committed FROZEN.md."""
    from src.models.freeze import frozen_rounds as fr
    return fr()


def frozen_epochs():
    from src.models.freeze import frozen_epochs as fe
    return fe()


def load_years(stage):
    """
    The training table for this stage.

    Stage 3 includes 2025. That is the one and only place in this project
    where the test year is used for FITTING, and the models it produces are
    never scored - they exist only so the app can use everything known.
    """
    parts = [pd.read_parquet(project_path(CFG["paths"][s]))
             for s in STAGE_YEARS[stage]]
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(["station", "time"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# The tree models and the linear baselines
# ---------------------------------------------------------------------------
def fit_tabular(name, task, h, train, features, rounds, out_dir):
    """One member, one task, one horizon, with a fixed number of rounds."""
    import joblib
    y = pd.to_numeric(train[C.target_name(task, h)], errors="coerce")
    ok = y.notna().to_numpy()
    X = train.loc[ok, features]
    y = y[ok]
    if task == "classification":
        y = y.astype(int)

    t0 = time.perf_counter()
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        cfg = dict(CFG["models"]["random_forest"])
        cw = cfg.pop("classifier_class_weight")
        rf = (RandomForestRegressor(random_state=SEED, **cfg)
              if task == "regression"
              else RandomForestClassifier(random_state=SEED,
                                          class_weight=cw, **cfg))
        model = Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("model", rf)]).fit(X, y)
        path = os.path.join(out_dir, f"{name}_{task}_h{h}.joblib")
        joblib.dump(model, path, compress=3)

    elif name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        cfg = dict(CFG["models"]["xgboost"])
        n = rounds[f"xgboost_{task}_h{h}"]
        common = dict(n_estimators=int(n), random_state=SEED, **cfg)
        if task == "regression":
            model = XGBRegressor(eval_metric="rmse", **common).fit(X, y)
        else:
            w = C.balanced_sample_weight(y)
            model = XGBClassifier(objective="multi:softprob", num_class=3,
                                  eval_metric="mlogloss", **common
                                  ).fit(X, y, sample_weight=w)
        path = os.path.join(out_dir, f"{name}_{task}_h{h}.json")
        model.save_model(path)

    elif name == "lightgbm":
        import lightgbm as lgb
        cfg = dict(CFG["models"]["lightgbm"])
        cat = cfg.pop("categorical_features")
        cw = cfg.pop("classifier_class_weight")
        n = rounds[f"lightgbm_{task}_h{h}"]
        Xl = X.copy()
        cats = {}
        for c in cat:
            if c in Xl.columns:
                cats[c] = sorted(Xl[c].astype("int32").unique())
                Xl[c] = pd.Categorical(Xl[c].astype("int32"), categories=cats[c])
        common = dict(n_estimators=int(n), random_state=SEED, verbose=-1, **cfg)
        model = (lgb.LGBMRegressor(objective="regression", **common)
                 if task == "regression"
                 else lgb.LGBMClassifier(objective="multiclass", num_class=3,
                                         class_weight=cw, **common))
        model.fit(Xl, y)
        path = os.path.join(out_dir, f"{name}_{task}_h{h}.joblib")
        joblib.dump({"model": model, "cats": cats}, path, compress=3)

    elif name == "linear":
        from src.models.baselines import make_linear
        model = make_linear(task).fit(X, y)
        path = os.path.join(out_dir, f"{name}_{task}_h{h}.joblib")
        joblib.dump(model, path, compress=3)
    else:
        raise ValueError(name)

    return path, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# The deep models
# ---------------------------------------------------------------------------
def fit_deep(name, stage, epochs, out_dir):
    """
    The chosen step-5 configuration, retrained on this stage's years for the
    frozen number of epochs - one model per seed, exactly as in step 5.
    The scaler is refitted on the same years.
    """
    import torch
    from src.models import sequences as S
    from src.models import train_dl as T

    settings = T.variant_settings(FT.get("dl_variant", "reg_hybrid_72h_small"))
    device = T.pick_device()
    splits = STAGE_YEARS[stage]
    seq_len = settings["sequence_length"]
    with_feats = settings["use_engineered_features"]

    # one SequenceData per split, all scaled by the scaler fitted on the
    # combined training years
    first = S.prepare(splits[0], seq_len=seq_len, with_features=with_feats)
    datasets = [first]
    for s in splits[1:]:
        datasets.append(S.prepare(s, scaler=first.scaler, medians=first.medians,
                                  seq_len=seq_len, with_features=with_feats,
                                  feature_scaler=first.feature_scaler,
                                  feature_medians=first.feature_medians))
    train_data = merge_sequence_data(datasets)
    do_scale = float(first.scaler.scale_[S.CHANNELS.index("do_mgl")])
    reg_t, cls_t = T.build_targets(train_data, do_scale)

    paths, seconds = [], 0.0
    for seed in CFG["dl_search"]["final_seeds"]:
        n_epochs = epochs[name][seed]
        t0 = time.perf_counter()
        model, hist, best, batch_size, _ = T.train_model(
            name, settings, train_data, train_data, reg_t, cls_t, reg_t,
            cls_t, do_scale, device, seed, max_epochs=n_epochs, verbose=False)
        seconds += time.perf_counter() - t0
        path = os.path.join(out_dir, f"{name}_seed{seed}.pt")
        torch.save({"state_dict": model.state_dict(), "channels": S.CHANNELS,
                    "seq_len": seq_len, "do_scale": do_scale, "seed": seed,
                    "epochs": n_epochs, "stage": stage,
                    "arch": T.architecture_overrides(settings, name)}, path)
        paths.append(path)
        print(f"    seed {seed}: {n_epochs} epochs, "
              f"{(time.perf_counter() - t0) / 60:.1f} min")
        del model
    # the scaler travels with the models
    import joblib
    joblib.dump({"scaler": first.scaler, "medians": first.medians,
                 "feature_scaler": first.feature_scaler,
                 "feature_medians": first.feature_medians,
                 "channels": S.CHANNELS, "seq_len": seq_len,
                 "do_scale": do_scale, "stage": stage},
                os.path.join(out_dir, "dl_scaler.joblib"))
    return paths, seconds


def merge_sequence_data(datasets):
    """
    Stick several SequenceData objects together. They share the station
    arrays (the hourly panel covers all three years), so only the row index
    and the metadata need concatenating.
    """
    from src.models.sequences import SequenceData
    base = datasets[0]
    index = np.concatenate([d.index for d in datasets])
    meta = pd.concat([d.meta for d in datasets], ignore_index=True)
    features = (None if base.features is None
                else np.concatenate([d.features for d in datasets]))
    order = np.lexsort((index[:, 1], index[:, 0]))
    merged = SequenceData(base.arrays, base.station_ids, index[order],
                          meta.iloc[order].reset_index(drop=True),
                          base.scaler, base.medians, seq_len=base.seq_len,
                          features=None if features is None else features[order],
                          feature_scaler=base.feature_scaler,
                          feature_medians=base.feature_medians,
                          feature_names=base.feature_names)
    return merged


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--stage", type=int, default=2, choices=[2, 3])
    ap.add_argument("--models", nargs="+",
                    default=["linear"] + MEMBERS)
    args = ap.parse_args()

    set_seeds(SEED)
    out_dir = project_path(STAGE_DIR[args.stage])
    os.makedirs(out_dir, exist_ok=True)
    rounds, epochs = frozen_rounds(), frozen_epochs()
    years = STAGE_YEARS[args.stage]
    print(f"STAGE {args.stage}: training on {years} "
          f"({'2023+2024' if args.stage == 2 else '2023+2024+2025'})")
    print(f"  everything frozen; no early stopping, no tuning")
    print(f"  -> {out_dir}\n")

    timings = {}
    tabular = [m for m in args.models if m != "bilstm" and m != "cnn_lstm"]
    if tabular:
        train = load_years(args.stage)
        features = C.feature_columns()
        print(f"  {len(train):,} training rows x {len(features)} features")
        for name in tabular:
            for task in ["regression", "classification"]:
                for h in HORIZONS:
                    n = rounds.get(f"{name}_{task}_h{h}", "-")
                    print(f"  {name:<14}{task[:3]} h{h:<3} rounds={n} ...",
                          end=" ", flush=True)
                    path, secs = fit_tabular(name, task, h, train, features,
                                             rounds, out_dir)
                    timings[f"{name}_{task}_h{h}"] = round(secs, 1)
                    print(f"{secs:.0f}s")
        del train

    for name in [m for m in args.models if m in ("bilstm", "cnn_lstm")]:
        print(f"\n  {name}: frozen epochs {epochs[name]}")
        paths, secs = fit_deep(name, args.stage, epochs, out_dir)
        timings[name] = round(secs, 1)

    with open(os.path.join(out_dir, "training_times.json"), "w") as f:
        json.dump(timings, f, indent=2)
    total = sum(timings.values()) / 60
    print(f"\nStage {args.stage} done: {total:.1f} minutes of training, "
          f"models in {out_dir}")


if __name__ == "__main__":
    main()
