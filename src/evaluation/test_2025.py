"""
STEP 8.2 / 8.3 - predict and score the 2025 test year. Once.

Run from the project root (after src.models.stage2):

    python -m src.evaluation.test_2025

Everything this script uses was frozen and committed in FROZEN.md before a
single 2025 target was read: the members' hyperparameters, the number of
rounds and epochs, the stacking meta-models fitted on 2024, and the warning
thresholds chosen on the out-of-fold 2024 probabilities. This script fits
nothing. It loads, predicts, and scores.

  Primary    members retrained on 2023+2024 (stage 2) -> the step-7
             stacking meta-models -> the final forecast.
  Robustness the same chain with the original 2023-only members, to show
             how much the extra training year was worth.

Also scores persistence, linear, each member, the simple average and ATAE,
with both argmax and the frozen warning rule, plus the multimodal ablation.

Writes everything to results/test/.
"""

import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.evaluation.metrics import (classification_metrics, danger_zone_rmse,
                                    early_warning_metrics, regression_metrics,
                                    to_class)
from src.models import common as C
from src.models import ensemble as E
from src.models import run_step7 as R

CFG = load_config()
FT = CFG["final_test"]
ENS = CFG["ensemble"]
HORIZONS = CFG["horizons"]
CLASS_NAMES = CFG["target"]["class_names"]
MEMBERS = ENS["members"]
LOW = CFG["target"]["thresholds"]["low"]
DO = CFG["target"]["column"]
SEED = CFG["seed"]
RESULTS = project_path(FT["results_dir"])
METHODS = ["persistence", "linear"] + MEMBERS + ["simple_average", "atae",
                                                 "stacking"]


# ---------------------------------------------------------------------------
# Loading the frozen pieces
# ---------------------------------------------------------------------------
def load_thresholds():
    with open(project_path(FT["thresholds_file"]), encoding="utf-8") as f:
        return json.load(f)


def tau_for(thresholds, method, h):
    return thresholds.get(f"{method}_h{h}", {}).get("tau")


def predict_tabular(name, task, h, test, features, folder):
    """A stage-2 (or stage-3) member predicting the test rows."""
    import joblib
    if name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        model = XGBRegressor() if task == "regression" else XGBClassifier()
        model.load_model(os.path.join(folder, f"{name}_{task}_h{h}.json"))
        X = test[features]
        return model.predict(X) if task == "regression" else model.predict_proba(X)
    if name == "lightgbm":
        bundle = joblib.load(os.path.join(folder, f"{name}_{task}_h{h}.joblib"))
        model, cats = bundle["model"], bundle["cats"]
        X = test[features].copy()
        for c, levels in cats.items():
            if c in X.columns:
                X[c] = pd.Categorical(X[c].astype("int32"), categories=levels)
        return model.predict(X) if task == "regression" else model.predict_proba(X)
    model = joblib.load(os.path.join(folder, f"{name}_{task}_h{h}.joblib"))
    X = test[features]
    return model.predict(X) if task == "regression" else model.predict_proba(X)


def predict_deep(name, folder, test_keys):
    """
    The three seeds of one deep model, averaged, over the test rows.
    Returns (reg (n,4), proba (n,4,3)).
    """
    import joblib
    import torch
    from src.models import sequences as S
    from src.models import train_dl as T
    from src.models.dl_models import MODELS

    bundle = joblib.load(os.path.join(folder, "dl_scaler.joblib"))
    seq_len, do_scale = bundle["seq_len"], bundle["do_scale"]
    data = S.prepare("test", scaler=bundle["scaler"], medians=bundle["medians"],
                     seq_len=seq_len, with_features=True,
                     feature_scaler=bundle["feature_scaler"],
                     feature_medians=bundle["feature_medians"])
    reg_t, cls_t = T.build_targets(data, do_scale)
    device = T.pick_device()
    n_stations = int(max(data.station_ids.values()) + 1)
    n_features = 0 if data.features is None else data.features.shape[1]
    from src.models.dl_models import MultiTaskLoss
    loss_fn = MultiTaskLoss(T.class_weights_from_train(cls_t)).to(device)

    preds, probs = [], []
    for seed in CFG["dl_search"]["final_seeds"]:
        ckpt = torch.load(os.path.join(folder, f"{name}_seed{seed}.pt"),
                          map_location=device)
        model = MODELS[name](S.N_CHANNELS, n_stations, cfg=ckpt["arch"],
                             n_features=n_features).to(device)
        model.load_state_dict(ckpt["state_dict"])
        _, do_pred, proba = T.predict(model, data, reg_t, cls_t, loss_fn,
                                      device, CFG["deep_learning"]["batch_size"],
                                      do_scale)
        preds.append(do_pred)
        probs.append(proba)
        del model
    keys = data.meta[["station", "time"]]
    assert (keys["time"].to_numpy() == test_keys["time"].to_numpy()).all(), \
        "the deep models' rows do not line up with the test table"
    return np.mean(preds, axis=0), np.mean(probs, axis=0)


def apply_stacking(task, h, reg, proba):
    """The step-7 stacking meta-model, fitted on 2024, applied to 2025."""
    import joblib
    path = os.path.join(project_path(CFG["context_ensemble"]["model_dir"]),
                        f"stacking_{task}_h{h}_final.joblib")
    model = joblib.load(path)["model"]
    if task == "regression":
        return model.predict(reg)
    out = model.predict_proba(proba.reshape(len(proba), -1))
    if out.shape[1] < 3:
        full = np.zeros((len(out), 3))
        full[:, model.classes_.astype(int)] = out
        out = full
    return out


# ---------------------------------------------------------------------------
# Building every method's 2025 predictions
# ---------------------------------------------------------------------------
def build_predictions(folder, test, features, keys, label):
    """All members and combinations, for one set of member models."""
    print(f"\n  predicting 2025 with the {label} members ({folder})")
    out = {h: {} for h in HORIZONS}
    reg_all, proba_all = {h: [] for h in HORIZONS}, {h: [] for h in HORIZONS}

    for name in MEMBERS:
        t0 = time.perf_counter()
        if name in ("bilstm", "cnn_lstm"):
            r, p = predict_deep(name, folder, keys)
            for j, h in enumerate(HORIZONS):
                reg_all[h].append(r[:, j])
                proba_all[h].append(p[:, j, :])
                out[h][name] = {"reg": r[:, j], "proba": p[:, j, :]}
        else:
            for h in HORIZONS:
                r = predict_tabular(name, "regression", h, test, features, folder)
                p = predict_tabular(name, "classification", h, test, features,
                                    folder)
                reg_all[h].append(r)
                proba_all[h].append(p)
                out[h][name] = {"reg": r, "proba": p}
        print(f"    {name:<15}{time.perf_counter() - t0:>6.0f}s")

    for h in HORIZONS:
        reg = np.column_stack(reg_all[h])
        proba = np.stack(proba_all[h], axis=1)
        out[h]["_reg"], out[h]["_proba"] = reg, proba

        avg_r, _ = E.simple_average("regression", reg, proba)
        avg_p, _ = E.simple_average("classification", reg, proba)
        out[h]["simple_average"] = {"reg": avg_r, "proba": avg_p}

        out[h]["stacking"] = {
            "reg": apply_stacking("regression", h, reg, proba),
            "proba": apply_stacking("classification", h, reg, proba)}
    return out


def add_atae(preds, keys, y_reg, y_cls, label):
    """ATAE with the settings chosen in step 6, for the record."""
    with open(project_path(ENS["chosen_file"]), encoding="utf-8") as f:
        chosen = json.load(f)
    for h in HORIZONS:
        reg, proba = preds[h]["_reg"], preds[h]["_proba"]
        cr, cc = chosen["regression"], chosen["classification"]
        r, _ = E.atae("regression", keys, y_reg[h], y_cls[h], reg, proba,
                      cr["alpha"], cr["sigma"], cr["window_hours"],
                      cr["temperature"], h)
        p, _ = E.atae("classification", keys, y_reg[h], y_cls[h], reg, proba,
                      cc["alpha"], cc["sigma"], cc["window_hours"],
                      cc["temperature"], h)
        preds[h]["atae"] = {"reg": r, "proba": p}
    return preds


def add_baselines(preds, test, keys):
    """persistence and linear, computed directly on the test rows."""
    now = test[DO].to_numpy(dtype="float64")
    features = C.feature_columns()
    folder = project_path(FT["stage2_dir"])
    for h in HORIZONS:
        preds[h]["persistence"] = {"reg": now, "proba": None,
                                   "cls": to_class(now)}
        preds[h]["linear"] = {
            "reg": predict_tabular("linear", "regression", h, test, features,
                                   folder),
            "proba": predict_tabular("linear", "classification", h, test,
                                     features, folder)}
    return preds


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_all(preds, y_reg, y_cls, class_now, thresholds, label):
    rows_r, rows_c, saved = [], [], {}
    for h in HORIZONS:
        persistence = preds[h]["persistence"]["reg"]
        for name in METHODS:
            m = preds[h].get(name)
            if m is None:
                continue
            reg = regression_metrics(y_reg[h], m["reg"], persistence)
            reg["danger_rmse"] = danger_zone_rmse(y_reg[h], m["reg"],
                                                  *ENS["danger_zone"])
            rows_r.append({"method": name, "horizon": h, "members": label,
                           **reg})

            proba = m.get("proba")
            argmax = m.get("cls") if proba is None else proba.argmax(1)
            tau = tau_for(thresholds, name, h)
            warn = argmax if (proba is None or tau is None) else \
                R.apply_warning(proba, tau)
            for rule, pred in [("argmax", argmax), ("warn", warn)]:
                cm = classification_metrics(y_cls[h], pred, proba)
                cm.update(early_warning_metrics(class_now, y_cls[h], pred))
                flat = {k: v for k, v in cm.items() if k != "confusion_matrix"}
                flat["confusion_matrix"] = ";".join(
                    str(int(x)) for x in cm["confusion_matrix"].ravel())
                rows_c.append({"method": name, "horizon": h, "rule": rule,
                               "members": label, "tau": tau, **flat})
            saved[(name, h)] = {"reg": m["reg"], "argmax": argmax,
                                "warn": warn, "proba": proba}
    return pd.DataFrame(rows_r), pd.DataFrame(rows_c), saved


# ---------------------------------------------------------------------------
# Lead time
# ---------------------------------------------------------------------------
def lead_times(keys, y_reg_now, warn_flags, h=24, window=None):
    """
    For every real low-DO event in 2025, how early the system warned.

    An event starts when DO drops below the threshold after at least
    `recovery` hours at or above it - the same definition step 1 used. A
    warning counts if the h-hour-ahead rule fired at any hour in the
    `window` hours before the onset, and the lead time is the earliest such
    hour.
    """
    recovery = FT["event_recovery_hours"]
    window = window or FT["lead_time_window_hours"]
    out = []
    df = pd.DataFrame({"station": keys["station"].to_numpy(),
                       "time": pd.to_datetime(keys["time"].to_numpy()),
                       "do": y_reg_now, "warn": warn_flags})
    for station, g in df.groupby("station", sort=True):
        g = g.sort_values("time").reset_index(drop=True)
        below = g["do"] < LOW
        recovered = g["do"].notna() & (g["do"] >= LOW)
        run = recovered.rolling(recovery, min_periods=1).sum()
        prev = run.shift(1).fillna(recovery)
        starts = np.flatnonzero((below & (prev >= recovery)).to_numpy())
        for s in starts:
            onset_time = g["time"].iloc[s]
            lo = onset_time - pd.Timedelta(hours=window)
            prior = g[(g["time"] >= lo) & (g["time"] < onset_time)]
            fired = prior[prior["warn"] == 1]
            if len(fired):
                lead = (onset_time - fired["time"].iloc[0]).total_seconds() / 3600
                out.append({"station": station, "onset": onset_time,
                            "warned": True, "lead_hours": lead})
            else:
                out.append({"station": station, "onset": onset_time,
                            "warned": False, "lead_hours": np.nan})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# The multimodal ablation
# ---------------------------------------------------------------------------
def run_ablation(train, test, y_reg, y_cls, class_now):
    """
    Three input sets, retrained on 2023+2024 with the frozen settings and
    rounds, scored on 2025 with the thresholds frozen in 8.0b.
    """
    import joblib
    from src.models.freeze import (ablation_features, fit_ablation_lightgbm,
                                   predict_ablation, frozen_rounds)
    rounds = frozen_rounds()
    with open(project_path(FT["ablation_thresholds_file"]), encoding="utf-8") as f:
        abl_tau = json.load(f)
    folder = project_path(FT["ablation_dir"])
    rows, preds = [], {}
    for set_name, groups in FT["ablation_sets"].items():
        cols = ablation_features(groups)
        for h in HORIZONS:
            for task in ["regression", "classification"]:
                n = rounds[f"lightgbm_{task}_h{h}"]
                model, cats = fit_ablation_lightgbm(task, h, cols, train, n)
                out = predict_ablation(model, cats, test, cols, task)
                joblib.dump({"model": model, "cats": cats, "cols": cols},
                            os.path.join(folder,
                                         f"ablation_{set_name}_{task}_h{h}_stage2.joblib"))
                preds[(set_name, task, h)] = out
            reg = preds[(set_name, "regression", h)]
            proba = preds[(set_name, "classification", h)]
            tau = abl_tau.get(f"{set_name}_h{h}", {}).get("tau")
            pred = proba.argmax(1) if tau is None else R.apply_warning(proba, tau)
            rm = regression_metrics(y_reg[h], reg)
            cm = classification_metrics(y_cls[h], pred, proba)
            ew = early_warning_metrics(class_now, y_cls[h], pred)
            rows.append({"set": set_name, "horizon": h, "n_features": len(cols),
                         "tau": tau, "rmse": rm["rmse"], "r2": rm["r2"],
                         "f1_macro": cm["f1_macro"],
                         "onset_recall": ew["onset_recall"],
                         "false_alarm_rate": ew["false_alarm_rate"]})
            print(f"    {set_name:<6} h{h:<3} RMSE {rm['rmse']:.4f}  "
                  f"macroF1 {cm['f1_macro']:.4f}  onset "
                  f"{ew['onset_recall']:.3f}")
    return pd.DataFrame(rows), preds


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS, exist_ok=True)
    thresholds = load_thresholds()
    features = C.feature_columns()
    test = pd.read_parquet(project_path(CFG["paths"]["test"]))
    keys = test[["station", "time"]].reset_index(drop=True)
    y_reg = {h: pd.to_numeric(test[f"y_reg_{h}"], errors="coerce").to_numpy()
             for h in HORIZONS}
    y_cls = {h: test[f"y_cls_{h}"].astype("float64").to_numpy()
             for h in HORIZONS}
    class_now = to_class(test[DO])
    print(f"TEST YEAR 2025: {len(test):,} rows, {test['station'].nunique()} "
          f"stations, {test['time'].min():%Y-%m-%d} .. {test['time'].max():%Y-%m-%d}")
    print(f"frozen thresholds from {FT['thresholds_file']}")

    results = {}
    for label, folder_key in [("stage2", "stage2_dir"), ("stage1", None)]:
        folder = (project_path(FT[folder_key]) if folder_key
                  else project_path(CFG["models"]["dirs"]["ml_models"]))
        if label == "stage1":
            # the original 2023-only members: trees in models/ml, deep in models/dl
            preds = build_stage1_predictions(test, features, keys)
        else:
            preds = build_predictions(folder, test, features, keys, label)
        preds = add_baselines(preds, test, keys)
        preds = add_atae(preds, keys, y_reg, y_cls, label)
        r, c, saved = score_all(preds, y_reg, y_cls, class_now, thresholds,
                                label)
        results[label] = {"reg": r, "cls": c, "saved": saved, "preds": preds}
        r.to_csv(os.path.join(RESULTS, f"metrics_regression_{label}.csv"),
                 index=False)
        c.to_csv(os.path.join(RESULTS, f"metrics_classification_{label}.csv"),
                 index=False)
        print(f"  scored: {label}")

    # save the primary predictions row by row
    primary = results["stage2"]["saved"]
    for (name, h), m in primary.items():
        out = keys.copy()
        out["y_true_reg"] = y_reg[h]
        out["y_true_cls"] = y_cls[h]
        out["class_now"] = class_now
        out["pred_reg"] = m["reg"]
        out["pred_argmax"] = m["argmax"]
        out["pred_warn"] = m["warn"]
        if m["proba"] is not None:
            for i, n in enumerate(CLASS_NAMES):
                out[f"proba_{n}"] = m["proba"][:, i]
        out.to_parquet(os.path.join(RESULTS, f"pred_{name}_h{h}.parquet"),
                       index=False)

    # lead time, for the primary stacking system at 24 h
    warn24 = primary[("stacking", 24)]["warn"] == 0
    lead = lead_times(keys, test[DO].to_numpy(dtype="float64"),
                      warn24.astype(int), h=24)
    lead.to_csv(os.path.join(RESULTS, "lead_times.csv"), index=False)
    print(f"\n  low-DO events in 2025: {len(lead)}, warned in advance: "
          f"{lead['warned'].sum()} ({lead['warned'].mean():.1%}), "
          f"median lead {lead.loc[lead['warned'], 'lead_hours'].median():.0f} h")

    # the multimodal ablation
    print("\n  multimodal ablation (retrain on 2023+2024, frozen thresholds)")
    train = pd.concat([pd.read_parquet(project_path(CFG["paths"]["train"])),
                       pd.read_parquet(project_path(CFG["paths"]["val"]))],
                      ignore_index=True)
    abl, _ = run_ablation(train, test, y_reg, y_cls, class_now)
    abl.to_csv(os.path.join(RESULTS, "ablation.csv"), index=False)

    print(f"\nWrote {RESULTS}")
    print("Next: python -m src.evaluation.final_report")


def build_stage1_predictions(test, features, keys):
    """
    The robustness check: the original members, trained on 2023 only, that
    steps 3 and 5 produced. Trees come from models/ml, deep models from
    models/dl.
    """
    print("\n  predicting 2025 with the stage-1 members (2023 only)")
    ml = project_path(CFG["models"]["dirs"]["ml_models"])
    dl = project_path(CFG["deep_learning"]["dirs"]["dl_models"])
    out = {h: {} for h in HORIZONS}
    reg_all, proba_all = {h: [] for h in HORIZONS}, {h: [] for h in HORIZONS}
    import joblib
    for name in MEMBERS:
        if name in ("bilstm", "cnn_lstm"):
            r, p = predict_deep_stage1(name, dl, keys)
            for j, h in enumerate(HORIZONS):
                reg_all[h].append(r[:, j])
                proba_all[h].append(p[:, j, :])
                out[h][name] = {"reg": r[:, j], "proba": p[:, j, :]}
        else:
            for h in HORIZONS:
                if name == "xgboost":
                    from xgboost import XGBClassifier, XGBRegressor
                    mr = XGBRegressor()
                    mr.load_model(os.path.join(ml, f"{name}_regression_h{h}.json"))
                    mc = XGBClassifier()
                    mc.load_model(os.path.join(ml, f"{name}_classification_h{h}.json"))
                    r = mr.predict(test[features])
                    p = mc.predict_proba(test[features])
                elif name == "lightgbm":
                    import lightgbm as lgb
                    r_b = lgb.Booster(model_file=os.path.join(
                        ml, f"{name}_regression_h{h}.txt"))
                    c_b = lgb.Booster(model_file=os.path.join(
                        ml, f"{name}_classification_h{h}.txt"))
                    X = test[features].copy()
                    X["station_id"] = X["station_id"].astype("category")
                    r = r_b.predict(X)
                    p = c_b.predict(X)
                else:
                    mr = joblib.load(os.path.join(ml, f"{name}_regression_h{h}.joblib"))
                    mc = joblib.load(os.path.join(ml, f"{name}_classification_h{h}.joblib"))
                    r = mr.predict(test[features])
                    p = mc.predict_proba(test[features])
                reg_all[h].append(r)
                proba_all[h].append(p)
                out[h][name] = {"reg": r, "proba": p}
        print(f"    {name}")
    for h in HORIZONS:
        reg = np.column_stack(reg_all[h])
        proba = np.stack(proba_all[h], axis=1)
        out[h]["_reg"], out[h]["_proba"] = reg, proba
        avg_r, _ = E.simple_average("regression", reg, proba)
        avg_p, _ = E.simple_average("classification", reg, proba)
        out[h]["simple_average"] = {"reg": avg_r, "proba": avg_p}
        out[h]["stacking"] = {
            "reg": apply_stacking("regression", h, reg, proba),
            "proba": apply_stacking("classification", h, reg, proba)}
    return out


def predict_deep_stage1(name, folder, keys):
    """The step-5 deep models (trained on 2023), predicting 2025."""
    import torch
    from src.models import sequences as S
    from src.models import train_dl as T
    from src.models.dl_models import MODELS, MultiTaskLoss

    bundle = S.load_scaler()
    seq_len = bundle["seq_len"]
    data = S.prepare("test", scaler=bundle["scaler"], medians=bundle["medians"],
                     seq_len=seq_len, with_features=True,
                     feature_scaler=bundle["feature_scaler"],
                     feature_medians=bundle["feature_medians"])
    do_scale = float(bundle["scaler"].scale_[S.CHANNELS.index("do_mgl")])
    reg_t, cls_t = T.build_targets(data, do_scale)
    device = T.pick_device()
    n_stations = int(max(data.station_ids.values()) + 1)
    n_features = 0 if data.features is None else data.features.shape[1]
    loss_fn = MultiTaskLoss(T.class_weights_from_train(cls_t)).to(device)
    preds, probs = [], []
    for seed in CFG["dl_search"]["final_seeds"]:
        ckpt = torch.load(os.path.join(folder, f"{name}_seed{seed}.pt"),
                          map_location=device)
        model = MODELS[name](S.N_CHANNELS, n_stations, cfg=ckpt["arch"],
                             n_features=n_features).to(device)
        model.load_state_dict(ckpt["state_dict"])
        _, do_pred, proba = T.predict(model, data, reg_t, cls_t, loss_fn,
                                      device, CFG["deep_learning"]["batch_size"],
                                      do_scale)
        preds.append(do_pred)
        probs.append(proba)
        del model
    return np.mean(preds, axis=0), np.mean(probs, axis=0)


if __name__ == "__main__":
    main()
