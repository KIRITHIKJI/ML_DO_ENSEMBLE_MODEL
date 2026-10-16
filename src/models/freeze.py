"""
STEP 8.0 - fix every remaining choice, then freeze.

Run from the project root, BEFORE anything reads a 2025 target:

    python -m src.models.freeze

Three jobs:

  a) The warning threshold tau, per method and horizon, chosen on the
     OUT-OF-FOLD 2024 probabilities from step 7 - never in-sample. Step 7
     showed that a tau tuned to sit exactly on the 2% false-alarm budget
     lands at 2.3-2.5% on data it has not seen, so the budget used to
     CHOOSE tau is tightened by a safety margin. The promise stays 2%; the
     selection aims at 1.8%.

  b) The ablation thresholds: a LightGBM per input set (water only, plus
     weather, plus air) trained on 2023 alone, predicting 2024, with tau
     chosen the same way. The ablation models that get scored on 2025 are
     retrained later on 2023+2024, but their thresholds are fixed here.

  c) FROZEN.md: the commit, the split, every member and its settings, the
     exact number of rounds and epochs to use when retraining, the
     thresholds, the pass criteria, and a SHA-256 for every model file the
     test will touch.

After this file is committed, nothing may be chosen again. Anything learned
from 2025 after this point would make the test meaningless.
"""

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.evaluation.metrics import early_warning_metrics
from src.models import run_step7 as R

CFG = load_config()
FT = CFG["final_test"]
ENS = CFG["ensemble"]
S7 = CFG["step7"]
HORIZONS = CFG["horizons"]
CLASS_NAMES = CFG["target"]["class_names"]
MEMBERS = ENS["members"]
OOF = project_path(S7["oof_predictions"])
BUDGET = FT["select_false_alarm_rate"]


# ---------------------------------------------------------------------------
# 8.0a - tau from the out-of-fold 2024 probabilities
# ---------------------------------------------------------------------------
def choose_tau_oof(proba, y_cls, class_now, budget=BUDGET):
    """
    The lowest-risk threshold that still catches the most onsets inside the
    budget. Identical rule to step 7, but applied to out-of-fold
    probabilities and with the tightened budget.
    """
    best = None
    for tau in R.tau_grid():
        pred = R.apply_warning(proba, tau)
        m = early_warning_metrics(class_now, y_cls, pred)
        rate, recall = m["false_alarm_rate"], m["onset_recall"]
        if np.isnan(rate) or np.isnan(recall) or rate > budget:
            continue
        key = (recall, -rate, tau)
        if best is None or key > best[0]:
            best = (key, float(tau), float(recall), float(rate))
    if best is None:
        return {"tau": None, "rule": "argmax", "onset_recall_2024": np.nan,
                "false_alarm_2024": np.nan}
    return {"tau": best[1], "rule": "P(Low) >= tau",
            "onset_recall_2024": best[2], "false_alarm_2024": best[3]}


def warning_thresholds():
    """One threshold per method and horizon, from step 7's out-of-fold files."""
    out = {}
    for method in S7["methods"]:
        for h in HORIZONS:
            path = os.path.join(OOF, f"{method}_classification_h{h}.parquet")
            if not os.path.exists(path):
                continue
            d = pd.read_parquet(path)
            cols = [f"proba_{n}" for n in CLASS_NAMES]
            if not all(c in d for c in cols):
                out[f"{method}_h{h}"] = {"tau": None, "rule": "own rule",
                                         "onset_recall_2024": np.nan,
                                         "false_alarm_2024": np.nan}
                continue
            res = choose_tau_oof(d[cols].to_numpy(),
                                 d["y_true"].to_numpy(),
                                 d["class_now"].to_numpy())
            out[f"{method}_h{h}"] = res
            tau = res["tau"]
            print(f"  {method:<16} h{h:<3} tau="
                  + (f"{tau:.2f}" if tau is not None else "argmax")
                  + f"  onset recall {res['onset_recall_2024']:.3f}"
                  f"  false alarm {res['false_alarm_2024']:.4f}")
    return out


# ---------------------------------------------------------------------------
# 8.0b - the ablation models and their thresholds
# ---------------------------------------------------------------------------
def ablation_features(groups_wanted):
    import json as _json
    with open(project_path(CFG["paths"]["feature_groups"]), encoding="utf-8") as f:
        groups = _json.load(f)
    return [c for g in groups_wanted for c in groups[g]]


def fit_ablation_lightgbm(task, h, cols, train, rounds):
    """LightGBM on one input set, with the step-3 settings and a fixed
    number of rounds (so nothing is stopped on data it should not watch)."""
    import lightgbm as lgb
    from src.models.common import balanced_sample_weight
    cfg = dict(CFG["models"]["lightgbm"])
    cat = cfg.pop("categorical_features")
    class_weight = cfg.pop("classifier_class_weight")
    y = pd.to_numeric(train[f"{'y_reg' if task == 'regression' else 'y_cls'}_{h}"],
                      errors="coerce")
    ok = y.notna().to_numpy()
    X = train.loc[ok, cols].copy()
    cats = {}
    for c in cat:
        if c in X.columns:
            cats[c] = sorted(X[c].astype("int32").unique())
            X[c] = pd.Categorical(X[c].astype("int32"), categories=cats[c])
    common = dict(n_estimators=int(rounds), random_state=CFG["seed"],
                  verbose=-1, **cfg)
    if task == "regression":
        model = lgb.LGBMRegressor(objective="regression", **common)
        model.fit(X, y[ok])
    else:
        model = lgb.LGBMClassifier(objective="multiclass", num_class=3,
                                   class_weight=class_weight, **common)
        model.fit(X, y[ok].astype(int))
    return model, cats


def predict_ablation(model, cats, df, cols, task):
    X = df[cols].copy()
    for c, levels in cats.items():
        if c in X.columns:
            X[c] = pd.Categorical(X[c].astype("int32"), categories=levels)
    return model.predict(X) if task == "regression" else model.predict_proba(X)


def ablation_thresholds(rounds_by_h):
    """
    Train each input set on 2023, predict 2024, choose tau. The models are
    saved so the freeze can hash them; the ones scored on 2025 are retrained
    on 2023+2024 later with the same settings.
    """
    import joblib
    from src.evaluation.metrics import to_class
    folder = project_path(FT["ablation_dir"])
    os.makedirs(folder, exist_ok=True)
    train = pd.read_parquet(project_path(CFG["paths"]["train"]))
    val = pd.read_parquet(project_path(CFG["paths"]["val"]))
    class_now_val = to_class(val[CFG["target"]["column"]])

    out = {}
    for set_name, groups in FT["ablation_sets"].items():
        cols = ablation_features(groups)
        for h in HORIZONS:
            model, cats = fit_ablation_lightgbm("classification", h, cols,
                                                train, rounds_by_h[h])
            proba = predict_ablation(model, cats, val, cols, "classification")
            res = choose_tau_oof(proba,
                                 val[f"y_cls_{h}"].astype("float64").to_numpy(),
                                 class_now_val)
            out[f"{set_name}_h{h}"] = res
            joblib.dump({"model": model, "cats": cats, "cols": cols,
                         "set": set_name, "horizon": h, "task": "classification",
                         "trained_on": "2023"},
                        os.path.join(folder,
                                     f"ablation_{set_name}_classification_h{h}_2023.joblib"))
            tau = res["tau"]
            print(f"  {set_name:<6} h{h:<3} {len(cols):>3} features  tau="
                  + (f"{tau:.2f}" if tau is not None else "argmax")
                  + f"  onset recall {res['onset_recall_2024']:.3f}")
    return out


# ---------------------------------------------------------------------------
# 8.0c - FROZEN.md
# ---------------------------------------------------------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def frozen_rounds():
    """The number of boosting rounds to use when retraining, from step 3."""
    out = {}
    for task in ["regression", "classification"]:
        d = pd.read_csv(project_path("results", "val", f"metrics_{task}.csv"))
        for model in ["xgboost", "lightgbm"]:
            for h in HORIZONS:
                row = d[(d["model"] == model) & (d["horizon"] == h)]
                if len(row) and not pd.isna(row["best_iteration"].iloc[0]):
                    out[f"{model}_{task}_h{h}"] = int(row["best_iteration"].iloc[0])
    return out


def frozen_epochs():
    """The number of epochs to use per deep model and seed, from step 5."""
    out = {}
    for name in ["bilstm", "cnn_lstm"]:
        path = project_path("models", "dl", f"{name}_seed_spread.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        out[name] = {int(r["seed"]): int(r["best_epoch"]) for _, r in d.iterrows()}
    return out


def write_frozen(thresholds, abl_thresholds, rounds, epochs):
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("# FROZEN - the state of the project before the 2025 test\n")
    lines.append(f"Written **{now}** at commit `{git_commit()}`.\n")
    lines.append(
        "Everything below was decided using 2023 and 2024 only. After this "
        "file was committed, no setting, threshold or model choice was "
        "changed. The 2025 test is scored once against exactly this.\n")

    lines.append("## The split\n")
    lines.append(f"- train **{CFG['split']['train_years']}**, "
                 f"validation **{CFG['split']['val_years']}**, "
                 f"test **{CFG['split']['test_years']}** (hold-out)")
    lines.append(f"- {CFG['download']['target_stations']} stations, horizons "
                 f"{HORIZONS} h, classes "
                 f"Low < {CFG['target']['thresholds']['low']} mg/L, Medium to "
                 f"{CFG['target']['thresholds']['good']}, Good above")
    lines.append(f"- seed {CFG['seed']} everywhere\n")

    lines.append("## The final system\n")
    lines.append(f"`final_system = {CFG['final_system']}` - chosen in step 7 "
                 f"by a rule written before the result was known. The "
                 f"context-aware ensemble (CAE) and the adaptive "
                 f"threshold-aware ensemble (ATAE) were both tested and "
                 f"**not adopted**; they are still scored on 2025 for the "
                 f"record.\n")

    lines.append("## Members and settings\n")
    for name in MEMBERS:
        key = {"random_forest": "random_forest", "xgboost": "xgboost",
               "lightgbm": "lightgbm"}.get(name)
        if key:
            lines.append(f"- **{name}**: `{CFG['models'][key]}`")
    lines.append(f"- **bilstm / cnn_lstm**: variant "
                 f"`{CFG['dl_search']['variants'][-1]}` "
                 f"(`{CFG['dl_variants']['reg_hybrid_72h_small']}`), "
                 f"sequence length {CFG['dl_variants']['reg_hybrid_72h_small']['sequence_length']} h, "
                 f"seeds {CFG['dl_search']['final_seeds']}, predictions averaged")
    lines.append("")

    lines.append("## Rounds and epochs to use when retraining (no early stopping)\n")
    lines.append("The 2025 year must not be watched, so the number of rounds "
                 "is fixed here at whatever step 3 and step 5 found on 2024.\n")
    lines.append(md_table(["model / task / horizon", "rounds"],
                          [[k, v] for k, v in sorted(rounds.items())]))
    lines.append("")
    lines.append(md_table(["deep model", "seed", "epochs"],
                          [[n, s, e] for n, d in epochs.items()
                           for s, e in sorted(d.items())]))
    lines.append("")

    lines.append("## Warning thresholds\n")
    lines.append(f"Chosen on the OUT-OF-FOLD 2024 probabilities from step 7, "
                 f"as the threshold catching the most onsets with a false-alarm "
                 f"rate at or below **{BUDGET:.1%}** "
                 f"(target {FT['target_false_alarm_rate']:.1%} minus a "
                 f"{FT['safety_margin']:.1%} safety margin, because step 7 "
                 f"showed a tau tuned to the limit overshoots out of sample).\n")
    rows = []
    for key, v in thresholds.items():
        method, h = key.rsplit("_h", 1)
        rows.append([method, f"{h} h",
                     "argmax" if v["tau"] is None else f"{v['tau']:.2f}",
                     "-" if np.isnan(v["onset_recall_2024"]) else f"{v['onset_recall_2024']:.3f}",
                     "-" if np.isnan(v["false_alarm_2024"]) else f"{v['false_alarm_2024']:.4f}"])
    lines.append(md_table(["method", "horizon", "tau",
                           "onset recall (2024 oof)", "false alarm (2024 oof)"],
                          rows))
    lines.append("")

    lines.append("## Ablation thresholds\n")
    rows = []
    for key, v in abl_thresholds.items():
        s, h = key.rsplit("_h", 1)
        rows.append([s, f"{h} h",
                     "argmax" if v["tau"] is None else f"{v['tau']:.2f}",
                     "-" if np.isnan(v["onset_recall_2024"]) else f"{v['onset_recall_2024']:.3f}"])
    lines.append(md_table(["input set", "horizon", "tau",
                           "onset recall (2024)"], rows))
    lines.append("")

    lines.append("## Pass criteria (unchanged since step 3)\n")
    lines.append(f"```\n{json.dumps(CFG['pass_criteria'], indent=2)}\n```\n")

    lines.append("## File hashes\n")
    lines.append("SHA-256 of every model file the 2025 test reads. If one of "
                 "these changes, the test is no longer the one described here.\n")
    rows = []
    for folder in ["models/dl", "models/ml", "models/ensemble",
                   "models/ablation"]:
        p = project_path(folder)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            full = os.path.join(p, f)
            if os.path.isfile(full):
                rows.append([f"{folder}/{f}", f"`{sha256(full)}`",
                             f"{os.path.getsize(full):,}"])
    lines.append(md_table(["file", "sha256", "bytes"], rows))
    lines.append("")

    path = project_path(FT["frozen_file"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path, len(rows)


def md_table(header, rows):
    out = ["| " + " | ".join(str(c) for c in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    t0 = time.perf_counter()
    os.makedirs(project_path("models", "ensemble"), exist_ok=True)
    print("8.0a - warning thresholds from the out-of-fold 2024 probabilities")
    print(f"       budget {BUDGET:.1%} "
          f"(target {FT['target_false_alarm_rate']:.1%} - "
          f"margin {FT['safety_margin']:.1%})")
    thresholds = warning_thresholds()
    with open(project_path(FT["thresholds_file"]), "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    print(f"  -> {FT['thresholds_file']}\n")

    rounds = frozen_rounds()
    epochs = frozen_epochs()
    rounds_cls = {h: rounds[f"lightgbm_classification_h{h}"] for h in HORIZONS}
    print("8.0b - ablation models on 2023, thresholds on 2024")
    abl = ablation_thresholds(rounds_cls)
    with open(project_path(FT["ablation_thresholds_file"]), "w",
              encoding="utf-8") as f:
        json.dump(abl, f, indent=2)
    print(f"  -> {FT['ablation_thresholds_file']}\n")

    print("8.0c - FROZEN.md")
    path, n_files = write_frozen(thresholds, abl, rounds, epochs)
    print(f"  -> {path} ({n_files} files hashed)")
    print(f"\nDone in {(time.perf_counter() - t0) / 60:.1f} min.")
    print("COMMIT THIS NOW, before anything touches 2025.")


if __name__ == "__main__":
    main()
