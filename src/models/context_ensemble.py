"""
STEP 7.2 - CAE, the Context-Aware Ensemble.

ATAE asked "which member has been right lately?". That turned out to be
worth very little over a plain average. CAE asks a different question:

    "Given what the river is doing RIGHT NOW, which member should I
     believe about what happens next?"

Step 6 showed the members differ by situation rather than by period - the
BiLSTM catches the most onsets at 24 h but has the worst macro F1, LightGBM
is the other way round. A weighting that only tracks recent average error
cannot express "trust the BiLSTM when the oxygen is falling towards 5, and
LightGBM when the river is comfortably above 7". A small supervised model
can.

So CAE is a meta-model, one per task and horizon, whose inputs are:

    member outputs      the 5 predicted DO values, or the 5 x 3 probabilities
    disagreement        how far apart the members are (std of the DO
                        predictions, std of their P(Low)) - the cheapest
                        available signal that this hour is hard
    recent errors       each member's threshold-weighted error over the last
                        336 hours, using only predictions whose target time
                        has already passed (the step-6 machinery)
    river context       DO now, how far it is from 5 mg/L, how fast it is
                        moving over 6 h and 24 h, its daily swing,
                        saturation, water temperature, hours since it was
                        last below 5, and the time of day and year

It is deliberately a SMALL LightGBM (15 leaves, 200 samples per leaf). The
referee of five strong models should not be the biggest model in the room.
There is no station_id: the point is to learn situations, not stations, so
it transfers to a period it has never seen.

Everything is fitted inside the cross-fitting of src/models/crossfit.py and
never on the rows it predicts. The 2025 test split is not read.
"""

import os

import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.evaluation.metrics import threshold_weight
from src.models import crossfit as X
from src.models import ensemble as E

CFG = load_config()
CAE = CFG["context_ensemble"]
ENS = CFG["ensemble"]
MEMBERS = ENS["members"]
CLASS_NAMES = CFG["target"]["class_names"]
N_CLASSES = len(CLASS_NAMES)
LOW = CFG["target"]["thresholds"]["low"]
DO = CFG["target"]["column"]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def context_columns():
    """The river-context columns, as they are named in data/processed."""
    return [c for c in CAE["context_features"] if c != "do_minus_threshold"]


def load_context(keys):
    """
    The river's state at time t, read from the processed features step 2
    built. Those were computed per station from data at or before t, so
    nothing here looks forward.
    """
    cols = context_columns()
    df = pd.read_parquet(project_path(CFG["paths"]["val"]),
                         columns=["station", "time"] + cols)
    out = keys[["station", "time"]].merge(df, on=["station", "time"], how="left")
    assert len(out) == len(keys), "context rows did not line up"
    out["do_minus_threshold"] = out[DO] - LOW
    return out[CAE["context_features"]].astype("float32")


def recent_error_features(keys, y_reg, y_cls, reg, proba, h, task):
    """
    How each member has been doing lately, in the threshold-aware sense,
    over the last `error_window_hours`. Reuses the step-6 code, so the
    "only predictions whose target time has already passed" rule is exactly
    the same one the ensemble uses - and is tested in tests/test_ensemble.py.
    """
    e = E.member_errors(task, y_reg, y_cls, reg, proba,
                        ENS["score_alpha"], ENS["score_sigma"])
    recent, _ = E.recent_errors(keys, e, h, CAE["error_window_hours"],
                                ENS["min_known_errors"])
    return recent


def build_features(task, keys, y_reg, y_cls, reg, proba, h, context=None):
    """
    The full input table for the meta-model. Returns (DataFrame, names).
    Every column is a value at time t or a summary of the past; none of it
    can see beyond t.
    """
    parts, names = [], []
    # take the member names from the data, not from config, so the function
    # works on any number of members (the tests use three)
    n_members = reg.shape[1]
    members = (MEMBERS[:n_members] if n_members <= len(MEMBERS)
               else [f"m{i}" for i in range(n_members)])

    # 1. what the members say
    if task == "regression":
        parts.append(reg)
        names += [f"pred_{m}" for m in members]
    else:
        parts.append(proba.reshape(len(proba), -1))
        names += [f"proba_{m}_{c}" for m in members for c in CLASS_NAMES]

    # 2. how much they disagree - a hard hour looks different from an easy one
    spread_reg = reg.std(axis=1, ddof=0)[:, None]
    spread_low = proba[:, :, 0].std(axis=1, ddof=0)[:, None]
    parts += [spread_reg, spread_low]
    names += ["disagreement_pred", "disagreement_p_low"]

    # 3. how each member has been doing lately
    recent = recent_error_features(keys, y_reg, y_cls, reg, proba, h, task)
    parts.append(recent)
    names += [f"recent_err_{m}" for m in members]

    # 4. what the river is doing
    ctx = load_context(keys) if context is None else context
    parts.append(ctx.to_numpy())
    names += list(ctx.columns)

    X_ = np.column_stack(parts).astype("float32")
    assert not any(n == "station_id" for n in names), \
        "station_id must not be a CAE input"
    return pd.DataFrame(X_, columns=names), names


# ---------------------------------------------------------------------------
# The meta-model
# ---------------------------------------------------------------------------
def make_model(task):
    import lightgbm as lgb
    p = dict(CAE["lightgbm"])
    p.pop("early_stopping_rounds", None)
    common = dict(num_leaves=p["num_leaves"],
                  min_child_samples=p["min_child_samples"],
                  learning_rate=p["learning_rate"],
                  feature_fraction=p["feature_fraction"],
                  lambda_l2=p["lambda_l2"],
                  n_estimators=p["n_estimators"],
                  random_state=CFG["seed"], n_jobs=-1, verbose=-1)
    if task == "regression":
        return lgb.LGBMRegressor(objective="regression", **common)
    return lgb.LGBMClassifier(objective="multiclass", num_class=N_CLASSES,
                              class_weight="balanced", **common)


def fit_cae(task, features, target, times, train_rows):
    """
    Fit one meta-model on the given training rows, stopping early on a
    time-ordered tail of them.
    """
    import lightgbm as lgb
    ok = train_rows[~np.isnan(target[train_rows])]
    inner_train, inner_valid = X.inner_split(times, ok)
    y = target
    if task == "classification":
        y = target.astype("float64")
    model = make_model(task)
    model.fit(features.iloc[inner_train], y[inner_train],
              eval_set=[(features.iloc[inner_valid], y[inner_valid])],
              callbacks=[lgb.early_stopping(
                  CAE["lightgbm"]["early_stopping_rounds"], verbose=False),
                  lgb.log_evaluation(0)])
    return model


def predict_cae(task, model, features, rows=None):
    """Predictions for the given rows (all of them by default)."""
    f = features if rows is None else features.iloc[rows]
    if task == "regression":
        return model.predict(f)
    p = model.predict_proba(f)
    if p.shape[1] < N_CLASSES:                 # a class absent from training
        full = np.zeros((len(p), N_CLASSES))
        full[:, model.classes_.astype(int)] = p
        p = full
    return p


def model_path(task, h, tag="final"):
    folder = project_path(CAE["model_dir"])
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"cae_{task}_h{h}_{tag}.joblib")


def save_model(model, task, h, names, tag="final"):
    import joblib
    path = model_path(task, h, tag)
    joblib.dump({"model": model, "features": names, "task": task,
                 "horizon": h, "members": MEMBERS}, path)
    return path
