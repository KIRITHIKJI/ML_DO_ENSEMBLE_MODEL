"""
STEP 3.3 - The three tree-based models: Random Forest, XGBoost, LightGBM.

Run from the project root (after src.models.baselines):

    python -m src.models.train_ml
    python -m src.models.train_ml --models xgboost --tasks regression

Every model is trained on 2023 and evaluated on 2024. The 2025 test year is
never loaded.

How missing values are handled:
  Random Forest  cannot cope with NaN, so it gets a median imputer fitted on
                 the TRAINING data only (inside a Pipeline, so it travels
                 with the model).
  XGBoost        handles NaN natively: it learns which way to send a missing
                 value at each split.
  LightGBM       handles NaN natively too, and gets station_id as a real
                 categorical feature.

Class imbalance (Low is about a fifth of the hours) is handled by weighting:
class_weight="balanced_subsample" for the forest, "balanced" for LightGBM,
and explicit balanced sample weights for XGBoost, which has no such argument.

A note on early stopping: XGBoost and LightGBM choose their number of trees
by watching the 2024 validation set. That is a legitimate use of validation,
but it does mean their validation scores are slightly optimistic compared
with the Random Forest, which never looks at 2024. The 2025 test set remains
completely untouched either way.

Writes, for each model / task / horizon:
  models/ml/<model>_<task>_h<h>.(joblib|json|txt)
  results/val/predictions/<model>_<task>_h<h>.parquet
  results/val/importance/<model>_<task>_h<h>.csv
  results/val/metrics_<task>.csv   (rows added or replaced)
"""

import argparse

import numpy as np
import pandas as pd

from src.config import set_seeds
from src.evaluation.metrics import classification_metrics, regression_metrics
from src.models import common as C
from src.models.baselines import metric_row

MODELS = ["random_forest", "xgboost", "lightgbm"]
TASKS = ["regression", "classification"]


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
def fit_random_forest(task, X_tr, y_tr, X_val, features):
    """Median imputation + Random Forest, all inside one Pipeline."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    cfg = dict(C.M["random_forest"])
    class_weight = cfg.pop("classifier_class_weight")
    if task == "regression":
        rf = RandomForestRegressor(random_state=C.SEED, **cfg)
    else:
        rf = RandomForestClassifier(random_state=C.SEED,
                                    class_weight=class_weight, **cfg)
    model = Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("model", rf)])
    model.fit(X_tr, y_tr)

    pred = model.predict(X_val)
    proba = model.predict_proba(X_val) if task == "classification" else None
    importance = model.named_steps["model"].feature_importances_
    return model, pred, proba, importance, {"n_trees": cfg["n_estimators"]}


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
def fit_xgboost(task, X_tr, y_tr, X_val, y_val_fit, features):
    """
    XGBoost with early stopping on the validation year. NaNs go in as they
    are. `y_val_fit` is the validation target with the NaN rows removed,
    used only to decide when to stop adding trees.
    """
    from xgboost import XGBClassifier, XGBRegressor

    cfg = dict(C.M["xgboost"])
    common = dict(n_estimators=C.M["max_rounds"],
                  early_stopping_rounds=C.M["early_stopping_rounds"],
                  random_state=C.SEED, **cfg)
    X_val_fit, y_val_ok = y_val_fit
    weights = None
    if task == "regression":
        model = XGBRegressor(eval_metric="rmse", **common)
    else:
        model = XGBClassifier(objective="multi:softprob", num_class=3,
                              eval_metric="mlogloss", **common)
        weights = C.balanced_sample_weight(y_tr)

    model.fit(X_tr, y_tr, sample_weight=weights,
              eval_set=[(X_val_fit, y_val_ok)], verbose=False)

    pred = model.predict(X_val)
    proba = model.predict_proba(X_val) if task == "classification" else None
    return (model, pred, proba, model.feature_importances_,
            {"best_iteration": int(model.best_iteration)})


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------
def as_lgbm_frame(X, categories):
    """station_id must be a pandas category for LightGBM to treat it as one."""
    X = X.copy()
    for col, cats in categories.items():
        X[col] = pd.Categorical(X[col].astype("int32"), categories=cats)
    return X


def fit_lightgbm(task, X_tr, y_tr, X_val, y_val_fit, features):
    """LightGBM with early stopping on validation and station_id categorical."""
    import lightgbm as lgb

    cfg = dict(C.M["lightgbm"])
    cat_cols = cfg.pop("categorical_features")
    class_weight = cfg.pop("classifier_class_weight")
    cats = {c: sorted(X_tr[c].astype("int32").unique()) for c in cat_cols
            if c in X_tr.columns}
    X_tr_l = as_lgbm_frame(X_tr, cats)
    X_val_l = as_lgbm_frame(X_val, cats)
    X_val_fit, y_val_ok = y_val_fit
    X_val_fit_l = as_lgbm_frame(X_val_fit, cats)

    common = dict(n_estimators=C.M["max_rounds"], random_state=C.SEED,
                  verbose=-1, **cfg)
    if task == "regression":
        model = lgb.LGBMRegressor(objective="regression", **common)
    else:
        model = lgb.LGBMClassifier(objective="multiclass", num_class=3,
                                   class_weight=class_weight, **common)

    model.fit(X_tr_l, y_tr, eval_set=[(X_val_fit_l, y_val_ok)],
              callbacks=[lgb.early_stopping(C.M["early_stopping_rounds"],
                                            verbose=False),
                         lgb.log_evaluation(0)])

    pred = model.predict(X_val_l)
    proba = model.predict_proba(X_val_l) if task == "classification" else None
    return (model, pred, proba, model.feature_importances_,
            {"best_iteration": int(model.best_iteration_ or 0)})


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
SAVE_AS = {"random_forest": "joblib", "xgboost": "xgboost",
           "lightgbm": "lightgbm"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    ap.add_argument("--horizons", nargs="+", type=int, default=C.HORIZONS)
    args = ap.parse_args()

    set_seeds(C.SEED)
    features = C.feature_columns()
    train = C.load_split("train")
    val = C.load_split("val")
    print(f"train {len(train):,} rows | val {len(val):,} rows | "
          f"{len(features)} features | seed {C.SEED}")
    print(f"models: {args.models} | tasks: {args.tasks} | "
          f"horizons: {args.horizons}\n")

    rows = {t: [] for t in TASKS}
    for task in args.tasks:
        for h in args.horizons:
            # training rows: target must exist
            X_tr, y_tr, _ = C.get_xy(train, task, h, features)
            if task == "classification":
                y_tr = y_tr.astype(int)
            # validation: predict for every row, score only the scorable ones
            X_val = val[features]
            y_val = pd.to_numeric(val[C.target_name(task, h)], errors="coerce")
            persist_val = C.persistence(val, task)
            # the subset used for early stopping
            ok = y_val.notna().to_numpy()
            y_val_ok = y_val[ok].astype(int if task == "classification" else float)
            val_fit = (X_val[ok], y_val_ok)

            for name in args.models:
                print(f"  {name:<14}{task[:3]} h{h:<3} "
                      f"training on {len(X_tr):,} rows...", end=" ", flush=True)
                with C.Timer() as t:
                    if name == "random_forest":
                        model, pred, proba, imp, extra = fit_random_forest(
                            task, X_tr, y_tr, X_val, features)
                    elif name == "xgboost":
                        model, pred, proba, imp, extra = fit_xgboost(
                            task, X_tr, y_tr, X_val, val_fit, features)
                    else:
                        model, pred, proba, imp, extra = fit_lightgbm(
                            task, X_tr, y_tr, X_val, val_fit, features)

                if task == "regression":
                    m = regression_metrics(y_val, pred, persist_val)
                    score = f"RMSE {m['rmse']:.4f}  skill {m['skill_vs_persistence']:+.3f}"
                else:
                    m = classification_metrics(y_val, pred, proba)
                    score = (f"macroF1 {m['f1_macro']:.4f}  "
                             f"Low recall {m['recall_Low']:.3f}")
                print(f"{t}  {score}  {extra}")

                C.save_predictions(name, task, h, val, pred, proba)
                C.save_importance(name, task, h, features, imp)
                C.save_model(model, name, task, h, how=SAVE_AS[name])
                row = metric_row(name, task, h, m, t.seconds)
                row.update(extra)
                rows[task].append(row)

    for task in args.tasks:
        if rows[task]:
            print(f"\nWrote {C.update_metrics_csv(task, rows[task])}")
    print("\nNext: python -m src.evaluation.step3_report")


if __name__ == "__main__":
    main()
