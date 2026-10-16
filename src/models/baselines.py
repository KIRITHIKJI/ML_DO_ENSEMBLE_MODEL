"""
STEP 3.2 - The baselines every real model has to beat.

Run from the project root:

    python -m src.models.baselines

Trained on 2023, evaluated on 2024. The 2025 test year is not touched.

  persistence  "DO will be what it is now". No training at all. This is the
               honest bar for a forecast: if a model cannot beat it, it has
               learned nothing useful.
  linear       Linear Regression for the DO value, and multinomial Logistic
               Regression for the class. Both in a sklearn Pipeline that
               imputes missing values with the TRAIN median and standardises
               the features - both fitted on train only, so nothing from 2024
               leaks into the preprocessing.

Writes, for each task and horizon:
  results/val/metrics_regression.csv, metrics_classification.csv
  results/val/predictions/<model>_<task>_h<h>.parquet
  models/ml/linear_<task>_h<h>.joblib
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import set_seeds
from src.evaluation.metrics import classification_metrics, regression_metrics
from src.models import common as C


def metric_row(model, task, h, metrics, seconds):
    """Flatten one result into a row for the metrics csv."""
    row = {"model": model, "task": task, "horizon": h,
           "train_seconds": round(seconds, 1)}
    for k, v in metrics.items():
        if k == "confusion_matrix":
            # keep the matrix in the csv as a flat list, easy to read back
            row["confusion_matrix"] = ";".join(str(int(x)) for x in v.ravel())
        else:
            row[k] = v
    return row


def make_linear(task):
    """
    Median imputation -> standardise -> linear model. Everything inside the
    pipeline is fitted on the training data only.
    """
    steps = [("impute", SimpleImputer(strategy="median")),
             ("scale", StandardScaler())]
    if task == "regression":
        steps.append(("model", LinearRegression()))
    else:
        steps.append(("model", LogisticRegression(
            max_iter=1000, multi_class="multinomial",
            class_weight="balanced", random_state=C.SEED)))
    return Pipeline(steps)


def main():
    set_seeds(C.SEED)
    features = C.feature_columns()
    train = C.load_split("train")
    val = C.load_split("val")
    print(f"train {len(train):,} rows | val {len(val):,} rows | "
          f"{len(features)} features")

    rows = {"regression": [], "classification": []}

    for task in ["regression", "classification"]:
        for h in C.HORIZONS:
            y_val_name = C.target_name(task, h)
            y_val = val[y_val_name].astype("float64")
            persist_val = C.persistence(val, task)

            # ---------------- persistence ----------------
            if task == "regression":
                m = regression_metrics(y_val, persist_val, persist_val)
            else:
                m = classification_metrics(y_val, persist_val)
            C.save_predictions("persistence", task, h, val, persist_val)
            rows[task].append(metric_row("persistence", task, h, m, 0.0))
            score = m["rmse"] if task == "regression" else m["f1_macro"]
            label = "RMSE" if task == "regression" else "macro F1"
            print(f"  persistence {task[:3]} h{h:<3} {label} {score:.4f}")

            # ---------------- linear / logistic ----------------
            X_tr, y_tr, _ = C.get_xy(train, task, h, features)
            model = make_linear(task)
            with C.Timer() as t:
                model.fit(X_tr, y_tr if task == "regression" else y_tr.astype(int))
            X_val = val[features]
            pred = model.predict(X_val)
            proba = model.predict_proba(X_val) if task == "classification" else None
            if task == "regression":
                m = regression_metrics(y_val, pred, persist_val)
            else:
                m = classification_metrics(y_val, pred, proba)
            C.save_predictions("linear", task, h, val, pred, proba)
            C.save_model(model, "linear", task, h, how="joblib")
            rows[task].append(metric_row("linear", task, h, m, t.seconds))
            score = m["rmse"] if task == "regression" else m["f1_macro"]
            print(f"  linear      {task[:3]} h{h:<3} {label} {score:.4f}"
                  f"   ({t})")

    for task in ["regression", "classification"]:
        path = C.update_metrics_csv(task, rows[task])
        print(f"Wrote {path}")
    print("\nNext: python -m src.models.train_ml")


if __name__ == "__main__":
    main()
