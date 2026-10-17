"""
STEP 8.4 check - are the stage-3 models sane?

Run from the project root (after `python -m src.models.stage2 --stage 3`):

    python -m src.models.sanity_stage3

Stage-3 models are trained on 2023 + 2024 + 2025, which means **they can
never be tested**: there is no year left that they have not seen. They
exist only so the app can forecast using everything known.

So this is not an evaluation. It only asks whether they are sane:

  1. every model file loads and produces predictions without error;
  2. on 2025, their predictions agree closely with the stage-2 models'.
     Stage 3 saw 2025 in training and stage 2 did not, so stage 3 should
     look a little better on those rows - but the two should still be
     making recognisably the same forecast. A correlation below 0.95 would
     mean something went wrong in training, and is flagged.

Writes results/test/stage3_sanity.csv.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.models import common as C
from src.models.predict_service import Forecaster

CFG = load_config()
FT = CFG["final_test"]
HORIZONS = CFG["horizons"]
MEMBERS = CFG["ensemble"]["members"]
RESULTS = project_path(FT["results_dir"])
MIN_CORRELATION = 0.95


def main():
    test = pd.read_parquet(project_path(CFG["paths"]["test"]))
    keys = test[["station", "time"]].reset_index(drop=True)
    features = C.feature_columns()
    print(f"Stage-3 sanity check on {len(test):,} rows of 2025")
    print("(these models were TRAINED on 2025 - this is a sanity check, "
          "not a score)\n")

    rows, problems = [], []
    for stage in ["stage2", "final"]:
        try:
            Forecaster(stage)
        except Exception as e:
            problems.append(f"{stage} does not load: {e}")
    if problems:
        for p in problems:
            print("  PROBLEM:", p)
        raise SystemExit(1)

    f2, f3 = Forecaster("stage2"), Forecaster("final")

    # --- the tabular members, member by member ---
    for name in MEMBERS:
        if name in ("bilstm", "cnn_lstm"):
            continue
        for task in ["regression", "classification"]:
            for h in HORIZONS:
                try:
                    a = f2._tabular(name, task, h, test)
                    b = f3._tabular(name, task, h, test)
                except Exception as e:
                    problems.append(f"{name} {task} h{h} failed to predict: {e}")
                    continue
                a = np.asarray(a, dtype="float64")
                b = np.asarray(b, dtype="float64")
                if a.ndim > 1:                 # probabilities: compare P(Low)
                    a, b = a[:, 0], b[:, 0]
                ok = np.isfinite(a) & np.isfinite(b)
                corr = float(np.corrcoef(a[ok], b[ok])[0, 1])
                mad = float(np.mean(np.abs(a[ok] - b[ok])))
                rows.append({"model": name, "task": task, "horizon": h,
                             "correlation": corr, "mean_abs_diff": mad,
                             "flag": corr < MIN_CORRELATION})
                mark = "  <-- FLAG" if corr < MIN_CORRELATION else ""
                print(f"  {name:<14}{task[:3]} h{h:<3} corr {corr:.4f}  "
                      f"mean|diff| {mad:.4f}{mark}")

    # --- the deep members, through the service (all three seeds averaged) ---
    from src.models import sequences as S
    import joblib
    for name in ["bilstm", "cnn_lstm"]:
        try:
            preds = {}
            for stage, fc in [("stage2", f2), ("final", f3)]:
                bundle = joblib.load(os.path.join(fc.folder, "dl_scaler.joblib"))
                data = S.prepare("test", scaler=bundle["scaler"],
                                 medians=bundle["medians"],
                                 seq_len=bundle["seq_len"], with_features=True,
                                 feature_scaler=bundle["feature_scaler"],
                                 feature_medians=bundle["feature_medians"])
                preds[stage] = fc._deep(name, data)
            for j, h in enumerate(HORIZONS):
                for task, idx in [("regression", 0), ("classification", 1)]:
                    a = preds["stage2"][idx]
                    b = preds["final"][idx]
                    a = a[:, j] if task == "regression" else a[:, j, 0]
                    b = b[:, j] if task == "regression" else b[:, j, 0]
                    ok = np.isfinite(a) & np.isfinite(b)
                    corr = float(np.corrcoef(a[ok], b[ok])[0, 1])
                    mad = float(np.mean(np.abs(a[ok] - b[ok])))
                    rows.append({"model": name, "task": task, "horizon": h,
                                 "correlation": corr, "mean_abs_diff": mad,
                                 "flag": corr < MIN_CORRELATION})
                    mark = "  <-- FLAG" if corr < MIN_CORRELATION else ""
                    print(f"  {name:<14}{task[:3]} h{h:<3} corr {corr:.4f}  "
                          f"mean|diff| {mad:.4f}{mark}")
        except Exception as e:
            problems.append(f"{name} failed: {e}")
            print(f"  PROBLEM with {name}: {e}")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "stage3_sanity.csv"), index=False)

    flagged = df[df["flag"]]
    print(f"\n{len(df)} comparisons, correlation "
          f"{df['correlation'].min():.4f} to {df['correlation'].max():.4f}, "
          f"median {df['correlation'].median():.4f}")
    print(f"mean absolute difference: median "
          f"{df['mean_abs_diff'].median():.4f}, worst "
          f"{df['mean_abs_diff'].max():.4f}")
    if len(flagged):
        print(f"\nFLAGGED ({len(flagged)} below correlation {MIN_CORRELATION}):")
        print(flagged.to_string(index=False))
    else:
        print(f"\nNothing flagged: every stage-3 model agrees with its "
              f"stage-2 counterpart above {MIN_CORRELATION} correlation.")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
    print(f"\nWrote {os.path.join(RESULTS, 'stage3_sanity.csv')}")


if __name__ == "__main__":
    main()
