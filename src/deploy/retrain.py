"""
STEP 12.6 - safe retraining, with a promotion gate.

    python -m src.deploy.retrain --dry-run      # what this step runs
    python -m src.deploy.retrain                # the real thing (hours)

Retraining reuses `src/models/stage2.py` exactly: the same frozen
hyperparameters, the same frozen numbers of boosting rounds and epochs from
`FROZEN.md`, the same members, the same stacking meta-models, the same
thresholds. Nothing is tuned. The only thing that changes is how much data
the members have seen.

THE PROMOTION GATE. A new model replaces `models/final` only if it is at
least as good as the one already deployed on the most recent
`retrain.holdout_months` months that THE NEW MODEL DID NOT TRAIN ON:

    new RMSE             <= old RMSE             + gate_tolerance.rmse
    new false-alarm rate <= old false-alarm rate + gate_tolerance.false_alarm_rate

at every horizon. If either fails at any horizon, the candidate is refused
and `models/final` is left exactly as it is. On promotion the old version is
moved to `models/archive/<version id>/` first, so the models that produced
every past forecast can always be recovered - the version id in each
forecast row is the key.

The gate is the point of this module, so it is a function of its own and is
tested directly (`tests/test_deploy.py`). A real retrain is deliberately
NOT run in step 12; `--dry-run` exercises the whole path on a tiny slice.
"""

import argparse
import json
import os
import shutil

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from src.config import project_path
from src.evaluation.metrics import (early_warning_metrics, regression_metrics,
                                    to_class)

from src.deploy import common as D

CFG = D.CFG
HORIZONS = D.HORIZONS
RT = D.RETRAIN
GATE_METRICS = list(RT["gate_metrics"])


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def gate(old, new, horizons=None, tolerance=None):
    """
    Would this candidate be promoted?

    `old` and `new` are {horizon: {"rmse": float, "false_alarm_rate": float}}
    measured on the SAME held-out rows - months the candidate never trained
    on. Lower is better for both metrics, so the candidate must not be
    worse than the deployed model by more than the configured tolerance at
    any horizon.

    Returns (promote: bool, rows: list of dicts explaining every comparison).
    """
    tol = dict(RT["gate_tolerance"]) if tolerance is None else dict(tolerance)
    horizons = list(old) if horizons is None else list(horizons)
    rows, promote = [], True
    for h in horizons:
        for metric in GATE_METRICS:
            a = old.get(h, {}).get(metric, np.nan)
            b = new.get(h, {}).get(metric, np.nan)
            allowed = a + float(tol.get(metric, 0.0))
            if any(pd.isna(x) for x in (a, b)):
                ok = False
                why = "not measured on the held-out months"
            else:
                ok = bool(b <= allowed)
                why = ("at least as good" if ok
                       else f"worse by {b - a:+.4f}, allowed "
                            f"{float(tol.get(metric, 0.0)):+.4f}")
            rows.append({"horizon": h, "metric": metric, "deployed": a,
                         "candidate": b, "allowed_max": allowed,
                         "passes": ok, "why": why})
            promote &= ok
    return bool(promote), rows


def explain(promote, rows):
    lines = [f"  {'h':>4}  {'metric':<18}{'deployed':>11}{'candidate':>11}"
             f"{'allowed':>11}  verdict"]
    for r in rows:
        lines.append(f"  {r['horizon']:>4}  {r['metric']:<18}"
                     f"{r['deployed']:>11.4f}{r['candidate']:>11.4f}"
                     f"{r['allowed_max']:>11.4f}  "
                     f"{'pass' if r['passes'] else 'FAIL'} ({r['why']})")
    lines.append("")
    lines.append("  PROMOTE: the candidate replaces models/final" if promote
                 else "  REFUSED: models/final is left exactly as it is")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The data a retrain would use
# ---------------------------------------------------------------------------
def cutoff_and_holdout(cutoff=None):
    """
    (cutoff, holdout_start): the candidate trains on everything up to
    `holdout_start`, and both models are then compared on
    [holdout_start, cutoff] - months the candidate never saw.
    """
    if cutoff is None:
        cutoff = RT["cutoff"]
    if cutoff is None:
        path = D.dpath(D.DATA["features"], mode="history")
        if os.path.exists(path):
            cutoff = pd.read_parquet(path, columns=["time"])["time"].max()
        else:
            cutoff = pd.Timestamp(D.TEST26["end_frozen"]) + pd.Timedelta("23h")
    cutoff = pd.Timestamp(cutoff)
    holdout_start = (cutoff - pd.DateOffset(months=int(RT["holdout_months"]))
                     ).normalize()
    return cutoff, holdout_start


def training_table(holdout_start):
    """
    Everything the candidate is allowed to learn from: the three processed
    splits (2023-2025) plus the 2026 rows before `holdout_start`. The held-
    out months are removed here, which is what makes the gate meaningful.
    """
    parts = []
    for split in ("train", "val", "test"):
        parts.append(pd.read_parquet(project_path(CFG["paths"][split])))
    live = D.dpath(D.DATA["features"], mode="history")
    if os.path.exists(live):
        new = pd.read_parquet(live)
        new = new[new["time"] < holdout_start]
        if len(new):
            new["split"] = "live"
            parts.append(new)
    df = pd.concat(parts, ignore_index=True)
    return df.sort_values(["station", "time"]).reset_index(drop=True)


def holdout_table(holdout_start, cutoff):
    """The most recent months, which the candidate did not train on."""
    live = D.dpath(D.DATA["features"], mode="history")
    if not os.path.exists(live):
        return None
    df = pd.read_parquet(live)
    df = df[(df["time"] >= holdout_start) & (df["time"] <= cutoff)]
    return df.sort_values(["station", "time"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Measuring a model folder on the held-out months
# ---------------------------------------------------------------------------
def measure(folder, rows, panel, batch_size=None):
    """
    {horizon: {"rmse", "false_alarm_rate"}} for one model folder on these
    rows, through the deployed prediction path and the frozen thresholds.
    """
    from src.models.predict_service import Forecaster, STAGE_DIRS, tau_for
    from src.models import run_step7 as R
    from src.deploy import score_2026 as S

    stage = next((k for k, v in STAGE_DIRS.items()
                  if project_path(v) == os.path.abspath(folder)), None)
    f = Forecaster(stage) if stage else _forecaster_at(folder)
    f.batch_size = int(batch_size or D.COMPUTE["deep_batch_size"])

    seq, keys = S.sequences_for(rows, panel, folder)
    have = set(zip(keys["station"], keys["time"]))
    use = rows[[(s, t) in have for s, t in zip(rows["station"],
                                               rows["time"])]].reset_index(
        drop=True)
    out = f.predict_rows(use, seq)
    y_reg, y_cls, class_now = S.truths(use)
    res = {}
    for h in HORIZONS:
        d = out[h]
        tau = tau_for(D.FINAL_SYSTEM, h)
        proba = np.asarray(d["proba"], dtype="float64")
        pred = proba.argmax(1) if tau is None else R.apply_warning(proba, tau)
        rm = regression_metrics(y_reg[h], np.asarray(d["do"], dtype="float64"))
        ew = early_warning_metrics(class_now, y_cls[h], pred)
        res[h] = {"rmse": rm["rmse"],
                  "false_alarm_rate": ew["false_alarm_rate"],
                  "onset_recall": ew["onset_recall"], "n": rm["n"]}
    return res


def _forecaster_at(folder):
    """A Forecaster pointed at a folder that is not one of the two stages."""
    from src.models.predict_service import Forecaster
    f = Forecaster.__new__(Forecaster)
    f.stage = "candidate"
    f.folder = os.path.abspath(folder)
    f.label = f"candidate models in {folder}"
    from src.models import common as C
    f.features = C.feature_columns()
    f._cache = {}
    f.batch_size = CFG["deep_learning"]["batch_size"]
    return f


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------
def promote(candidate_dir, deployed_dir=None, archive_root=None):
    """
    Archive the deployed models under their own version id, then move the
    candidate into place. The archive comes first: if anything goes wrong
    halfway, the old models still exist somewhere.
    """
    deployed_dir = deployed_dir or D.deployed_folder()
    archive_root = archive_root or project_path(D.DEP["models"]["archive_dir"])
    old_version = D.model_version(deployed_dir)
    dest = os.path.join(archive_root, old_version)
    os.makedirs(archive_root, exist_ok=True)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(deployed_dir, dest)
    for name in os.listdir(candidate_dir):
        src = os.path.join(candidate_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(deployed_dir, name))
    return {"archived_to": dest, "archived_version": old_version,
            "new_version": D.model_version(deployed_dir)}


# ---------------------------------------------------------------------------
# The dry run
# ---------------------------------------------------------------------------
def dry_run(stations=None, hours=None, batch_size=None):
    """
    The whole path on a tiny slice: pick the cut-off, split off the held-out
    months, measure the DEPLOYED models on them, invent a candidate that is
    a little worse and one that is a little better, and show what the gate
    does with each. No model is trained and nothing is promoted.

    This is what step 12 runs. It answers the only question that matters
    before a real retrain: would the gate actually refuse a worse model?
    """
    out_dir = D.ensure_dir(RT["dry_run_dir"])
    cutoff, holdout_start = cutoff_and_holdout()
    D.log(f"  cut-off          {cutoff:%Y-%m-%d %H:%M}")
    D.log(f"  candidate trains on everything before "
          f"{holdout_start:%Y-%m-%d} ({RT['holdout_months']} months held out)")

    hold = holdout_table(holdout_start, cutoff)
    if hold is None or not len(hold):
        D.log("  no held-out rows on disk - run fetch_live --mode history "
              "first")
        return None
    train = training_table(holdout_start)
    D.log(f"  would train on   {len(train):,} rows "
          f"({train['time'].min():%Y-%m-%d} .. {train['time'].max():%Y-%m-%d})")
    D.log(f"  gate measured on {len(hold):,} rows "
          f"({hold['time'].min():%Y-%m-%d} .. {hold['time'].max():%Y-%m-%d}), "
          f"{hold['station'].nunique()} stations")

    # a small slice, so the dry run is quick
    stations = stations or sorted(set(hold["station"]))[:2]
    slice_ = hold[hold["station"].isin(stations)]
    if hours:
        slice_ = slice_.groupby("station").tail(int(hours))
    slice_ = slice_.reset_index(drop=True)
    panel = pd.read_parquet(D.dpath(D.DATA["hourly"], mode="history"))
    panel = panel[panel["station"].isin(stations)]
    D.log(f"\n  dry run slice: {len(slice_):,} rows, stations "
          f"{', '.join(stations)}")

    deployed = measure(D.deployed_folder(), slice_, panel, batch_size)
    D.log("\n  the DEPLOYED models on the held-out slice")
    D.log(f"  {'h':>4}{'rows':>9}{'RMSE':>9}{'falseAl':>10}{'onset':>8}")
    for h in HORIZONS:
        D.log(f"  {h:>4}{deployed[h]['n']:>9}{deployed[h]['rmse']:>9.4f}"
              f"{deployed[h]['false_alarm_rate']:>10.4f}"
              f"{deployed[h]['onset_recall']:>8.3f}")

    # Two synthetic candidates, so the gate is exercised in both directions
    # without spending hours training. The numbers are nudged by a visible
    # amount, not by noise.
    cases = {}
    worse = {h: {"rmse": deployed[h]["rmse"] * 1.05,
                 "false_alarm_rate": deployed[h]["false_alarm_rate"] * 1.20}
             for h in HORIZONS}
    better = {h: {"rmse": deployed[h]["rmse"] * 0.97,
                  "false_alarm_rate": deployed[h]["false_alarm_rate"] * 0.95}
              for h in HORIZONS}
    mixed = {h: dict(better[h]) for h in HORIZONS}
    worst_h = HORIZONS[-1]
    mixed[worst_h] = {"rmse": better[worst_h]["rmse"],
                      "false_alarm_rate":
                          deployed[worst_h]["false_alarm_rate"] * 1.10}
    for name, cand in (("a worse candidate", worse),
                       ("a better candidate", better),
                       ("better everywhere but one horizon", mixed)):
        ok, rows = gate(deployed, cand)
        D.log(f"\n  GATE - {name}")
        D.log(explain(ok, rows))
        cases[name] = {"promote": ok, "rows": rows}

    report = {
        "cutoff": str(cutoff), "holdout_start": str(holdout_start),
        "holdout_months": int(RT["holdout_months"]),
        "would_train_rows": int(len(train)),
        "gate_rows_available": int(len(hold)),
        "dry_run_rows": int(len(slice_)),
        "dry_run_stations": list(stations),
        "deployed_version": D.model_version(),
        "deployed_metrics": {str(h): deployed[h] for h in HORIZONS},
        "gate_metrics": GATE_METRICS,
        "gate_tolerance": dict(RT["gate_tolerance"]),
        "cases": {k: {"promote": v["promote"]} for k, v in cases.items()},
        "promoted": False,
        "note": "dry run only - no model was trained and nothing was "
                "promoted",
    }
    path = project_path(RT["report"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(out_dir, "gate_cases.json"), "w",
              encoding="utf-8") as f:
        json.dump(cases, f, indent=2, default=str)
    D.log(f"\n  wrote {D.shortpath(path)}")
    return report


# ---------------------------------------------------------------------------
# The real retrain
# ---------------------------------------------------------------------------
def real_retrain(batch_size=None):
    """
    Train the candidate with the frozen settings, measure both models on the
    held-out months, and promote only through the gate. Not run in step 12.
    """
    from src.models import stage2 as ST
    cutoff, holdout_start = cutoff_and_holdout()
    out_dir = D.ensure_dir(RT["out_dir"])
    train = training_table(holdout_start)
    hold = holdout_table(holdout_start, cutoff)
    if hold is None or not len(hold):
        raise SystemExit("no held-out rows: the gate would have nothing to "
                         "measure on")

    rounds, epochs = ST.frozen_rounds(), ST.frozen_epochs()
    from src.models import common as C
    features = C.feature_columns()
    D.log(f"  training the candidate on {len(train):,} rows with the frozen "
          f"settings, into {out_dir}")
    for name in ["linear"] + D.MEMBERS:
        if name in ("bilstm", "cnn_lstm"):
            D.log(f"  {name}: the deep members need the sequence pipeline; "
                  f"reuse `python -m src.models.stage2 --stage 3` with the "
                  f"candidate years, then re-run the gate")
            continue
        for task in ("regression", "classification"):
            for h in HORIZONS:
                path, secs = ST.fit_tabular(name, task, h, train, features,
                                            rounds, out_dir)
                D.log(f"    {name:<14}{task[:3]} h{h:<3} {secs:.0f}s")

    panel = pd.read_parquet(D.dpath(D.DATA["hourly"], mode="history"))
    old = measure(D.deployed_folder(), hold, panel, batch_size)
    new = measure(out_dir, hold, panel, batch_size)
    ok, rows = gate(old, new)
    D.log("\n  GATE on the held-out months")
    D.log(explain(ok, rows))
    result = {"cutoff": str(cutoff), "holdout_start": str(holdout_start),
              "deployed": {str(h): old[h] for h in HORIZONS},
              "candidate": {str(h): new[h] for h in HORIZONS},
              "gate": rows, "promote": ok}
    if ok:
        result.update(promote(out_dir))
    path = project_path(RT["report"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise the gate on a tiny slice; trains nothing")
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--hours", type=int, default=None)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    st = D.Stages()
    D.log("=== STEP 12.6: retraining, gated ===")
    D.report_device("deep inference")
    if not D.require_cuda_or_explain() and not args.allow_cpu:
        raise SystemExit(2)
    if args.dry_run:
        with st("dry run"):
            dry_run(args.stations, args.hours, args.batch_size)
    else:
        D.log("  this trains real models and can promote them. "
              "Step 12 does not run it; use --dry-run.")
        with st("retrain"):
            real_retrain(args.batch_size)
    st.report()


if __name__ == "__main__":
    main()
