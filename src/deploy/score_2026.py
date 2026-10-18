"""
STEP 12.3 - score the deployed models on 2026.

    python -m src.deploy.fetch_live --mode history     # first
    python -m src.deploy.score_2026
    python -m src.deploy.score_2026 --penalty          # the live-data penalty

The models are `models/final` - the five members retrained on
2023+2024+2025 with the frozen settings - combined by the step-7 stacking
meta-model and thresholded by the frozen warning rule. Nothing is fitted,
tuned or chosen here, and nothing about 2026 is allowed to change a 2025
number.

The predictions come from `predict_service.Forecaster.predict_rows`, which
is the same object the app forecasts with, so what is scored here is
literally the deployed path. The metrics are `src/evaluation/metrics.py`,
the warning rule is `run_step7.apply_warning`, the events and lead times
are `test_2025.lead_times`, and the intervals are 1,000 station-week
resamples through `src/deploy/boot.py`. All of it already existed.

The criteria were written down in reports/plan_2026.md before any 2026
dissolved-oxygen value was read, and this script refuses to score a window
that ends later than the plan allows.
"""

import argparse
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from src.config import project_path
from src.evaluation.metrics import (classification_metrics, danger_zone_rmse,
                                    early_warning_metrics, regression_metrics,
                                    to_class)
from src.evaluation.test_2025 import lead_times, predict_tabular
from src.models import ensemble as E
from src.models import run_step7 as R
from src.models import common as C
from src.models.predict_service import Forecaster, tau_for

from src.deploy import boot as B
from src.deploy import common as D
from src.deploy import fetch_live as F

CFG = D.CFG
HORIZONS = D.HORIZONS
MEMBERS = D.MEMBERS
CLASS_NAMES = D.CLASS_NAMES
DO = D.DO
FINAL = D.FINAL_SYSTEM
METHODS = ["persistence", "linear"] + MEMBERS + ["simple_average", "stacking"]
RESULTS = D.TEST26["results_dir"]


# ---------------------------------------------------------------------------
# Guards: the plan comes first
# ---------------------------------------------------------------------------
def check_plan(end):
    """
    Refuse to score anything the pre-registration did not allow: the plan
    file must exist, and the window must not run past the date written into
    it before 2026 was read.
    """
    plan = project_path(D.TEST26["plan_file"])
    if not os.path.exists(plan):
        raise SystemExit(
            f"{D.TEST26['plan_file']} is missing. The 2026 test is only a "
            f"test if the plan was written and committed first "
            f"(step 12.1).")
    frozen = pd.Timestamp(D.TEST26["end_frozen"]) + pd.Timedelta("23h")
    if pd.Timestamp(end) > frozen:
        raise SystemExit(
            f"the window ends {end:%Y-%m-%d %H:%M} but the plan fixed the "
            f"end at {frozen:%Y-%m-%d %H:%M}. Extending the test period "
            f"after the fact is exactly what pre-registration is for.")
    D.log(f"  plan: {D.TEST26['plan_file']} (window end fixed at "
          f"{frozen:%Y-%m-%d %H:%M})")


# ---------------------------------------------------------------------------
# Predicting
# ---------------------------------------------------------------------------
def sequences_for(features, panel, folder):
    """
    The deep members' inputs for these rows, built by the project's own
    `sequences.prepare` with the scaler that travels with models/final.
    Returns (SequenceData, the rows it actually covers) - a row needs a full
    72-hour window behind it, so the sequence set can be a subset.
    """
    import joblib
    from src.models import sequences as S
    bundle = joblib.load(os.path.join(folder, "dl_scaler.joblib"))
    data = S.prepare(None, scaler=bundle["scaler"], medians=bundle["medians"],
                     seq_len=bundle["seq_len"], with_features=True,
                     feature_scaler=bundle["feature_scaler"],
                     feature_medians=bundle["feature_medians"],
                     rows=features, panel=panel)
    keys = data.meta[["station", "time"]].copy()
    return data, keys


def predict(features, panel, batch_size=None):
    """
    Every method's 2026 predictions, on the rows all five members can see.

    Returns (keys, preds) where preds[h][method] = {"reg": ..., "proba": ...}
    """
    folder = D.deployed_folder()
    f = Forecaster(D.DEP["models"]["stage"])
    f.batch_size = int(batch_size or D.COMPUTE["deep_batch_size"])
    D.log(f"  models: {D.shortpath(folder)}  "
          f"version {D.model_version()}")

    seq, keys = sequences_for(features, panel, folder)
    rows = features.merge(keys.assign(_keep=1), on=["station", "time"],
                          how="inner")
    assert len(rows) == len(keys), "the feature rows and the sequences differ"
    # predict_rows keeps the order of `processed_rows`, and the deep members
    # keep the order of `seq`, so the two must be in the same order
    order = (keys.reset_index().set_index(["station", "time"])
             .loc[list(zip(rows["station"], rows["time"])), "index"].to_numpy())
    assert (order == np.arange(len(order))).all(), \
        "the sequence rows are not in the same order as the feature rows"
    D.log(f"  {len(rows):,} scoreable rows, "
          f"{rows['station'].nunique()} stations, "
          f"{rows['time'].min():%Y-%m-%d} .. {rows['time'].max():%Y-%m-%d}")

    t0 = time.perf_counter()
    out = f.predict_rows(rows, seq)
    D.log(f"  members + ensemble: {time.perf_counter() - t0:.0f}s")

    preds = {h: {} for h in HORIZONS}
    for h in HORIZONS:
        d = out[h]
        for name in MEMBERS:
            preds[h][name] = {"reg": d["member_reg"][name],
                              "proba": d["member_proba"][name]}
        reg = np.column_stack([preds[h][m]["reg"] for m in MEMBERS])
        proba = np.stack([preds[h][m]["proba"] for m in MEMBERS], axis=1)
        avg_r, _ = E.simple_average("regression", reg, proba)
        avg_p, _ = E.simple_average("classification", reg, proba)
        preds[h]["simple_average"] = {"reg": avg_r, "proba": avg_p}
        preds[h]["stacking"] = {"reg": np.asarray(d["do"], dtype="float64"),
                                "proba": np.asarray(d["proba"],
                                                    dtype="float64")}

    # the two baselines, on the same rows
    now = rows[DO].to_numpy(dtype="float64")
    features_cols = C.feature_columns()
    for h in HORIZONS:
        preds[h]["persistence"] = {"reg": now, "proba": None,
                                   "cls": to_class(now)}
        preds[h]["linear"] = {
            "reg": predict_tabular("linear", "regression", h, rows,
                                   features_cols, folder),
            "proba": predict_tabular("linear", "classification", h, rows,
                                     features_cols, folder)}
    return rows, preds


# ---------------------------------------------------------------------------
# Scoring - the project's own metric code
# ---------------------------------------------------------------------------
def truths(rows):
    y_reg = {h: pd.to_numeric(rows[f"y_reg_{h}"], errors="coerce").to_numpy()
             for h in HORIZONS}
    y_cls = {h: rows[f"y_cls_{h}"].astype("float64").to_numpy()
             for h in HORIZONS}
    return y_reg, y_cls, to_class(rows[DO])


def score(preds, y_reg, y_cls, class_now, label="live_2026"):
    rows_r, rows_c, saved = [], [], {}
    for h in HORIZONS:
        persistence = preds[h]["persistence"]["reg"]
        for name in METHODS:
            m = preds[h].get(name)
            if m is None:
                continue
            reg = regression_metrics(y_reg[h], m["reg"], persistence)
            reg["danger_rmse"] = danger_zone_rmse(
                y_reg[h], m["reg"], *CFG["ensemble"]["danger_zone"])
            rows_r.append({"method": name, "horizon": h, "members": label,
                           **reg})

            proba = m.get("proba")
            argmax = m.get("cls") if proba is None else proba.argmax(1)
            tau = tau_for(name, h)
            warn = argmax if (proba is None or tau is None) \
                else R.apply_warning(proba, tau)
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


def per_station(keys, y_reg, y_cls, class_now, saved, h=24):
    reg = saved[(FINAL, h)]["reg"]
    warn = saved[(FINAL, h)]["warn"]
    d = pd.DataFrame({"station": keys["station"].to_numpy(),
                      "y_reg": y_reg[h], "y_cls": y_cls[h],
                      "class_now": class_now, "reg": reg, "warn": warn})
    rows = []
    for station, g in d.groupby("station"):
        rm = regression_metrics(g["y_reg"], g["reg"])
        cm = classification_metrics(g["y_cls"], g["warn"])
        ew = early_warning_metrics(g["class_now"], g["y_cls"], g["warn"])
        rows.append({"station": station, "rows": len(g), "rmse": rm["rmse"],
                     "r2": rm["r2"], "f1_macro": cm["f1_macro"],
                     "onset_recall": ew["onset_recall"],
                     "false_alarm_rate": ew["false_alarm_rate"],
                     "n_onsets": ew["n_onsets"]})
    return pd.DataFrame(rows).sort_values("n_onsets", ascending=False)


# ---------------------------------------------------------------------------
# 2025 against 2026, with the pre-registered verdicts
# ---------------------------------------------------------------------------
def reference_2025(parallel=True):
    """
    The 2025 point estimates and 95% station-week intervals of the frozen
    system, computed from the 2025 predictions that step 8 already wrote.
    Nothing about 2025 is changed - this only re-reads it.
    """
    out = []
    for h in HORIZONS:
        d = pd.read_parquet(project_path(CFG["final_test"]["results_dir"],
                                         f"pred_stacking_h{h}.parquet"))
        res = B.bootstrap(d[["station", "time"]], d["y_true_reg"].to_numpy(),
                          d["y_true_cls"].to_numpy(), d["class_now"].to_numpy(),
                          {"reg": d["pred_reg"].to_numpy(),
                           "cls": d["pred_warn"].to_numpy()},
                          parallel=parallel)
        out.append(B.intervals_frame("test_2025", h, res))
    return pd.concat(out, ignore_index=True)


def verdict(metric, value, lo, hi):
    """
    The three words fixed in plan_2026.md section 7.2. Lower is better for
    RMSE and false alarms; higher is better for onset recall and macro F1.
    """
    if any(pd.isna(x) for x in (value, lo, hi)):
        return "not scored"
    lower_better = metric in ("rmse", "false_alarm_rate")
    if lower_better:
        if value <= hi:
            return "improved" if value < lo else "holds"
        return "degraded"
    if value >= lo:
        return "improved" if value > hi else "holds"
    return "degraded"


def compare(ref, reg26, cls26):
    """The side-by-side table and the verdict for each horizon and metric."""
    rows = []
    for h in HORIZONS:
        got = {
            "rmse": float(reg26[(reg26["method"] == FINAL)
                                & (reg26["horizon"] == h)]["rmse"].iloc[0]),
        }
        c = cls26[(cls26["method"] == FINAL) & (cls26["horizon"] == h)
                  & (cls26["rule"] == "warn")].iloc[0]
        got["f1_macro"] = float(c["f1_macro"])
        got["onset_recall"] = float(c["onset_recall"])
        got["false_alarm_rate"] = float(c["false_alarm_rate"])
        for metric, value in got.items():
            r = ref[(ref["horizon"] == h) & (ref["metric"] == metric)]
            lo = float(r["lo"].iloc[0]) if len(r) else np.nan
            hi = float(r["hi"].iloc[0]) if len(r) else np.nan
            mean = float(r["mean"].iloc[0]) if len(r) else np.nan
            rows.append({"horizon": h, "metric": metric,
                         "value_2025": mean, "ci_lo_2025": lo,
                         "ci_hi_2025": hi, "value_2026": value,
                         "change": value - mean,
                         "verdict": verdict(metric, value, lo, hi)})
    return pd.DataFrame(rows)


def overall_verdict(cmp_table):
    """
    PASS / PARTIAL / FAIL, exactly as section 7.3 of the plan defines it.
    """
    budget = float(CFG["final_test"]["target_false_alarm_rate"])
    head = 24 if 24 in HORIZONS else HORIZONS[len(HORIZONS) // 2]
    at_h = cmp_table[cmp_table["horizon"] == head].set_index("metric")
    far = float(at_h.loc["false_alarm_rate", "value_2026"])
    if far > budget:
        return "FAIL", (f"the false-alarm rate at {head} h is {far:.2%}, "
                        f"above the {budget:.0%} the system promises")
    bad = [m for m in ("rmse", "onset_recall")
           if at_h.loc[m, "verdict"] == "degraded"]
    if bad:
        return "PARTIAL", (
            f"the {budget:.0%} false-alarm promise holds ({far:.2%} at "
            f"{head} h), but " + " and ".join(bad)
            + (" at " + str(head) + " h "
               + ("fall" if len(bad) > 1 else "falls")
               + " outside the 2025 interval"))
    return "PASS", (f"the false-alarm rate at {head} h is {far:.2%}, inside "
                    f"the {budget:.0%} promise, and neither RMSE nor onset "
                    f"recall at {head} h is outside the 2025 interval")


# ---------------------------------------------------------------------------
# The live-data penalty
# ---------------------------------------------------------------------------
def penalty(parallel=True, batch_size=None):
    """
    The same hours scored twice: ERA5 weather against the forecast API's
    recent past. Water data is identical in both, so the difference is the
    price of not waiting five days for a reanalysis.
    """
    out_dir = D.ensure_dir(D.TEST26["penalty_dir"])
    frames, scored = {}, {}
    for which in ("era5", "forecast"):
        fpath = D.dpath(D.DATA["features"], mode=f"penalty_{which}")
        hpath = D.dpath(D.DATA["hourly"], mode=f"penalty_{which}")
        if not (os.path.exists(fpath) and os.path.exists(hpath)):
            raise SystemExit(
                f"{D.shortpath(fpath)} is missing - "
                f"run `python -m src.deploy.fetch_live --mode penalty` first")
        frames[which] = (pd.read_parquet(fpath), pd.read_parquet(hpath))

    # only the hours BOTH versions have, so the comparison is hour for hour
    keysets = [set(zip(f["station"], f["time"])) for f, _ in frames.values()]
    common = keysets[0] & keysets[1]
    D.log(f"\n  live-data penalty on {len(common):,} shared station-hours "
          f"({D.TEST26['penalty_days']} days)")

    for which, (feat, panel) in frames.items():
        keep = [k in common for k in zip(feat["station"], feat["time"])]
        feat = feat[keep].reset_index(drop=True)
        D.log(f"\n  --- {which} weather ---")
        rows, preds = predict(feat, panel, batch_size)
        y_reg, y_cls, class_now = truths(rows)
        reg, cls, saved = score(preds, y_reg, y_cls, class_now,
                                label=f"penalty_{which}")
        reg.to_csv(os.path.join(out_dir, f"metrics_regression_{which}.csv"),
                   index=False)
        cls.to_csv(os.path.join(out_dir,
                                f"metrics_classification_{which}.csv"),
                   index=False)
        scored[which] = {"rows": rows, "reg": reg, "cls": cls, "saved": saved,
                         "y_reg": y_reg, "y_cls": y_cls,
                         "class_now": class_now}

    a, b = scored["era5"], scored["forecast"]
    same = (a["rows"][["station", "time"]].reset_index(drop=True)
            .equals(b["rows"][["station", "time"]].reset_index(drop=True)))
    rows = []
    for h in HORIZONS:
        ra = a["reg"][(a["reg"]["method"] == FINAL)
                      & (a["reg"]["horizon"] == h)].iloc[0]
        rb = b["reg"][(b["reg"]["method"] == FINAL)
                      & (b["reg"]["horizon"] == h)].iloc[0]
        ca = a["cls"][(a["cls"]["method"] == FINAL)
                      & (a["cls"]["horizon"] == h)
                      & (a["cls"]["rule"] == "warn")].iloc[0]
        cb = b["cls"][(b["cls"]["method"] == FINAL)
                      & (b["cls"]["horizon"] == h)
                      & (b["cls"]["rule"] == "warn")].iloc[0]
        row = {"horizon": h, "rows": int(ra["n"]), "aligned": bool(same),
               "rmse_era5": ra["rmse"], "rmse_forecast": rb["rmse"],
               "rmse_diff": rb["rmse"] - ra["rmse"],
               "f1_era5": ca["f1_macro"], "f1_forecast": cb["f1_macro"],
               "f1_diff": cb["f1_macro"] - ca["f1_macro"],
               "onset_era5": ca["onset_recall"],
               "onset_forecast": cb["onset_recall"],
               "onset_diff": cb["onset_recall"] - ca["onset_recall"],
               "fa_era5": ca["false_alarm_rate"],
               "fa_forecast": cb["false_alarm_rate"],
               "fa_diff": cb["false_alarm_rate"] - ca["false_alarm_rate"]}
        if same:
            res = B.bootstrap(
                a["rows"][["station", "time"]], a["y_reg"][h], a["y_cls"][h],
                a["class_now"],
                {"reg": b["saved"][(FINAL, h)]["reg"],
                 "cls": b["saved"][(FINAL, h)]["warn"]},
                {"reg": a["saved"][(FINAL, h)]["reg"],
                 "cls": a["saved"][(FINAL, h)]["warn"]}, parallel=parallel)
            for m in ("rmse", "f1_macro", "onset_recall", "false_alarm_rate"):
                row[f"{m}_ci_lo"] = res[m]["lo"]
                row[f"{m}_ci_hi"] = res[m]["hi"]
        rows.append(row)
    out = pd.DataFrame(rows)
    path = os.path.join(out_dir, "penalty.csv")
    out.to_csv(path, index=False)
    D.log(f"\n  wrote {D.shortpath(path)}")
    D.log("\nLIVE-DATA PENALTY (forecast API minus ERA5, same hours)")
    D.log(f"  {'h':>4}{'RMSE era5':>12}{'RMSE fc':>10}{'diff':>9}"
          f"{'onset era5':>12}{'onset fc':>10}{'diff':>9}"
          f"{'FA era5':>10}{'FA fc':>9}")
    for r in out.itertuples():
        D.log(f"  {r.horizon:>4}{r.rmse_era5:>12.4f}{r.rmse_forecast:>10.4f}"
              f"{r.rmse_diff:>+9.4f}{r.onset_era5:>12.3f}"
              f"{r.onset_forecast:>10.3f}{r.onset_diff:>+9.3f}"
              f"{r.fa_era5:>10.4f}{r.fa_forecast:>9.4f}")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--penalty", action="store_true",
                    help="only the ERA5 vs forecast-API comparison")
    ap.add_argument("--serial-bootstrap", action="store_true",
                    help="run the bootstrap in this process (for the test "
                         "that it gives the same answer)")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    st = D.Stages()
    D.log("=== STEP 12.3: the deployed models on 2026 ===")
    D.report_device("deep inference")
    if not D.require_cuda_or_explain() and not args.allow_cpu:
        raise SystemExit(2)
    parallel = not args.serial_bootstrap

    if args.penalty:
        with st("penalty"):
            penalty(parallel, args.batch_size)
        st.report()
        return

    fpath = D.dpath(D.DATA["features"], mode="history")
    hpath = D.dpath(D.DATA["hourly"], mode="history")
    if not os.path.exists(fpath):
        raise SystemExit("run `python -m src.deploy.fetch_live --mode "
                         "history` first")
    features = pd.read_parquet(fpath)
    panel = pd.read_parquet(hpath)
    check_plan(features["time"].max())

    out_dir = D.ensure_dir(RESULTS)
    with st("predict"):
        rows, preds = predict(features, panel, args.batch_size)
    y_reg, y_cls, class_now = truths(rows)
    keys = rows[["station", "time"]].reset_index(drop=True)

    with st("score"):
        reg, cls, saved = score(preds, y_reg, y_cls, class_now)
        reg.to_csv(os.path.join(out_dir, "metrics_regression.csv"),
                   index=False)
        cls.to_csv(os.path.join(out_dir, "metrics_classification.csv"),
                   index=False)
        ps = per_station(keys, y_reg, y_cls, class_now, saved)
        ps.to_csv(os.path.join(out_dir, "per_station.csv"), index=False)

    with st("predictions"):
        for (name, h), m in saved.items():
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
            out.to_parquet(os.path.join(out_dir, f"pred_{name}_h{h}.parquet"),
                           index=False)

    with st("lead times"):
        head = 24 if 24 in HORIZONS else HORIZONS[len(HORIZONS) // 2]
        warn = saved[(FINAL, head)]["warn"] == CLASS_NAMES.index("Low")
        lead = lead_times(keys, rows[DO].to_numpy(dtype="float64"),
                          warn.astype(int), h=head)
        lead.to_csv(os.path.join(out_dir, "lead_times.csv"), index=False)

    with st("bootstrap 2026"):
        frames = []
        for h in HORIZONS:
            res = B.bootstrap(keys, y_reg[h], y_cls[h], class_now,
                              {"reg": saved[(FINAL, h)]["reg"],
                               "cls": saved[(FINAL, h)]["warn"]},
                              parallel=parallel)
            frames.append(B.intervals_frame("live_2026", h, res))
        boot26 = pd.concat(frames, ignore_index=True)
    with st("bootstrap 2025 reference"):
        ref = reference_2025(parallel)
    pd.concat([ref, boot26], ignore_index=True).to_csv(
        os.path.join(out_dir, "bootstrap.csv"), index=False)

    cmp_table = compare(ref, reg, cls)
    cmp_table.to_csv(os.path.join(out_dir, "compare_2025_2026.csv"),
                     index=False)
    word, why = overall_verdict(cmp_table)

    # which stations did not make it, and why
    skipped = pd.read_csv(D.dpath(D.DATA["skipped"], mode="history"),
                          dtype={"station": str}) \
        if os.path.exists(D.dpath(D.DATA["skipped"], mode="history")) \
        else pd.DataFrame(columns=["station", "stage", "reason"])
    all_sites = set(D.stations_table()["site_no"].astype(str))
    scored_sites = set(rows["station"].astype(str))
    for s in sorted(all_sites - scored_sites):
        if s not in set(skipped["station"].astype(str)):
            skipped = pd.concat([skipped, pd.DataFrame(
                [{"station": s, "stage": "score",
                  "reason": "no scoreable rows in the 2026 window"}])],
                ignore_index=True)
    skipped.to_csv(os.path.join(out_dir, "skipped.csv"), index=False)

    summary = {
        "window": [str(rows["time"].min()), str(rows["time"].max())],
        "rows": int(len(rows)),
        "stations_scored": sorted(scored_sites),
        "stations_skipped": skipped.to_dict("records"),
        "model_version": D.model_version(),
        "model_folder": D.DEP["models"]["stage"],
        "verdict": word, "verdict_reason": why,
        "events": int(len(lead)),
        "warned": int(lead["warned"].sum()),
        "median_lead_hours": (
            float(lead.loc[lead["warned"], "lead_hours"].median())
            if lead["warned"].any() else None),
        "timings_seconds": {k: round(v, 1) for k, v in st.times.items()},
    }
    with open(os.path.join(out_dir, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    terminal(reg, cls, cmp_table, lead, skipped, word, why)
    st.report()
    D.log(f"\nWrote {RESULTS}")
    D.log("Next: python -m src.deploy.score_2026 --penalty, then "
          "python -m src.deploy.walk_forward, then "
          "python -m src.evaluation.final_report")


def terminal(reg, cls, cmp_table, lead, skipped, word, why):
    def r(m, h, c):
        row = reg[(reg["method"] == m) & (reg["horizon"] == h)]
        return row[c].iloc[0] if len(row) else np.nan

    def c(m, h, rule, col):
        row = cls[(cls["method"] == m) & (cls["horizon"] == h)
                  & (cls["rule"] == rule)]
        return row[col].iloc[0] if len(row) else np.nan

    D.log("\n2026, the deployed models (stacking + the frozen warning rule)")
    D.log(f"  {'h':>4}{'R2':>9}{'RMSE':>9}{'skill':>8}{'macroF1':>9}"
          f"{'LowRec':>8}{'onset':>8}{'falseAl':>9}")
    for h in HORIZONS:
        D.log(f"  {h:>4}{r(FINAL, h, 'r2'):>9.4f}{r(FINAL, h, 'rmse'):>9.4f}"
              f"{r(FINAL, h, 'skill_vs_persistence'):>8.3f}"
              f"{c(FINAL, h, 'warn', 'f1_macro'):>9.4f}"
              f"{c(FINAL, h, 'warn', 'recall_Low'):>8.3f}"
              f"{c(FINAL, h, 'warn', 'onset_recall'):>8.3f}"
              f"{c(FINAL, h, 'warn', 'false_alarm_rate'):>9.4f}")
    D.log("\n2026 against the 2025 95% intervals (pre-registered verdicts)")
    D.log(f"  {'h':>4}  {'metric':<18}{'2025':>9}{'2025 95% CI':>22}"
          f"{'2026':>9}{'change':>10}  verdict")
    for row in cmp_table.itertuples():
        D.log(f"  {row.horizon:>4}  {row.metric:<18}{row.value_2025:>9.4f}"
              f"  [{row.ci_lo_2025:>8.4f},{row.ci_hi_2025:>8.4f}]"
              f"{row.value_2026:>9.4f}{row.change:>+10.4f}  {row.verdict}")
    if len(lead):
        w = lead["warned"]
        D.log(f"\nLEAD TIME 2026: {int(w.sum())}/{len(lead)} events warned "
              f"({w.mean():.1%})"
              + (f", median {lead.loc[w, 'lead_hours'].median():.0f} h"
                 if w.any() else ""))
    if len(skipped):
        D.log(f"\nSTATIONS NOT SCORED ({len(skipped)}):")
        for s in skipped.itertuples():
            D.log(f"  {s.station}  ({s.stage}) {s.reason}")
    D.log(f"\nVERDICT against plan_2026.md: {word}")
    D.log(f"  {why}")


if __name__ == "__main__":
    main()
