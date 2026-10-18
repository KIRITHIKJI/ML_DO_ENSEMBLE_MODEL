"""
STEP 12 - station-week bootstrap confidence intervals, in parallel.

The blocks, the metrics and the resample count are the project's own: the
blocks come from `step6_report.station_week_blocks`, RMSE and macro F1 from
`step6_report.fast_rmse` / `fast_macro_f1`, onset recall and false alarms
from `metrics.early_warning_metrics`, and `n_resamples` from
config.yaml (`ensemble.bootstrap`). Nothing is reimplemented here.

What is new is only HOW the work is divided. Each resample draws its block
picks from its own generator, spawned from one seed sequence:

    children = np.random.SeedSequence(seed).spawn(n_resamples)

so resample i is the same draw whether it is computed on this core or
another one. `parallel=False` therefore gives bit-identical intervals to
`parallel=True`, which `tests/test_deploy.py` checks.
"""

import numpy as np
import pandas as pd

from src.evaluation.metrics import early_warning_metrics
from src.evaluation.step6_report import (fast_macro_f1, fast_rmse,
                                         station_week_blocks)

from src.deploy import common as D

METRICS = ["rmse", "f1_macro", "onset_recall", "false_alarm_rate"]
BOOT_BATCH = int(D.BOOT["batch_size"])


def n_resamples():
    return int(D.CFG["ensemble"]["bootstrap"]["n_resamples"])


def _blocks(keys):
    return station_week_blocks(keys, np.ones(len(keys), dtype=bool))


def _one(idx, y_reg, y_cls, class_now, reg, cls, ok_r, ok_c):
    """The four metrics on one resample's rows."""
    r = idx[ok_r[idx]]
    c = idx[ok_c[idx]]
    out = {}
    out["rmse"] = fast_rmse(y_reg[r], reg[r]) if len(r) else np.nan
    if len(c):
        yc = y_cls[c].astype(int)
        pc = cls[c].astype(int)
        out["f1_macro"] = fast_macro_f1(yc, pc)
        ew = early_warning_metrics(class_now[c], yc, pc)
        out["onset_recall"] = ew["onset_recall"]
        out["false_alarm_rate"] = ew["false_alarm_rate"]
    else:
        out["f1_macro"] = out["onset_recall"] = np.nan
        out["false_alarm_rate"] = np.nan
    return out


def _batch(children, blocks, y_reg, y_cls, class_now, a, b, ok_r, ok_c):
    """
    One parallel task: several resamples. Returns a list of dicts. When `b`
    is given every metric is (a minus b) on the same resampled rows, which
    is the paired comparison the project uses everywhere else.
    """
    n_blocks = len(blocks)
    rows = []
    for child in children:
        rng = np.random.default_rng(child)
        pick = rng.integers(0, n_blocks, n_blocks)
        idx = np.concatenate([blocks[i] for i in pick])
        va = _one(idx, y_reg, y_cls, class_now, a["reg"], a["cls"], ok_r, ok_c)
        if b is None:
            rows.append(va)
        else:
            vb = _one(idx, y_reg, y_cls, class_now, b["reg"], b["cls"],
                      ok_r, ok_c)
            rows.append({k: va[k] - vb[k] for k in va})
    return rows


def _summarise(rows):
    out = {}
    for k in METRICS:
        v = np.asarray([r[k] for r in rows], dtype="float64")
        v = v[~np.isnan(v)]
        if not len(v):
            out[k] = {"mean": np.nan, "lo": np.nan, "hi": np.nan,
                      "significant": False, "n": 0}
            continue
        out[k] = {"mean": float(v.mean()),
                  "lo": float(np.percentile(v, 2.5)),
                  "hi": float(np.percentile(v, 97.5)),
                  "n": int(len(v))}
        out[k]["significant"] = bool(out[k]["lo"] > 0 or out[k]["hi"] < 0)
    return out


def bootstrap(keys, y_reg, y_cls, class_now, a, b=None, n=None, seed=None,
              parallel=True, n_jobs=None, batch_size=None):
    """
    95% station-week bootstrap intervals.

    keys        DataFrame with `station` and `time`, one row per prediction
    y_reg/y_cls the truths; NaN rows are skipped per metric
    class_now   the class at time t, for the early-warning metrics
    a, b        dicts {"reg": array, "cls": array}. With `b`, the interval
                is for (a minus b) on the same resampled rows.
    parallel    False runs the same resamples in this process; the answer is
                identical, which is the point of spawning one generator per
                resample.
    """
    n = n_resamples() if n is None else int(n)
    seed = D.SEED if seed is None else int(seed)
    blocks = _blocks(keys)
    if not blocks:
        return _summarise([])

    y_reg = np.asarray(y_reg, dtype="float64")
    y_cls = np.asarray(y_cls, dtype="float64")
    class_now = np.asarray(class_now, dtype="float64")
    ok_r = ~np.isnan(y_reg) & ~np.isnan(np.asarray(a["reg"], dtype="float64"))
    if b is not None:
        ok_r &= ~np.isnan(np.asarray(b["reg"], dtype="float64"))
    ok_c = ~np.isnan(y_cls) & ~np.isnan(class_now)

    a = {"reg": np.asarray(a["reg"], dtype="float64"),
         "cls": np.asarray(a["cls"], dtype="float64")}
    if b is not None:
        b = {"reg": np.asarray(b["reg"], dtype="float64"),
             "cls": np.asarray(b["cls"], dtype="float64")}

    children = np.random.SeedSequence(seed).spawn(n)
    size = int(batch_size or BOOT_BATCH)
    chunks = [children[i:i + size] for i in range(0, n, size)]

    if not parallel or len(chunks) == 1:
        rows = []
        for ch in chunks:
            rows += _batch(ch, blocks, y_reg, y_cls, class_now, a, b,
                           ok_r, ok_c)
        return _summarise(rows)

    from joblib import Parallel, delayed
    jobs = n_jobs if n_jobs is not None else int(D.BOOT["n_jobs"])
    parts = Parallel(n_jobs=jobs, backend=D.BOOT.get("backend", "loky"))(
        delayed(_batch)(ch, blocks, y_reg, y_cls, class_now, a, b, ok_r, ok_c)
        for ch in chunks)
    rows = [r for part in parts for r in part]
    return _summarise(rows)


def intervals_frame(label, h, res):
    """One tidy row per metric, for a csv."""
    return pd.DataFrame([{"label": label, "horizon": h, "metric": m,
                          **res[m]} for m in METRICS])
