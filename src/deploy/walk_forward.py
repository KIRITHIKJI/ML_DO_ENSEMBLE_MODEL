"""
STEP 12.4 - the walk-forward table: more than one test year.

    python -m src.deploy.walk_forward

One table, all four horizons, three expanding windows:

    train 2023        -> test 2024    (NOT CLEAN: 2024 was the tuning year)
    train 2023-2024   -> test 2025    (the frozen result)
    train 2023-2025   -> test 2026    (the deployed models, from 12.3)

Nothing is trained, tuned or re-scored here. Every number is read out of a
file some earlier step already wrote:

    2024   results/val/metrics_step7_{regression,classification}.csv
           the blocked cross-fitted out-of-fold predictions from step 7
    2025   results/test/metrics_*_stage2.csv
           the one and only 2025 test
    2026   results/live_2026/metrics_*.csv
           step 12.3

The 2024 row carries its warning permanently. Those numbers were produced
on the year every hyperparameter, the ensemble and the warning thresholds
were chosen on, so they are an in-sample upper bound, not a test result.
They are in the table because leaving a year out of a walk-forward table is
how a walk-forward table stops being one - not because they are comparable
with the other two rows.
"""

import argparse
import os

import pandas as pd

from src.config import project_path
from src.evaluation.step3_report import fmt, md_table

from src.deploy import common as D

CFG = D.CFG
HORIZONS = D.HORIZONS
FINAL = D.FINAL_SYSTEM
WF = D.DEP["walk_forward"]

SOURCES = {
    "oof_2024": ("results/val", "metrics_step7_regression.csv",
                 "metrics_step7_classification.csv"),
    "test_2025": (CFG["final_test"]["results_dir"],
                  "metrics_regression_stage2.csv",
                  "metrics_classification_stage2.csv"),
    "live_2026": (D.TEST26["results_dir"], "metrics_regression.csv",
                  "metrics_classification.csv"),
}


def read(source):
    """(regression, classification) for one source, or (None, None)."""
    folder, rfile, cfile = SOURCES[source]
    rp = project_path(folder, rfile)
    cp = project_path(folder, cfile)
    if not (os.path.exists(rp) and os.path.exists(cp)):
        return None, None
    return pd.read_csv(rp), pd.read_csv(cp)


def pick(df, h, col, rule=None):
    if df is None:
        return float("nan")
    m = (df["method"] == FINAL) & (df["horizon"] == h)
    if rule is not None:
        m &= df["rule"] == rule
    row = df[m]
    return float(row[col].iloc[0]) if len(row) else float("nan")


def table():
    """One row per (window, horizon)."""
    rows = []
    for spec in WF["rows"]:
        reg, cls = read(spec["source"])
        for h in HORIZONS:
            rows.append({
                "train": spec["train"], "test": spec["test"],
                "source": spec["source"], "note": spec["note"],
                "horizon": h,
                "n": pick(reg, h, "n"),
                "r2": pick(reg, h, "r2"),
                "rmse": pick(reg, h, "rmse"),
                "skill_vs_persistence": pick(reg, h, "skill_vs_persistence"),
                "f1_macro": pick(cls, h, "f1_macro", "warn"),
                "recall_Low": pick(cls, h, "recall_Low", "warn"),
                "onset_recall": pick(cls, h, "onset_recall", "warn"),
                "false_alarm_rate": pick(cls, h, "false_alarm_rate", "warn"),
                "available": reg is not None,
            })
    return pd.DataFrame(rows)


def markdown(df):
    """The table as markdown, one line per window and horizon."""
    head = ["train", "test", "horizon", "rows", "R2", "RMSE",
            "skill vs no-change", "macro F1", "Low recall", "onset recall",
            "false alarms", "note"]
    body = []
    for r in df.itertuples():
        if not r.available:
            body.append([r.train, r.test, f"{r.horizon} h"]
                        + ["-"] * 8 + ["not scored yet"])
            continue
        body.append([
            r.train, r.test, f"{r.horizon} h",
            f"{int(r.n):,}" if r.n == r.n else "-",
            fmt(r.r2), fmt(r.rmse),
            f"{r.skill_vs_persistence:+.3f}",
            f"{r.f1_macro:.4f}", f"{r.recall_Low:.3f}",
            f"{r.onset_recall:.3f}", f"{r.false_alarm_rate:.2%}",
            r.note])
    return md_table(head, body)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()

    df = table()
    path = project_path(WF["out_csv"])
    if not args.print_only:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        D.log(f"Wrote {D.shortpath(path)}")

    D.log("\nWALK-FORWARD: the same frozen system, three expanding windows\n")
    D.log(f"  {'train':<12}{'test':<7}{'h':>4}{'rows':>10}{'RMSE':>9}"
          f"{'macroF1':>9}{'onset':>8}{'falseAl':>9}  note")
    for r in df.itertuples():
        if not r.available:
            D.log(f"  {r.train:<12}{r.test:<7}{r.horizon:>4}"
                  f"{'-':>10}{'-':>9}{'-':>9}{'-':>8}{'-':>9}  "
                  f"not scored yet")
            continue
        D.log(f"  {r.train:<12}{r.test:<7}{r.horizon:>4}{int(r.n):>10,}"
              f"{r.rmse:>9.4f}{r.f1_macro:>9.4f}{r.onset_recall:>8.3f}"
              f"{r.false_alarm_rate:>9.4f}  {r.note}")
    return df


if __name__ == "__main__":
    main()
