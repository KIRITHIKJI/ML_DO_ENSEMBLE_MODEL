"""
STEP 7.1 - Blocked cross-fitting over 2024, with a purge.

Why this replaces the val-A / val-B split for step 7: several methods now
learn something from 2024 (stacking's coefficients, the CAE meta-model,
the warning threshold, ATAE's four settings). Scoring any of them on rows
they were fitted on flatters them, and a single half-year split wastes half
the data and gives a noisy answer.

So 2024 is cut into six two-month folds. Each fold is predicted by a model
fitted on the other five, and the scores are computed on the six held-out
pieces stitched back together - every row of 2024 is predicted exactly
once, by a model that never saw it.

The purge: hours near a fold boundary are dropped from the TRAINING side,
never from the held-out side. Two things leak across a boundary otherwise:

  * a target 48 hours ahead, which overlaps the next fold, and
  * the 336-hour (14-day) error window the ensembles look back over.

14 days on both sides covers both, so a training row's inputs and target
can never touch the fold being predicted.

Nothing here reads 2025.
"""

import numpy as np
import pandas as pd

from src.config import load_config

CFG = load_config()
CV = CFG["crossfit"]


def fold_of(times):
    """
    Which two-month fold each timestamp belongs to: 0 = Jan-Feb, 1 = Mar-Apr,
    ... 5 = Nov-Dec. Anything outside the validation year gets -1.
    """
    t = pd.to_datetime(pd.Series(np.asarray(times)).reset_index(drop=True))
    year = CFG["split"]["val_years"][0]
    months_per_fold = int(CV["months_per_fold"])
    fold = ((t.dt.month - 1) // months_per_fold).to_numpy()
    fold = np.where(t.dt.year.to_numpy() == year, fold, -1)
    return fold.astype(int)


def fold_bounds(fold_id):
    """First and last instant of a fold, as timestamps."""
    year = CFG["split"]["val_years"][0]
    m = int(CV["months_per_fold"])
    start = pd.Timestamp(year=year, month=fold_id * m + 1, day=1)
    end = start + pd.DateOffset(months=m) - pd.Timedelta(seconds=1)
    return start, end


def split_indices(times, fold_id):
    """
    (train_rows, test_rows) for one fold.

    test  = the fold itself, untouched.
    train = everything else in 2024, minus a purge of `purge_days` days on
            both sides of the fold.
    """
    t = pd.to_datetime(pd.Series(np.asarray(times)).reset_index(drop=True))
    folds = fold_of(times)
    start, end = fold_bounds(fold_id)
    purge = pd.Timedelta(days=int(CV["purge_days"]))

    test = np.flatnonzero(folds == fold_id)
    in_year = folds >= 0
    too_close = ((t >= start - purge) & (t <= end + purge)).to_numpy()
    train = np.flatnonzero(in_year & ~too_close)
    return train, test


def all_folds():
    return list(range(int(CV["n_folds"])))


def inner_split(times, train_rows, holdout_fraction=None):
    """
    A time-ordered inner split of the training rows, for early stopping.
    The LAST `holdout_fraction` of the training period is held out, so the
    model is stopped on data that comes after what it learned from - never
    on a random scatter of hours interleaved with training ones.
    """
    frac = CV["inner_holdout"] if holdout_fraction is None else holdout_fraction
    t = pd.to_datetime(pd.Series(np.asarray(times)).reset_index(drop=True))
    sub = t.iloc[train_rows]
    cutoff = sub.quantile(1 - frac)
    inner_train = train_rows[(sub <= cutoff).to_numpy()]
    inner_valid = train_rows[(sub > cutoff).to_numpy()]
    if len(inner_valid) == 0:                    # degenerate: keep it working
        inner_train, inner_valid = train_rows[:-1], train_rows[-1:]
    return inner_train, inner_valid


def describe(times):
    """A small table of the folds, for the report and for sanity."""
    folds = fold_of(times)
    rows = []
    for k in all_folds():
        train, test = split_indices(times, k)
        start, end = fold_bounds(k)
        rows.append({"fold": k,
                     "period": f"{start:%b}-{end:%b}",
                     "test_rows": len(test),
                     "train_rows": len(train),
                     "purged_rows": int((folds >= 0).sum()) - len(train) - len(test)})
    return pd.DataFrame(rows)
