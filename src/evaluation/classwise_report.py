"""
Class-wise report for the upgraded ensemble, on the 2025 test year.

Run from the project root (after src.models.ensemble_v21):

    python -m src.evaluation.classwise_report
    python -m src.evaluation.classwise_report --horizon 12   # one table only

Nothing is fitted, predicted or re-scored here. It reads the rows
src/models/ensemble_v21.py already wrote to

    results/test_v21/metrics_classification_stage2.csv

keeps the `argmax` rule at the four horizons, and breaks the macro numbers
back out per class - Low, Medium, Good - for ensemble_v21 and for the best
single model, so the question "where does the ensemble actually win" has an
answer that is not an average over three very unequal classes.

Support comes from the saved confusion matrix: the column holds nine counts
separated by ";", row-major, rows = true class and columns = predicted
class, in the order Low / Medium / Good, so a row sum is how many hours of
2025 really were that class. The per-class precision and recall in the csv
are checked against the same matrix before anything is printed, which is
what makes a mistake about that orientation impossible to miss.

Writes:
    results/test_v21/classwise_metrics.csv
    results/test_v21/plots/confusion_ensemble_v21_h<h>.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config, project_path

CFG = load_config()
HORIZONS = CFG["horizons"]
CLASS_NAMES = CFG["target"]["class_names"]          # Low, Medium, Good
N_CLASSES = len(CLASS_NAMES)
RULE = "argmax"

METHOD = "ensemble_v21"
BEST_SINGLE = "lightgbm"
METHODS = [METHOD, BEST_SINGLE]

RESULTS = "results/test_v21"
SOURCE = "results/test_v21/metrics_classification_stage2.csv"
PLOTS = "results/test_v21/plots"
OUT_CSV = "results/test_v21/classwise_metrics.csv"


# ---------------------------------------------------------------------------
# Reading the rows, and the confusion matrix out of its one cell
# ---------------------------------------------------------------------------
def load_rows(path=None):
    """The argmax rows at the project's horizons, for the two methods."""
    path = project_path(path or SOURCE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{SOURCE} does not exist - run "
            f"`python -m src.models.ensemble_v21` first")
    d = pd.read_csv(path)
    d = d[(d["rule"] == RULE) & d["horizon"].isin(HORIZONS)
          & d["method"].isin(METHODS)]
    missing = [(m, h) for m in METHODS for h in HORIZONS
               if not len(d[(d["method"] == m) & (d["horizon"] == h)])]
    assert not missing, f"{SOURCE} has no {RULE} row for {missing}"
    return d


def confusion(row):
    """The 3x3 matrix back out of the `confusion_matrix` cell."""
    counts = [int(float(x)) for x in str(row["confusion_matrix"]).split(";")]
    assert len(counts) == N_CLASSES * N_CLASSES, \
        (f"expected {N_CLASSES * N_CLASSES} numbers in confusion_matrix, "
         f"got {len(counts)}")
    return np.array(counts, dtype="int64").reshape(N_CLASSES, N_CLASSES)


def classwise(cm):
    """
    precision, recall, f1 and support per class, from the matrix alone.

    Rows are the true class, so a row sum is the support and the diagonal
    over the row is recall; columns are the predicted class, so the diagonal
    over the column is precision. A class a model never predicts scores 0
    rather than raising, which is the zero_division=0 the project's own
    metrics use.
    """
    tp = np.diag(cm).astype("float64")
    support = cm.sum(axis=1).astype("int64")
    predicted = cm.sum(axis=0).astype("float64")
    precision = np.divide(tp, predicted, out=np.zeros_like(tp),
                          where=predicted > 0)
    recall = np.divide(tp, support.astype("float64"),
                       out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp),
                   where=denom > 0)
    return precision, recall, f1, support


def check_against_the_csv(row, precision, recall, f1):
    """
    The matrix and the per-class columns in the same row must agree. If they
    do not, the matrix is being read in the wrong orientation and every
    number below would be quietly wrong.
    """
    for i, name in enumerate(CLASS_NAMES):
        for metric, got in [("precision", precision[i]), ("recall", recall[i]),
                            ("f1", f1[i])]:
            want = float(row[f"{metric}_{name}"])
            assert abs(got - want) < 1e-6, \
                (f"{row['method']} h{row['horizon']}: {metric}_{name} from "
                 f"the confusion matrix is {got:.6f}, but the csv says "
                 f"{want:.6f} - the matrix is not rows=true, cols=predicted")
    n = int(row["n"])
    assert int(precision.size) == N_CLASSES
    return n


def build(rows):
    """One tidy frame: a row per method, horizon and class."""
    out = []
    for method in METHODS:
        for h in HORIZONS:
            row = rows[(rows["method"] == method)
                       & (rows["horizon"] == h)].iloc[0]
            cm = confusion(row)
            precision, recall, f1, support = classwise(cm)
            n = check_against_the_csv(row, precision, recall, f1)
            assert int(support.sum()) == n, \
                (f"{method} h{h}: the confusion matrix holds "
                 f"{support.sum():,} rows but the csv counts {n:,}")
            for i, name in enumerate(CLASS_NAMES):
                out.append({
                    "method": method, "horizon": h, "rule": RULE,
                    "class": name,
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                    "support_share": float(support[i] / n) if n else np.nan,
                    "n": n,
                    "f1_macro": float(row["f1_macro"]),
                    "tau": (None if pd.isna(row["tau"])
                            else float(row["tau"])),
                    "confusion_row": ";".join(str(int(x)) for x in cm[i]),
                })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------
def table(frame, h):
    """The class-wise table at one horizon, as text."""
    lines = [f"\n2025 test year, {h} h ahead, rule = {RULE}",
             f"{'method':<19}{'class':<9}{'precision':>11}{'recall':>9}"
             f"{'f1':>9}{'support':>11}{'share':>8}"]
    lines.append("-" * len(lines[-1]))
    for method in METHODS:
        sub = frame[(frame["method"] == method) & (frame["horizon"] == h)]
        label = (method if method != BEST_SINGLE
                 else f"{method} (single)")
        # `class` is a keyword, so itertuples would rename it to a position
        for i, (_, r) in enumerate(sub.iterrows()):
            lines.append(f"{label if i == 0 else '':<19}{r['class']:<9}"
                         f"{r['precision']:>11.4f}{r['recall']:>9.4f}"
                         f"{r['f1']:>9.4f}{r['support']:>11,}"
                         f"{r['support_share']:>8.1%}")
        macro = sub["f1"].mean()
        lines.append(f"{'':<19}{'macro':<9}{sub['precision'].mean():>11.4f}"
                     f"{sub['recall'].mean():>9.4f}{macro:>9.4f}"
                     f"{sub['support'].sum():>11,}{1.0:>8.1%}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The heatmaps
# ---------------------------------------------------------------------------
def plot_confusion(rows, h, folder=None):
    """
    A 3x3 heatmap for ensemble_v21 at one horizon: the count and the row
    percentage in each cell, shaded by the row percentage so the diagonal of
    a good model reads as the dark band regardless of how rare a class is.
    """
    folder = project_path(folder or PLOTS)
    os.makedirs(folder, exist_ok=True)
    row = rows[(rows["method"] == METHOD) & (rows["horizon"] == h)].iloc[0]
    cm = confusion(row)
    share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{cm[i, j]:,}\n{100 * share[i, j]:.1f}%",
                    ha="center", va="center", fontsize=9,
                    color="white" if share[i, j] > .5 else "black")
    ax.set_xticks(range(N_CLASSES), CLASS_NAMES)
    ax.set_yticks(range(N_CLASSES), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Ensemble v21 - {h}h ahead")
    fig.tight_layout()
    path = os.path.join(folder, f"confusion_{METHOD}_h{h}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--horizon", type=int, choices=HORIZONS,
                    help="print one horizon instead of all four")
    args = ap.parse_args()

    rows = load_rows()
    frame = build(rows)
    os.makedirs(project_path(RESULTS), exist_ok=True)
    out = project_path(OUT_CSV)
    frame.to_csv(out, index=False)

    print(f"class-wise 2025 results for {METHOD} and {BEST_SINGLE} "
          f"(best single), rule = {RULE}")
    print(f"  read   {SOURCE}")
    print(f"  wrote  {OUT_CSV}  ({len(frame)} rows)")
    for h in HORIZONS:
        path = plot_confusion(rows, h)
        print(f"  wrote  {os.path.relpath(path, project_path()).replace(os.sep, '/')}")

    for h in ([args.horizon] if args.horizon else HORIZONS):
        print(table(frame, h))


if __name__ == "__main__":
    main()
