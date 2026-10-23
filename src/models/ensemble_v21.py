"""
ensemble v21 - Caruana greedy ensemble selection, with replacement.

Run from the project root:

    python -m src.models.ensemble_v21            # select, score 2025, report
    python -m src.models.ensemble_v21 --quick    # 5 iterations, no bootstrap

Nothing is trained here and nothing already on disk is rewritten. The five
members (random_forest, xgboost, lightgbm, bilstm, cnn_lstm) already wrote
their 2024 validation predictions to results/val/predictions/ and their 2025
test predictions to results/test/pred_<member>_h<h>.parquet; this module only
reads them, decides how to weight them, and scores the result.

The algorithm, per task and per horizon:

  1. Start from the single member with the best 2024 score.
  2. For up to 30 further iterations, try adding each member - repeats
     allowed, which is what "with replacement" means - and keep the one that
     improves the 2024 score the most. Stop the moment no addition improves
     it. Because the first step is the best single member and an addition is
     only ever accepted when it strictly improves, the guarantee holds: if no
     blend beats the best single member, the weights come out as 100% that
     member.
  3. The weights are the selection counts divided by their total. Selecting a
     member twice is simply how a greedy search with a fixed step size
     reaches non-uniform weights.

The score, per task:

  regression       the negative RMSE on 2024 (so higher is better here too),
                   over the rows whose target exists.
  classification   macro F1 on 2024 of the WARNING rule, after tuning the
                   Low-class threshold tau on 2024 under exactly the rule the
                   rest of the project uses (src.models.freeze.choose_tau_oof:
                   the tau catching the most onsets inside the tightened
                   false-alarm budget, ties broken towards fewer false alarms
                   and then the higher threshold). The tau is part of the
                   candidate, so a blend is judged the way it would be used.

Only 2024 is used to choose the weights and tau. 2025 is read once, at the
end, to be scored - never to select anything.

Writes:
    models/ensemble_v21/weights.json          the weights, taus and the trace
    results/test_v21/metrics_regression_stage2.csv
    results/test_v21/metrics_classification_stage2.csv
        the 2025 metrics in exactly the format of results/test/, with every
        existing method's rows copied across unchanged for comparison.
"""

import json
import os

import numpy as np
import pandas as pd

from src.config import load_config, project_path

CFG = load_config()
FT = CFG["final_test"]
ENS = CFG["ensemble"]
HORIZONS = CFG["horizons"]
CLASS_NAMES = CFG["target"]["class_names"]
N_CLASSES = len(CLASS_NAMES)
MEMBERS = ENS["members"]
TASKS = ["regression", "classification"]
SEED = CFG["seed"]

METHOD = "ensemble_v21"
MAX_ITERS = 30
# The budget tau is chosen against: the 2% the system promises, minus the
# safety margin step 7 showed is needed. The same number freeze.py used.
BUDGET = FT["select_false_alarm_rate"]

WEIGHTS_DIR = "models/ensemble_v21"
WEIGHTS_FILE = "models/ensemble_v21/weights.json"
RESULTS_DIR = "results/test_v21"
SOURCE_RESULTS = FT["results_dir"]

# The flag lives in its own file beside config.yaml, which is the paper's
# evidence and is byte-identical to the freeze commit - the same reason step
# 12 put its settings in config_deploy.yaml rather than in config.yaml.
CONFIG_FILE = "config_v21.yaml"
FLAG = "final_system_v21"


def flag_is_set():
    """True when config_v21.yaml asks for ensemble_v21 to be served."""
    import yaml
    path = project_path(CONFIG_FILE)
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get(FLAG) == METHOD


# ---------------------------------------------------------------------------
# The weights, once they exist: what the prediction service reads
# ---------------------------------------------------------------------------
def load_weights(path=None):
    """The saved selection, or None when it has not been produced yet."""
    path = project_path(path or WEIGHTS_FILE)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def weights_for(task, h, saved=None):
    """
    (weight vector in MEMBERS order, tau) for one task and horizon, or None
    when v21 was not adopted there - which is the caller's signal to fall
    back to stacking.
    """
    saved = load_weights() if saved is None else saved
    if not saved:
        return None
    entry = saved.get(task, {}).get(str(h))
    if not entry or not entry.get("beats_stacking_2024", False):
        return None
    if entry.get("members", MEMBERS) != MEMBERS:
        return None
    w = np.array([entry["weights"][m] for m in MEMBERS], dtype="float64")
    return w, entry.get("tau")


def blend(task, w, reg, proba):
    """
    The weighted average of the members' predictions: the DO value for
    regression, the class probabilities for classification. `reg` is
    (n, n_members) and `proba` is (n, n_members, 3), both in MEMBERS order -
    the same layout src.models.ensemble.combine expects.
    """
    w = np.asarray(w, dtype="float64")
    if task == "regression":
        return (reg * w[None, :]).sum(axis=1)
    return (proba * w[None, :, None]).sum(axis=1)


# ---------------------------------------------------------------------------
# tau, exactly the existing rule - but computed in one pass over the grid
# ---------------------------------------------------------------------------
def tau_grid():
    from src.models.run_step7 import tau_grid as g
    return g()


def choose_tau(p_low, y_cls, class_now, budget=BUDGET):
    """
    The same choice freeze.choose_tau_oof makes, vectorised over the grid.

    Under the warning rule a row is predicted Low exactly when
    P(Low) >= tau, so onset recall and the false-alarm rate are both just
    "what share of these rows have P(Low) >= tau". Counting them with a
    searchsorted over the sorted probabilities gives every tau on the grid at
    once, which is what makes it cheap enough to put inside a greedy search.

    Both are non-increasing in tau, so the key (recall, -rate, tau) the
    existing code maximises is attained at the LAST grid point that still
    reaches the best feasible recall. `_check_tau` asserts the two agree.
    """
    ok = ~np.isnan(y_cls) & ~np.isnan(class_now)
    p = np.asarray(p_low, dtype="float64")[ok]
    y = np.asarray(y_cls, dtype="float64")[ok].astype(int)
    now = np.asarray(class_now, dtype="float64")[ok].astype(int)

    not_low_now = now != 0
    onset = not_low_now & (y == 0)
    stays_safe = not_low_now & (y != 0)
    n_on, n_safe = int(onset.sum()), int(stays_safe.sum())
    argmax_rule = {"tau": None, "rule": "argmax", "onset_recall_2024": np.nan,
                   "false_alarm_2024": np.nan}
    if n_on == 0 or n_safe == 0:            # recall or rate is NaN everywhere
        return argmax_rule

    taus = tau_grid()
    on_sorted = np.sort(p[onset])
    safe_sorted = np.sort(p[stays_safe])
    recall = (n_on - np.searchsorted(on_sorted, taus, side="left")) / n_on
    rate = (n_safe - np.searchsorted(safe_sorted, taus, side="left")) / n_safe

    feasible = rate <= budget
    if not feasible.any():
        return argmax_rule
    best_recall = recall[feasible].max()
    i = int(np.flatnonzero(feasible & (recall == best_recall))[-1])
    return {"tau": float(taus[i]), "rule": "P(Low) >= tau",
            "onset_recall_2024": float(recall[i]),
            "false_alarm_2024": float(rate[i])}


def _check_tau(proba, y_cls, class_now, budget=BUDGET):
    """Assert the fast tau search agrees with the one already in the project."""
    from src.models.freeze import choose_tau_oof
    slow = choose_tau_oof(proba, y_cls, class_now, budget=budget)
    fast = choose_tau(proba[:, 0], y_cls, class_now, budget=budget)
    for k in ("tau", "onset_recall_2024", "false_alarm_2024"):
        a, b = slow[k], fast[k]
        both_nan = (a is not None and b is not None
                    and np.isnan(a) and np.isnan(b))
        same = ((a is None and b is None) or both_nan
                or (a is not None and b is not None and abs(a - b) < 1e-12))
        assert same, f"the fast tau search disagrees on {k}: {a} vs {b}"
    return fast


def warn_predictions(proba, tau):
    """The warning rule's class per row, or argmax when no tau met the budget."""
    from src.models.run_step7 import apply_warning
    if tau is None:
        return proba.argmax(axis=1)
    return apply_warning(proba, tau)


# ---------------------------------------------------------------------------
# What the greedy search maximises (higher is better for both tasks)
# ---------------------------------------------------------------------------
def score(task, pred, y_reg, y_cls, class_now, ok_reg, ok_cls):
    """
    Returns (score, extras). Regression: the negative RMSE. Classification:
    the macro F1 of the warning rule, with tau tuned on these same 2024 rows.
    """
    from src.evaluation.step6_report import fast_macro_f1
    if task == "regression":
        err = pred[ok_reg] - y_reg[ok_reg]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        return -rmse, {"rmse": rmse}
    res = choose_tau(pred[:, 0], y_cls, class_now)
    cls = warn_predictions(pred, res["tau"])
    f1 = fast_macro_f1(y_cls[ok_cls].astype(int), cls[ok_cls].astype(int))
    return f1, {"f1_macro": f1, **res}


# ---------------------------------------------------------------------------
# Caruana greedy selection, with replacement
# ---------------------------------------------------------------------------
def greedy_select(task, d, max_iters=MAX_ITERS, verbose=True):
    """
    One task, one horizon. Returns the counts, the weights, the chosen tau
    and the full trace of what the search did.
    """
    P = d["reg"] if task == "regression" else d["proba"]
    args = (d["y_reg"], d["y_cls"], d["class_now"], d["ok_reg"], d["ok_cls"])
    lines = []

    # --- step 1: the best single member ---
    singles = [score(task, P[:, j], *args)[0] for j in range(len(MEMBERS))]
    first = int(np.argmax(singles))
    counts = np.zeros(len(MEMBERS), dtype="int64")
    counts[first] = 1
    acc = P[:, first].astype("float64").copy()
    total = 1
    best = singles[first]
    trace = [{"iteration": 0, "added": MEMBERS[first], "score": float(best)}]
    lines.append(f"    best single: {MEMBERS[first]} ({best:+.6f})")

    # --- step 2: add, repeats allowed, while it strictly improves ---
    for it in range(1, max_iters + 1):
        cand = [score(task, (acc + P[:, j]) / (total + 1), *args)[0]
                for j in range(len(MEMBERS))]
        j = int(np.argmax(cand))
        if not cand[j] > best:
            lines.append(f"    iteration {it}: no addition improves "
                         f"({cand[j]:+.6f} <= {best:+.6f}), stopped")
            break
        acc = acc + P[:, j]
        total += 1
        counts[j] += 1
        best = cand[j]
        trace.append({"iteration": it, "added": MEMBERS[j],
                      "score": float(best)})
        lines.append(f"    iteration {it:<2} +{MEMBERS[j]:<14} {best:+.6f}")

    w = counts / total
    final = blend(task, w, d["reg"], d["proba"])
    s, extras = score(task, final, *args)
    assert abs(s - best) < 1e-9, "the weights do not reproduce the search score"
    if task == "classification":
        _check_tau(final, d["y_cls"], d["class_now"])

    out = {"members": list(MEMBERS),
           "counts": {m: int(c) for m, c in zip(MEMBERS, counts)},
           "n_selections": int(total),
           "weights": {m: float(x) for m, x in zip(MEMBERS, w)},
           "best_single": MEMBERS[first],
           "best_single_score_2024": float(singles[first]),
           "single_scores_2024": {m: float(v)
                                  for m, v in zip(MEMBERS, singles)},
           "val_2024_score": float(best),
           "tau": extras.get("tau"),
           "rule": extras.get("rule", "argmax"),
           "val_2024": {k: v for k, v in extras.items() if k != "rule"},
           "trace": trace,
           "log": lines}
    if verbose:
        print("\n".join(lines))
    return out


# ---------------------------------------------------------------------------
# 2024: the members, and what stacking scores under the same rule
# ---------------------------------------------------------------------------
def load_val(h):
    """The members' 2024 predictions for one horizon, plus the valid masks."""
    from src.models import ensemble as E
    keys, y_reg, y_cls, reg, proba = E.load_member_predictions(h)
    class_now = keys["class_now"].to_numpy(dtype="float64")
    assert not np.isnan(reg).any() and not np.isnan(proba).any(), \
        f"a member has a missing 2024 prediction at h{h}"
    return {"h": h, "keys": keys[["station", "time"]].reset_index(drop=True),
            "y_reg": y_reg, "y_cls": y_cls, "class_now": class_now,
            "reg": reg, "proba": proba,
            "ok_reg": ~np.isnan(y_reg),
            "ok_cls": ~np.isnan(y_cls)}


def stacking_val_score(task, h, d):
    """
    The deployed stacking meta-model - the one fitted on all of 2024 in step 7
    and saved to models/ensemble/ - scored on 2024 under exactly the rule v21
    is scored by. Both are fitted on all of 2024 and read back on all of 2024,
    so the comparison that decides the flag is like for like.
    """
    import joblib
    path = os.path.join(project_path(CFG["context_ensemble"]["model_dir"]),
                        f"stacking_{task}_h{h}_final.joblib")
    model = joblib.load(path)["model"]
    if task == "regression":
        pred = model.predict(d["reg"])
    else:
        pred = model.predict_proba(d["proba"].reshape(len(d["proba"]), -1))
        if pred.shape[1] < N_CLASSES:
            full = np.zeros((len(pred), N_CLASSES))
            full[:, model.classes_.astype(int)] = pred
            pred = full
    return score(task, pred, d["y_reg"], d["y_cls"], d["class_now"],
                 d["ok_reg"], d["ok_cls"])


def stacking_oof_score(task, h, d):
    """
    The same score on step 7's OUT-OF-FOLD 2024 predictions - the honest
    number for stacking, reported alongside so the in-sample comparison above
    cannot be mistaken for one.
    """
    path = os.path.join(project_path(CFG["step7"]["oof_predictions"]),
                        f"stacking_{task}_h{h}.parquet")
    if not os.path.exists(path):
        return np.nan
    oof = pd.read_parquet(path)
    assert (oof["time"].to_numpy() == d["keys"]["time"].to_numpy()).all(), \
        "the out-of-fold rows do not line up with the member predictions"
    if task == "regression":
        pred = oof["y_pred"].to_numpy(dtype="float64")
        ok = d["ok_reg"] & ~np.isnan(pred)
        return -float(np.sqrt(np.mean((pred[ok] - d["y_reg"][ok]) ** 2)))
    pred = oof[[f"proba_{n}" for n in CLASS_NAMES]].to_numpy(dtype="float64")
    bad = np.isnan(pred).any(axis=1)
    ok_cls = d["ok_cls"] & ~bad
    y_cls = np.where(ok_cls, d["y_cls"], np.nan)
    class_now = np.where(ok_cls, d["class_now"], np.nan)
    pred = np.where(bad[:, None], 1.0 / N_CLASSES, pred)
    return score(task, pred, d["y_reg"], y_cls, class_now,
                 d["ok_reg"], ok_cls)[0]


# ---------------------------------------------------------------------------
# 2025: read the members' saved test predictions, blend, score
# ---------------------------------------------------------------------------
def load_test(h):
    """
    The stage-2 members' 2025 predictions for one horizon, from the parquets
    src.evaluation.test_2025 already wrote. Nothing is re-predicted.
    """
    src = project_path(SOURCE_RESULTS)
    reg, proba, keys = [], [], None
    y_reg = y_cls = class_now = None
    for name in MEMBERS:
        d = pd.read_parquet(os.path.join(src, f"pred_{name}_h{h}.parquet"))
        if keys is None:
            keys = d[["station", "time"]].reset_index(drop=True)
            y_reg = d["y_true_reg"].to_numpy(dtype="float64")
            y_cls = d["y_true_cls"].astype("float64").to_numpy()
            class_now = d["class_now"].astype("float64").to_numpy()
        else:
            assert (d["time"].to_numpy() == keys["time"].to_numpy()).all(), \
                f"{name} is not aligned with the other members on 2025"
        reg.append(d["pred_reg"].to_numpy(dtype="float64"))
        proba.append(d[[f"proba_{n}" for n in CLASS_NAMES]]
                     .to_numpy(dtype="float64"))
    pers = pd.read_parquet(os.path.join(src, f"pred_persistence_h{h}.parquet"))
    return {"h": h, "keys": keys, "y_reg": y_reg, "y_cls": y_cls,
            "class_now": class_now, "reg": np.column_stack(reg),
            "proba": np.stack(proba, axis=1),
            "persistence": pers["pred_reg"].to_numpy(dtype="float64")}


def score_2025(name, t, pred_reg, pred_proba, tau, label="stage2"):
    """
    One method's 2025 rows, built exactly the way src.evaluation.test_2025
    builds them, so the two tables can sit in the same file.
    """
    from src.evaluation.metrics import (classification_metrics,
                                        danger_zone_rmse,
                                        early_warning_metrics,
                                        regression_metrics)
    rows_r, rows_c = [], []
    reg = regression_metrics(t["y_reg"], pred_reg, t["persistence"])
    reg["danger_rmse"] = danger_zone_rmse(t["y_reg"], pred_reg,
                                          *ENS["danger_zone"])
    rows_r.append({"method": name, "horizon": t["h"], "members": label, **reg})

    argmax = pred_proba.argmax(axis=1)
    warn = argmax if tau is None else warn_predictions(pred_proba, tau)
    for rule, pred in [("argmax", argmax), ("warn", warn)]:
        cm = classification_metrics(t["y_cls"], pred, pred_proba)
        cm.update(early_warning_metrics(t["class_now"], t["y_cls"], pred))
        flat = {k: v for k, v in cm.items() if k != "confusion_matrix"}
        flat["confusion_matrix"] = ";".join(
            str(int(x)) for x in cm["confusion_matrix"].ravel())
        rows_c.append({"method": name, "horizon": t["h"], "rule": rule,
                       "members": label, "tau": tau, **flat})
    return rows_r, rows_c


def check_reproduces_stacking(t):
    """
    A guard on the whole 2025 path: re-score the SAVED stacking predictions
    with the code above and check the numbers land on the committed
    results/test/ rows. If they do, the ensemble_v21 rows written next to
    them were produced the same way.
    """
    src = project_path(SOURCE_RESULTS)
    h = t["h"]
    d = pd.read_parquet(os.path.join(src, f"pred_stacking_h{h}.parquet"))
    proba = d[[f"proba_{n}" for n in CLASS_NAMES]].to_numpy(dtype="float64")
    tau = _frozen_tau("stacking", h)
    rows_r, rows_c = score_2025("stacking", t,
                                d["pred_reg"].to_numpy(dtype="float64"),
                                proba, tau)
    reg = pd.read_csv(os.path.join(src, "metrics_regression_stage2.csv"))
    cls = pd.read_csv(os.path.join(src, "metrics_classification_stage2.csv"))
    want_r = reg[(reg["method"] == "stacking") & (reg["horizon"] == h)].iloc[0]
    for col in ["n", "r2", "rmse", "mae", "skill_vs_persistence", "danger_rmse"]:
        assert abs(float(rows_r[0][col]) - float(want_r[col])) < 1e-9, \
            f"the 2025 regression scoring path does not reproduce {col} at h{h}"
    for row in rows_c:
        want = cls[(cls["method"] == "stacking") & (cls["horizon"] == h)
                   & (cls["rule"] == row["rule"])].iloc[0]
        for col in ["n", "accuracy", "f1_macro", "roc_auc_ovr_macro",
                    "onset_recall", "false_alarm_rate"]:
            assert abs(float(row[col]) - float(want[col])) < 1e-9, \
                (f"the 2025 classification scoring path does not reproduce "
                 f"{col} at h{h} ({row['rule']})")
    return True


def _frozen_tau(method, h):
    with open(project_path(FT["thresholds_file"]), encoding="utf-8") as f:
        return json.load(f).get(f"{method}_h{h}", {}).get("tau")


# ---------------------------------------------------------------------------
# The bootstrap: the macro-F1 difference, v21 minus the best single model
# ---------------------------------------------------------------------------
def bootstrap_f1_difference(t, a_cls, b_cls, n_resamples, rng):
    """
    A station-week block bootstrap of macro_F1(a) - macro_F1(b) on 2025,
    using the project's own blocks and its own fast macro F1. Returns the
    mean and the 2.5 / 97.5 percentiles.
    """
    from src.evaluation.step6_report import fast_macro_f1, station_week_blocks
    ok = ~np.isnan(t["y_cls"])
    blocks = station_week_blocks(t["keys"], np.ones(len(t["y_cls"]), bool))
    y = t["y_cls"]
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[j] for j in pick])
        idx = idx[ok[idx]]
        yc = y[idx].astype(int)
        diffs[i] = (fast_macro_f1(yc, a_cls[idx].astype(int))
                    - fast_macro_f1(yc, b_cls[idx].astype(int)))
    return {"mean": float(diffs.mean()),
            "lo": float(np.percentile(diffs, 2.5)),
            "hi": float(np.percentile(diffs, 97.5)),
            "n_resamples": int(n_resamples)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
REG_COLS = ["r2", "rmse", "mae", "skill_vs_persistence", "danger_rmse"]
CLS_COLS = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
            "roc_auc_ovr_macro"]


def _table(title, methods, labels, values, cols):
    """A small fixed-width table: one row per method, one column per metric."""
    w = max(len(x) for x in labels) + 2
    head = f"{'method':<{w}}" + "".join(f"{c:>21}" for c in cols)
    out = [title, head, "-" * len(head)]
    for name, label in zip(methods, labels):
        row = f"{label:<{w}}"
        for c in cols:
            v = values[(name, c)]
            blank = v is None or (isinstance(v, float) and np.isnan(v))
            row += f"{'-':>21}" if blank else f"{v:>21.4f}"
        out.append(row)
    return "\n".join(out)


def report_tables(reg, cls, rule="argmax"):
    """The two comparison tables, at h24 and as the mean over all horizons."""
    pretty = {METHOD: "ensemble_v21", "stacking": "stacking"}
    blocks = []
    for title, frame, cols, task in [
            ("REGRESSION", reg, REG_COLS, "regression"),
            ("CLASSIFICATION", cls, CLS_COLS, "classification")]:
        best = FT["best_single"][task]
        methods = [METHOD, "stacking", best]
        labels = [pretty.get(m, f"{m} (best single)") for m in methods]
        f = frame if task == "regression" else frame[frame["rule"] == rule]
        suffix = "" if task == "regression" else f", rule = {rule}"
        for scope, sel in [("horizon 24 h", f[f["horizon"] == 24]),
                           ("mean over 6/12/24/48 h", f)]:
            vals = {}
            for m in methods:
                sub = sel[sel["method"] == m]
                for c in cols:
                    vals[(m, c)] = float(sub[c].mean()) if len(sub) else np.nan
            blocks.append(_table(f"\n{title} on 2025 - {scope}{suffix}",
                                 methods, labels, vals, cols))
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    import argparse
    import time

    from joblib import Parallel, delayed

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--quick", action="store_true",
                    help="5 greedy additions and 50 bootstrap resamples")
    ap.add_argument("--jobs", type=int, default=-1,
                    help="joblib n_jobs (default -1: every core)")
    args = ap.parse_args()
    max_iters = 5 if args.quick else MAX_ITERS
    n_boot = 50 if args.quick else ENS["bootstrap"]["n_resamples"]

    os.makedirs(project_path(WEIGHTS_DIR), exist_ok=True)
    os.makedirs(project_path(RESULTS_DIR), exist_ok=True)
    t0 = time.perf_counter()
    print(f"ensemble_v21: Caruana greedy selection with replacement, "
          f"members {MEMBERS}")
    print("  weights and tau are chosen on 2024 only; 2025 is scored once, "
          "at the end")
    print(f"  tau budget {BUDGET:.1%} (final_test.select_false_alarm_rate), "
          f"up to {max_iters} additions\n")

    # ---- 2024 ----
    print("  reading the members' 2024 validation predictions ...")
    val = dict(zip(HORIZONS,
                   Parallel(n_jobs=min(len(HORIZONS), 4))(
                       delayed(load_val)(h) for h in HORIZONS)))
    print(f"    {len(val[HORIZONS[0]]['keys']):,} rows x {len(MEMBERS)} "
          f"members x {len(HORIZONS)} horizons "
          f"({time.perf_counter() - t0:.0f}s)")

    jobs = [(task, h) for task in TASKS for h in HORIZONS]
    print("\n  greedy selection on 2024")
    selected = Parallel(n_jobs=args.jobs, prefer="threads")(
        delayed(greedy_select)(task, val[h], max_iters, False)
        for task, h in jobs)
    chosen = {task: {} for task in TASKS}
    for (task, h), sel in zip(jobs, selected):
        print(f"  --- {task} h{h} ---")
        print("\n".join(sel.pop("log")))
        chosen[task][str(h)] = sel

    # ---- how it compares with stacking on the same 2024 rows ----
    print("\n  2024 validation: ensemble_v21 vs the deployed stacking model")
    print(f"  {'task':<15}{'h':>4}{'v21':>12}{'stacking':>12}{'>=':>6}"
          f"{'stacking (oof)':>17}")
    for task in TASKS:
        for h in HORIZONS:
            sel = chosen[task][str(h)]
            s_val, s_extras = stacking_val_score(task, h, val[h])
            s_oof = stacking_oof_score(task, h, val[h])
            beats = bool(sel["val_2024_score"] >= s_val)
            sel["stacking_val_2024_score"] = float(s_val)
            sel["stacking_val_2024_tau"] = s_extras.get("tau")
            sel["stacking_oof_2024_score"] = (None if np.isnan(s_oof)
                                              else float(s_oof))
            sel["beats_stacking_2024"] = beats
            sel["score_name"] = ("negative RMSE" if task == "regression"
                                 else "macro F1 (warning rule)")
            print(f"  {task:<15}{h:>4}{sel['val_2024_score']:>12.5f}"
                  f"{s_val:>12.5f}{('yes' if beats else 'no'):>6}"
                  f"{s_oof:>17.5f}")

    payload = {
        "method": METHOD,
        "algorithm": "Caruana greedy ensemble selection, with replacement",
        "members": list(MEMBERS),
        "max_iterations": max_iters,
        "selected_on": "2024 validation (results/val/predictions)",
        "tau_budget": BUDGET,
        "scored_on": "2025 test (results/test/pred_<member>_h<h>.parquet)",
        "seed": SEED,
        "note": ("the weights and tau come from 2024 only. The 2024 scores "
                 "of ensemble_v21 and of stacking are both in-sample on "
                 "2024, because both are fitted on all of 2024; step 7's "
                 "out-of-fold stacking score is kept alongside so the "
                 "in-sample comparison cannot be mistaken for an honest "
                 "estimate of stacking's 2024 skill."),
        **chosen}
    path = project_path(WEIGHTS_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  wrote {WEIGHTS_FILE}")

    # ---- 2025, once ----
    print("\n  reading the members' 2025 test predictions ...")
    test = dict(zip(HORIZONS,
                    Parallel(n_jobs=min(len(HORIZONS), 4))(
                        delayed(load_test)(h) for h in HORIZONS)))
    rows_r, rows_c, v21_cls = [], [], {}
    for h in HORIZONS:
        t = test[h]
        check_reproduces_stacking(t)
        wr = np.array([chosen["regression"][str(h)]["weights"][m]
                       for m in MEMBERS])
        wc = np.array([chosen["classification"][str(h)]["weights"][m]
                       for m in MEMBERS])
        tau = chosen["classification"][str(h)]["tau"]
        pred_reg = blend("regression", wr, t["reg"], t["proba"])
        pred_proba = blend("classification", wc, t["reg"], t["proba"])
        r, c = score_2025(METHOD, t, pred_reg, pred_proba, tau)
        rows_r += r
        rows_c += c
        argmax = pred_proba.argmax(axis=1)
        v21_cls[h] = {"argmax": argmax,
                      "warn": (argmax if tau is None
                               else warn_predictions(pred_proba, tau))}
        print(f"    h{h:<3} scored (the stacking rows reproduce "
              f"results/test exactly)")

    src = project_path(SOURCE_RESULTS)
    out = project_path(RESULTS_DIR)
    frames = {}
    for task, rows in [("regression", rows_r), ("classification", rows_c)]:
        old = pd.read_csv(os.path.join(src, f"metrics_{task}_stage2.csv"))
        new = pd.DataFrame(rows).reindex(columns=old.columns)
        both = pd.concat([old, new], ignore_index=True)
        both.to_csv(os.path.join(out, f"metrics_{task}_stage2.csv"),
                    index=False)
        frames[task] = both
        print(f"  wrote {RESULTS_DIR}/metrics_{task}_stage2.csv "
              f"({len(old)} rows copied unchanged + {len(new)} new)")

    # ---- the comparison tables ----
    print(report_tables(frames["regression"], frames["classification"]))

    # ---- the bootstrap ----
    best_cls = FT["best_single"]["classification"]
    print(f"\n\nBOOTSTRAP: macro F1, ensemble_v21 minus {best_cls} "
          f"(the best single model), 2025")
    print(f"  {n_boot:,} station-week resamples, 95% percentile interval, "
          f"seed {SEED}")
    print(f"  {'h':>4}{'rule':>9}{'difference':>14}{'95% CI':>28}"
          f"{'excludes 0':>12}")
    boot = {}
    for h in HORIZONS:
        t = test[h]
        single = pd.read_parquet(
            os.path.join(src, f"pred_{best_cls}_h{h}.parquet"))
        for rule in ["argmax", "warn"]:
            rng = np.random.default_rng(SEED)
            res = bootstrap_f1_difference(t, v21_cls[h][rule],
                                          single[f"pred_{rule}"].to_numpy(),
                                          n_boot, rng)
            res["significant"] = bool(res["lo"] > 0 or res["hi"] < 0)
            boot[f"h{h}_{rule}"] = res
            ci = f"[{res['lo']:+.5f}, {res['hi']:+.5f}]"
            print(f"  {h:>4}{rule:>9}{res['mean']:>+14.5f}{ci:>28}"
                  f"{('yes' if res['significant'] else 'no'):>12}")

    payload["bootstrap_2025_f1_vs_best_single"] = {
        "best_single": best_cls, "n_resamples": n_boot, **boot}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    adopted = [f"{task} h{h}" for task in TASKS for h in HORIZONS
               if chosen[task][str(h)]["beats_stacking_2024"]]
    print(f"\n  ensemble_v21 >= stacking on 2024 for: "
          f"{', '.join(adopted) if adopted else 'nothing'}")
    print(f"  total {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
