"""
STEP 7 - run every method out-of-fold over 2024, under one honest protocol.

Run from the project root:

    python -m src.models.run_step7

What happens here:

  1. Six two-month folds of 2024. Anything that LEARNS from 2024 - the
     stacking coefficients, the CAE meta-model, ATAE's four settings, and
     the warning threshold tau - is fitted on five folds and used on the
     sixth, with 14 days purged either side. Every row of 2024 ends up
     predicted exactly once by a model that never saw it.
  2. Methods that learn nothing (persistence, linear, the five members, the
     simple average) are scored on the same rows, so the comparison is like
     for like.
  3. The warning rule: instead of taking the most likely class, issue a Low
     warning when P(Low) >= tau. tau is picked on the training folds to
     catch as many onsets as possible within a false-alarm budget.
  4. Finally each learned method is refitted on ALL of 2024 and saved to
     models/ensemble/, ready for the 2025 test in step 8.

The 2025 test split is never read.

Writes:
    results/val/oof/<method>_<task>_h<h>.parquet   out-of-fold predictions
    results/val/oof/taus.csv                       the chosen thresholds
    results/val/oof/folds.csv                      the fold table
    results/val/oof/cae_importance_<task>_h<h>.csv
    models/ensemble/                               the final full-2024 fits
"""

import json
import os
import time

import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.evaluation.metrics import early_warning_metrics
from src.models import context_ensemble as CAE
from src.models import crossfit as X
from src.models import ensemble as E

CFG = load_config()
ENS = CFG["ensemble"]
WR = CFG["warning_rule"]
S7 = CFG["step7"]
MEMBERS = ENS["members"]
HORIZONS = CFG["horizons"]
CLASS_NAMES = CFG["target"]["class_names"]
N_CLASSES = len(CLASS_NAMES)
OOF_DIR = project_path(S7["oof_predictions"])


# ---------------------------------------------------------------------------
# The warning rule
# ---------------------------------------------------------------------------
def tau_grid():
    return np.arange(WR["tau_grid_start"], WR["tau_grid_stop"] + 1e-9,
                     WR["tau_grid_step"])


def apply_warning(proba, tau):
    """
    Issue a Low warning when P(Low) reaches tau; otherwise pick the better
    of Medium and Good. (Falling back to a plain argmax would contradict
    the rule, since argmax can return Low below tau.)
    """
    pred = 1 + proba[:, 1:].argmax(axis=1)
    pred[proba[:, 0] >= tau] = 0
    return pred


def choose_tau(class_now, y_cls, proba, rows):
    """
    The tau that catches the most onsets while keeping the false-alarm rate
    inside the budget - chosen on the TRAINING rows handed in here, never on
    the fold being predicted. Returns (tau, achieved recall, achieved rate)
    or (None, ...) when no tau fits the budget, in which case the caller
    falls back to argmax.
    """
    budget = WR["max_false_alarm_rate"]
    best = None
    for tau in tau_grid():
        pred = apply_warning(proba[rows], tau)
        m = early_warning_metrics(class_now[rows], y_cls[rows], pred)
        rate, recall = m["false_alarm_rate"], m["onset_recall"]
        if np.isnan(rate) or np.isnan(recall) or rate > budget:
            continue
        # Most onsets caught wins. Among taus that catch the same number,
        # prefer fewer false alarms, and then the higher threshold - the
        # more conservative choice of two that look identical here.
        key = (recall, -rate, tau)
        if best is None or key > best[0]:
            best = (key, float(tau), float(recall), float(rate))
    return (None, -1.0, np.nan) if best is None else (best[1], best[2], best[3])


# ---------------------------------------------------------------------------
# Methods that learn nothing
# ---------------------------------------------------------------------------
def fixed_methods(h, d):
    """persistence, linear, the five members, the simple average."""
    from src.evaluation.step6_report import load_baseline
    out = {}
    for name in ENS["baselines"]:
        rp, cp, pr = load_baseline(name, h, d["keys"])
        out[name] = {"reg": rp, "cls": cp, "proba": pr}
    for i, name in enumerate(MEMBERS):
        out[name] = {"reg": d["reg"][:, i], "cls": d["proba"][:, i, :].argmax(1),
                     "proba": d["proba"][:, i, :]}
    avg_r, _ = E.simple_average("regression", d["reg"], d["proba"])
    avg_p, _ = E.simple_average("classification", d["reg"], d["proba"])
    out["simple_average"] = {"reg": avg_r, "cls": avg_p.argmax(1), "proba": avg_p}
    return out


# ---------------------------------------------------------------------------
# ATAE, cross-fitted: its four settings are chosen per fold
# ---------------------------------------------------------------------------
def atae_candidates(d, h):
    """
    Pre-compute ATAE's weights for every grid point ONCE. The rolling error
    itself is causal and fold-independent - it only ever uses the past - so
    only the CHOICE of settings has to be made per fold.
    """
    grid = ENS["grid"]
    cands = {}
    combos = []
    for alpha in grid["alpha"]:
        for sigma in grid["sigma"]:
            if alpha == 0 and any(c[0] == 0 for c in combos):
                continue
            combos.append((alpha, sigma))
    for task in ["regression", "classification"]:
        for alpha, sigma in combos:
            e = E.member_errors(task, d["y_reg"], d["y_cls"], d["reg"],
                                d["proba"], alpha, sigma)
            for window in grid["window_hours"]:
                recent, _ = E.recent_errors(d["keys"], e, h, window,
                                            ENS["min_known_errors"])
                for temperature in grid["temperature"]:
                    w = E.weights_from_errors(recent, temperature)
                    pred = E.combine(task, w, d["reg"], d["proba"])
                    cands[(task, alpha, sigma, window, temperature)] = pred
    return cands


def atae_oof(d, h, cands):
    """Pick ATAE's settings on each fold's training rows, predict the fold."""
    reg = np.full(len(d["y_reg"]), np.nan)
    proba = np.full((len(d["y_reg"]), N_CLASSES), np.nan)
    chosen = []
    times = d["keys"]["time"]
    for k in X.all_folds():
        train, test = X.split_indices(times, k)
        for task in ["regression", "classification"]:
            best, best_score = None, -np.inf
            for key, pred in cands.items():
                if key[0] != task:
                    continue
                s = E.tuning_score(task, d["y_reg"], d["y_cls"],
                                   d["class_now"], pred, train,
                                   ENS["score_alpha"], ENS["score_sigma"])
                if s > best_score:
                    best, best_score = key, s
            if task == "regression":
                reg[test] = cands[best][test]
            else:
                proba[test] = cands[best][test]
            chosen.append({"fold": k, "task": task, "alpha": best[1],
                           "sigma": best[2], "window_hours": best[3],
                           "temperature": best[4]})
    return reg, proba, pd.DataFrame(chosen)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OOF_DIR, exist_ok=True)
    data = E.load_all()
    times = data[HORIZONS[0]]["keys"]["time"]
    folds = X.describe(times)
    folds.to_csv(os.path.join(OOF_DIR, "folds.csv"), index=False)
    print("Cross-fitting folds over 2024:")
    print(folds.to_string(index=False))
    print(f"\npurge: {CFG['crossfit']['purge_days']} days each side | "
          f"members: {MEMBERS}\n")

    taus, atae_settings, importances = [], [], []
    for h in HORIZONS:
        d = data[h]
        t0 = time.perf_counter()
        print(f"=== horizon {h} h ===")
        methods = fixed_methods(h, d)

        # ---- stacking and CAE, fold by fold ----
        stack_reg = np.full(len(d["y_reg"]), np.nan)
        stack_proba = np.full((len(d["y_reg"]), N_CLASSES), np.nan)
        cae_reg = np.full(len(d["y_reg"]), np.nan)
        cae_proba = np.full((len(d["y_reg"]), N_CLASSES), np.nan)

        context = CAE.load_context(d["keys"])
        feats = {task: CAE.build_features(task, d["keys"], d["y_reg"],
                                          d["y_cls"], d["reg"], d["proba"],
                                          h, context)[0]
                 for task in ["regression", "classification"]}

        for k in X.all_folds():
            train, test = X.split_indices(times, k)
            # stacking: fitted on the training folds only
            sr, _ = E.stacking("regression", d["reg"], d["proba"], d["y_reg"],
                               d["y_cls"], np.isin(np.arange(len(d["y_reg"])), train))
            sp, _ = E.stacking("classification", d["reg"], d["proba"],
                               d["y_reg"], d["y_cls"],
                               np.isin(np.arange(len(d["y_reg"])), train))
            stack_reg[test] = sr[test]
            stack_proba[test] = sp[test]

            # CAE
            m_r = CAE.fit_cae("regression", feats["regression"], d["y_reg"],
                              times, train)
            cae_reg[test] = CAE.predict_cae("regression", m_r,
                                            feats["regression"], test)
            m_c = CAE.fit_cae("classification", feats["classification"],
                              d["y_cls"], times, train)
            cae_proba[test] = CAE.predict_cae("classification", m_c,
                                              feats["classification"], test)
            print(f"  fold {k}: train {len(train):,} test {len(test):,} ok")

        methods["stacking"] = {"reg": stack_reg, "cls": stack_proba.argmax(1),
                               "proba": stack_proba}
        methods["cae"] = {"reg": cae_reg, "cls": cae_proba.argmax(1),
                          "proba": cae_proba}

        # ---- ATAE ----
        cands = atae_candidates(d, h)
        a_reg, a_proba, a_chosen = atae_oof(d, h, cands)
        a_chosen["horizon"] = h
        atae_settings.append(a_chosen)
        methods["atae"] = {"reg": a_reg, "cls": a_proba.argmax(1),
                           "proba": a_proba}

        # ---- the warning rule, per method, chosen per fold ----
        for name, m in methods.items():
            if m["proba"] is None:               # persistence has no probabilities
                m["cls_warn"] = m["cls"]
                taus.append({"method": name, "horizon": h, "fold": -1,
                             "tau": np.nan, "rule": "own rule"})
                continue
            warn = np.empty(len(m["cls"]), dtype=int)
            for k in X.all_folds():
                train, test = X.split_indices(times, k)
                tau, rec, rate = choose_tau(d["class_now"], d["y_cls"],
                                            m["proba"], train)
                if tau is None:
                    warn[test] = m["proba"][test].argmax(1)
                else:
                    warn[test] = apply_warning(m["proba"][test], tau)
                taus.append({"method": name, "horizon": h, "fold": k,
                             "tau": tau if tau is not None else np.nan,
                             "rule": "argmax (no tau met the budget)"
                                     if tau is None else "P(Low) >= tau",
                             "train_onset_recall": rec,
                             "train_false_alarm": rate})
            m["cls_warn"] = warn

        # ---- save the out-of-fold predictions ----
        for name, m in methods.items():
            base = d["keys"][["station", "time"]].copy()
            r = base.copy()
            r["y_true"] = d["y_reg"]
            r["y_pred"] = m["reg"]
            r.to_parquet(os.path.join(OOF_DIR, f"{name}_regression_h{h}.parquet"),
                         index=False)
            c = base.copy()
            c["y_true"] = d["y_cls"]
            c["y_pred"] = m["cls"]
            c["y_pred_warn"] = m["cls_warn"]
            c["class_now"] = d["class_now"]
            if m["proba"] is not None:
                for i, n in enumerate(CLASS_NAMES):
                    c[f"proba_{n}"] = m["proba"][:, i]
            c.to_parquet(os.path.join(OOF_DIR,
                                      f"{name}_classification_h{h}.parquet"),
                         index=False)

        # ---- the final fits on ALL of 2024, for step 8 ----
        all_rows = np.flatnonzero(X.fold_of(times) >= 0)
        for task, target in [("regression", d["y_reg"]),
                             ("classification", d["y_cls"])]:
            model = CAE.fit_cae(task, feats[task], target, times, all_rows)
            CAE.save_model(model, task, h, list(feats[task].columns))
            imp = pd.DataFrame({"feature": feats[task].columns,
                                "importance": model.feature_importances_,
                                "task": task, "horizon": h})
            imp = imp.sort_values("importance", ascending=False)
            imp.to_csv(os.path.join(OOF_DIR,
                                    f"cae_importance_{task}_h{h}.csv"),
                       index=False)
            importances.append(imp)
        # stacking, fitted on all of 2024
        import joblib
        for task in ["regression", "classification"]:
            _, model = E.stacking(task, d["reg"], d["proba"], d["y_reg"],
                                  d["y_cls"],
                                  np.isin(np.arange(len(d["y_reg"])), all_rows))
            joblib.dump({"model": model, "members": MEMBERS, "task": task,
                         "horizon": h},
                        os.path.join(project_path(CFG["context_ensemble"]["model_dir"]),
                                     f"stacking_{task}_h{h}_final.joblib"))
        print(f"  done in {(time.perf_counter() - t0) / 60:.1f} min\n")

    pd.DataFrame(taus).to_csv(os.path.join(OOF_DIR, "taus.csv"), index=False)
    pd.concat(atae_settings).to_csv(os.path.join(OOF_DIR, "atae_folds.csv"),
                                    index=False)
    pd.concat(importances).to_csv(os.path.join(OOF_DIR, "cae_importance.csv"),
                                  index=False)
    print(f"Wrote out-of-fold predictions to {OOF_DIR}")
    print("Next: python -m src.evaluation.step7_report")


if __name__ == "__main__":
    main()
