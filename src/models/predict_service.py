"""
The prediction service the app uses.

One place that knows how to go from raw hourly readings to a forecast and a
warning, so the app never reimplements anything. It reuses the project's own
feature code (src.features.build_features) and the project's own sequence
code (src.models.sequences), loads the saved models, and applies the frozen
warning thresholds.

Nothing here trains, tunes or scores. It only loads and predicts.

    from src.models.predict_service import Forecaster
    f = Forecaster(stage="final")          # or "stage2"
    out = f.predict_station("02198950", "2025-07-15 12:00")
"""

import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.models import common as C

CFG = load_config()
FT = CFG["final_test"]
ENS = CFG["ensemble"]
HORIZONS = CFG["horizons"]
CLASS_NAMES = CFG["target"]["class_names"]
MEMBERS = ENS["members"]
LOW = CFG["target"]["thresholds"]["low"]
DO = CFG["target"]["column"]

STAGE_DIRS = {"final": FT["stage3_dir"], "stage2": FT["stage2_dir"]}
STAGE_LABEL = {
    "final": "models/final - trained on 2023+2024+2025 (never scored; "
             "for live use)",
    "stage2": "models/stage2 - trained on 2023+2024 (these are the models "
              "the reported 2025 results come from)",
}


@lru_cache(maxsize=4)
def load_thresholds():
    with open(project_path(FT["thresholds_file"]), encoding="utf-8") as f:
        return json.load(f)


def tau_for(method, h):
    return load_thresholds().get(f"{method}_h{h}", {}).get("tau")


def classify(do_value):
    from src.evaluation.metrics import to_class
    return to_class(do_value)


class Forecaster:
    """Loads one set of models and answers forecast questions with them."""

    def __init__(self, stage="final"):
        if stage not in STAGE_DIRS:
            raise ValueError(f"stage must be one of {list(STAGE_DIRS)}")
        self.stage = stage
        self.folder = project_path(STAGE_DIRS[stage])
        if not os.path.isdir(self.folder):
            raise FileNotFoundError(
                f"{self.folder} does not exist - run "
                f"`python -m src.models.stage2 "
                f"{'--stage 3' if stage == 'final' else ''}` first")
        self.label = STAGE_LABEL[stage]
        self.features = C.feature_columns()
        self._cache = {}

    # -- members -----------------------------------------------------------
    def _tabular(self, name, task, h, X):
        import joblib
        key = (name, task, h)
        if key not in self._cache:
            if name == "xgboost":
                from xgboost import XGBClassifier, XGBRegressor
                m = XGBRegressor() if task == "regression" else XGBClassifier()
                m.load_model(os.path.join(self.folder,
                                          f"{name}_{task}_h{h}.json"))
                self._cache[key] = ("xgb", m, None)
            elif name == "lightgbm":
                b = joblib.load(os.path.join(self.folder,
                                             f"{name}_{task}_h{h}.joblib"))
                self._cache[key] = ("lgb", b["model"], b["cats"])
            else:
                m = joblib.load(os.path.join(self.folder,
                                             f"{name}_{task}_h{h}.joblib"))
                self._cache[key] = ("sk", m, None)
        kind, model, cats = self._cache[key]
        Xi = X[self.features].copy()
        if kind == "lgb":
            for c, levels in (cats or {}).items():
                if c in Xi.columns:
                    Xi[c] = pd.Categorical(Xi[c].astype("int32"),
                                           categories=levels)
        return model.predict(Xi) if task == "regression" else model.predict_proba(Xi)

    def _deep(self, name, rows):
        """
        The three seeds of one deep model, averaged. `rows` must be a
        SequenceData-like object built by sequences.prepare.
        """
        import joblib
        import torch
        from src.models import sequences as S
        from src.models import train_dl as T
        from src.models.dl_models import MODELS, MultiTaskLoss

        bundle = joblib.load(os.path.join(self.folder, "dl_scaler.joblib"))
        reg_t, cls_t = T.build_targets(rows, bundle["do_scale"])
        device = T.pick_device()        # falls back to CPU automatically
        n_features = 0 if rows.features is None else rows.features.shape[1]
        loss_fn = MultiTaskLoss(T.class_weights_from_train(cls_t)).to(device)
        preds, probs = [], []
        for seed in CFG["dl_search"]["final_seeds"]:
            ckpt = torch.load(os.path.join(self.folder, f"{name}_seed{seed}.pt"),
                              map_location=device)
            # the embedding has one row per station the model was TRAINED on;
            # a single-station request must not shrink it
            n_stations = int(ckpt["state_dict"]["embedding.weight"].shape[0])
            model = MODELS[name](S.N_CHANNELS, n_stations, cfg=ckpt["arch"],
                                 n_features=n_features).to(device)
            model.load_state_dict(ckpt["state_dict"])
            _, do_pred, proba = T.predict(model, rows, reg_t, cls_t, loss_fn,
                                          device,
                                          CFG["deep_learning"]["batch_size"],
                                          bundle["do_scale"])
            preds.append(do_pred)
            probs.append(proba)
            del model
        return np.mean(preds, axis=0), np.mean(probs, axis=0)

    # -- the stacking meta-model ------------------------------------------
    def _stack(self, task, h, reg, proba):
        import joblib
        path = os.path.join(project_path(CFG["context_ensemble"]["model_dir"]),
                            f"stacking_{task}_h{h}_final.joblib")
        model = joblib.load(path)["model"]
        if task == "regression":
            return model.predict(reg)
        out = model.predict_proba(proba.reshape(len(proba), -1))
        if out.shape[1] < 3:
            full = np.zeros((len(out), 3))
            full[:, model.classes_.astype(int)] = out
            out = full
        return out

    # -- the public call ---------------------------------------------------
    def predict_rows(self, processed_rows, sequence_rows=None):
        """
        Forecast for rows that are already in the processed-feature format.

        `processed_rows`  the 211 engineered features at time t (one row per
                          forecast) - exactly what src/features produces.
        `sequence_rows`   the matching SequenceData for the deep members. If
                          it is None, the deep members are skipped and the
                          ensemble runs on the three tree models only, which
                          the result reports honestly.
        """
        out = {}
        for h in HORIZONS:
            reg_parts, proba_parts, used = [], [], []
            for name in MEMBERS:
                if name in ("bilstm", "cnn_lstm"):
                    continue
                reg_parts.append(self._tabular(name, "regression", h,
                                               processed_rows))
                proba_parts.append(self._tabular(name, "classification", h,
                                                 processed_rows))
                used.append(name)
            if sequence_rows is not None:
                for name in ("bilstm", "cnn_lstm"):
                    r, p = self._deep(name, sequence_rows)
                    j = HORIZONS.index(h)
                    reg_parts.append(r[:, j])
                    proba_parts.append(p[:, j, :])
                    used.append(name)

            reg = np.column_stack(reg_parts)
            proba = np.stack(proba_parts, axis=1)
            # the stacking meta-model expects all five members in order
            complete = used == MEMBERS
            if complete:
                pred_reg = self._stack("regression", h, reg, proba)
                pred_proba = self._stack("classification", h, proba, proba) \
                    if False else self._stack("classification", h, reg, proba)
                method = "stacking"
            else:
                pred_reg = reg.mean(axis=1)
                pred_proba = proba.mean(axis=1)
                method = "average of " + ", ".join(used)

            tau = tau_for("stacking" if complete else "simple_average", h)
            p_low = pred_proba[:, 0]
            if tau is None:
                warn = pred_proba.argmax(axis=1) == 0
            else:
                warn = p_low >= tau
            out[h] = {"do": pred_reg, "proba": pred_proba, "p_low": p_low,
                      "warn": warn, "tau": tau, "method": method,
                      "members": used,
                      "member_reg": dict(zip(used, reg.T)),
                      "member_p_low": dict(zip(used, proba[:, :, 0].T))}
        return out

    # -- convenience: one station, one hour, from the processed test file ---
    def predict_station(self, station, time, split="test"):
        """
        A forecast for one station-hour taken from data/processed. This is
        what the app's Predict tab uses when the user picks a station and a
        time rather than uploading a file.
        """
        from src.models import sequences as S
        df = pd.read_parquet(project_path(CFG["paths"][split]))
        row = df[(df["station"] == str(station))
                 & (df["time"] == pd.Timestamp(time))]
        if not len(row):
            raise KeyError(f"no row for {station} at {time} in the {split} split")
        seq = self._sequence_for(station, time, split)
        res = self.predict_rows(row, seq)
        return {h: {k: (v[0] if isinstance(v, np.ndarray) else v)
                    for k, v in d.items()
                    if k not in ("member_reg", "member_p_low")}
                | {"member_reg": {m: float(a[0]) for m, a in d["member_reg"].items()},
                   "member_p_low": {m: float(a[0]) for m, a in d["member_p_low"].items()}}
                for h, d in res.items()}, row

    def _sequence_for(self, station, time, split):
        """One SequenceData row for the deep members, or None if unavailable."""
        try:
            import joblib
            from src.models import sequences as S
            bundle = joblib.load(os.path.join(self.folder, "dl_scaler.joblib"))
            data = S.prepare(split, scaler=bundle["scaler"],
                             medians=bundle["medians"],
                             stations=[str(station)],
                             seq_len=bundle["seq_len"], with_features=True,
                             feature_scaler=bundle["feature_scaler"],
                             feature_medians=bundle["feature_medians"])
            keep = np.flatnonzero(
                (data.meta["time"] == pd.Timestamp(time)).to_numpy())
            if not len(keep):
                return None
            return _subset(data, keep)
        except Exception:
            return None


def _subset(data, rows):
    """A SequenceData holding only the given rows."""
    from src.models.sequences import SequenceData
    return SequenceData(data.arrays, data.station_ids, data.index[rows],
                        data.meta.iloc[rows].reset_index(drop=True),
                        data.scaler, data.medians, seq_len=data.seq_len,
                        features=None if data.features is None
                        else data.features[rows],
                        feature_scaler=data.feature_scaler,
                        feature_medians=data.feature_medians,
                        feature_names=data.feature_names)


def available_stages():
    """
    Only the stages whose files are actually complete. Stage 3 takes hours
    to train, so the folder can exist while the models are still arriving.
    """
    out = []
    for stage, d in STAGE_DIRS.items():
        folder = project_path(d)
        if not os.path.isdir(folder):
            continue
        files = set(os.listdir(folder))
        needed = {"dl_scaler.joblib"}
        needed |= {f"lightgbm_regression_h{h}.joblib" for h in HORIZONS}
        needed |= {f"{n}_seed{s}.pt" for n in ("bilstm", "cnn_lstm")
                   for s in CFG["dl_search"]["final_seeds"]}
        if needed <= files:
            out.append(stage)
    return out


# ---------------------------------------------------------------------------
# Forecasting from an uploaded window of hourly readings
# ---------------------------------------------------------------------------
HOURLY_TEMPLATE_COLUMNS = None      # filled in on first use


def template_columns():
    """
    The columns an uploaded file needs. This is the format of
    data/interim/hourly.parquet - the water sensors, their missing flags,
    and the weather and air-quality variables - because those are what the
    features are built from. `app/sample_data/` holds two real examples.
    """
    global HOURLY_TEMPLATE_COLUMNS
    if HOURLY_TEMPLATE_COLUMNS is None:
        mask = CFG["cleaning"]["mask_suffix"]
        water = list(CFG["download"]["usgs_params"].values())
        HOURLY_TEMPLATE_COLUMNS = (["time"] + water + [v + mask for v in water]
                                   + CFG["download"]["weather_vars"]
                                   + CFG["download"]["air_vars"])
    return HOURLY_TEMPLATE_COLUMNS


def check_upload(df):
    """What is wrong with this file, as a list of plain sentences."""
    problems = []
    need = set(template_columns())
    missing = sorted(need - set(df.columns))
    if missing:
        problems.append(f"missing {len(missing)} columns, including: "
                        f"{', '.join(missing[:6])}")
    if "time" in df.columns:
        t = pd.to_datetime(df["time"], errors="coerce")
        if t.isna().any():
            problems.append("some timestamps could not be read")
        elif len(t) < 72:
            problems.append(f"only {len(t)} hours of history; at least 72 are "
                            f"needed (the deep models look back 72 h)")
        else:
            gaps = t.sort_values().diff().dropna()
            if (gaps != pd.Timedelta("1h")).any():
                problems.append("the rows are not one hour apart throughout")
    return problems


def predict_from_hourly(forecaster, df, station=None, lat=None, lon=None):
    """
    A forecast made from an uploaded window, using the project's own feature
    code rather than anything reimplemented here.

    The forecast is for the LAST hour in the file. `station` picks which
    station embedding and coordinates to use; if it is not given, the first
    station in stations.csv is used and the result says so.
    """
    from src.features.build_features import build_station
    from src.models import sequences as S

    stations = pd.read_csv(project_path(CFG["paths"]["stations"]),
                           dtype={"site_no": str})
    codes = {s: i for i, s in enumerate(sorted(stations["site_no"]))}
    if station is None or station not in codes:
        station = sorted(stations["site_no"])[0]
    meta = stations[stations["site_no"] == station].iloc[0]
    lat = float(meta["lat"]) if lat is None else lat
    lon = float(meta["lon"]) if lon is None else lon

    g = df.copy()
    g["time"] = pd.to_datetime(g["time"])
    g = g.sort_values("time").set_index("time")

    # 1. the 211 engineered features, from the project's own builder
    feats = build_station(g, station, codes[station], lat, lon)
    last = feats.iloc[[-1]].copy()

    # 2. the sequence for the deep members, from the project's own code
    seq_rows = None
    try:
        import joblib
        bundle = joblib.load(os.path.join(forecaster.folder, "dl_scaler.joblib"))
        seq_len = bundle["seq_len"]
        if len(g) >= seq_len:
            # station_table wants `time` as a column, not an index
            table = S.fill_past_only(
                S.station_table(g.reset_index().assign(station=station)))
            table = table.fillna(bundle["medians"]).fillna(0.0)
            arr = bundle["scaler"].transform(table.to_numpy()).astype("float32")
            fcols = S.engineered_feature_columns()
            fvals = last.reindex(columns=fcols).astype("float32")
            fvals = fvals.fillna(bundle["feature_medians"]).fillna(0.0)
            scaled = bundle["feature_scaler"].transform(fvals.to_numpy())
            meta_rows = pd.DataFrame({
                "station": [station], "time": [g.index[-1]],
                "do_now": [float(g[DO].iloc[-1])],
                **{f"y_reg_{h}": [np.nan] for h in HORIZONS},
                **{f"y_cls_{h}": [np.nan] for h in HORIZONS}})
            seq_rows = S.SequenceData(
                {station: arr}, {station: codes[station]},
                np.array([[0, len(arr) - 1]], dtype="int64"), meta_rows,
                bundle["scaler"], bundle["medians"], seq_len=seq_len,
                features=scaled.astype("float32"),
                feature_scaler=bundle["feature_scaler"],
                feature_medians=bundle["feature_medians"],
                feature_names=fcols)
    except Exception as e:                      # the trees still work alone
        seq_rows = None
        forecaster.last_sequence_error = f"{type(e).__name__}: {e}"

    out = forecaster.predict_rows(last, seq_rows)
    return out, station, g.index[-1], float(g[DO].iloc[-1])
