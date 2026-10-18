"""
STEP 4.1 - Turn the hourly table into sequences for the neural networks.

A tree model gets 211 engineered features. A recurrent network is supposed
to find the lags and rolling statistics itself, so it gets the raw hourly
history instead: for a forecast made at time t, the last N hours
(t-(N-1) .. t, N = deep_learning.sequence_length, 48 by default) of

    6 water variables + their 6 "was missing" flags
    DO saturation (DO / solubility at the current water temperature)
    8 weather variables
    8 air-quality variables
    hour and day-of-year as sin/cos

plus the station, which goes into a small embedding rather than the sequence.

Two things matter here:

  * The rows are EXACTLY the rows of data/processed/{train,val}.parquet, the
    same station/time keys with the same targets and the same purge, so the
    deep models and the tree models are scored on identical rows.
  * Nothing is built from the future. Missing values are forward filled
    within a station (never backwards), and anything still missing is filled
    with the TRAINING median. The scaler is fitted on training rows only.

Sequences are never all held in memory: one array stays per station and each
batch of windows is cut out with vectorised numpy indexing.
"""

import os

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, project_path
from src.features.build_features import do_solubility

CFG = load_config()
P = CFG["paths"]
DL = CFG["deep_learning"]
MASK = CFG["cleaning"]["mask_suffix"]
HORIZONS = CFG["horizons"]
SEQ_LEN = DL["sequence_length"]
DO = CFG["target"]["column"]

WATER = DL["water_vars"]
WEATHER = CFG["download"]["weather_vars"]
AIR = CFG["download"]["air_vars"]


def channel_names():
    """The variables in each sequence, in a fixed order."""
    names = list(WATER)
    if DL["use_missing_flags"]:
        names += [v + MASK for v in WATER]
    if DL["use_do_saturation"]:
        names += ["do_saturation"]
    names += list(WEATHER) + list(AIR)
    if DL["use_time_features"]:
        names += ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    return names


CHANNELS = channel_names()
N_CHANNELS = len(CHANNELS)


# ---------------------------------------------------------------------------
# Building the per-station arrays
# ---------------------------------------------------------------------------
def station_table(g):
    """
    One station's hourly rows as the channel table above, on its complete
    hourly index. `g` is a slice of data/interim/hourly.parquet.
    """
    g = g.sort_values("time").set_index("time")
    out = {}
    for v in WATER:
        out[v] = g[v]
        if DL["use_missing_flags"]:
            out[v + MASK] = g[v + MASK].astype("float32")
    if DL["use_do_saturation"]:
        out["do_saturation"] = g[DO] / do_solubility(g["water_temp_c"])
    for v in WEATHER + AIR:
        out[v] = g[v]
    if DL["use_time_features"]:
        t = g.index
        out["hour_sin"] = pd.Series(np.sin(2 * np.pi * t.hour / 24), index=t)
        out["hour_cos"] = pd.Series(np.cos(2 * np.pi * t.hour / 24), index=t)
        out["doy_sin"] = pd.Series(np.sin(2 * np.pi * t.dayofyear / 365.25), index=t)
        out["doy_cos"] = pd.Series(np.cos(2 * np.pi * t.dayofyear / 365.25), index=t)
    return pd.DataFrame(out)[CHANNELS]


def fill_past_only(df):
    """
    Forward fill inside one station: a hole is filled with the last value
    before it, never with a value after it. Hours before a station's first
    reading stay NaN and are dealt with by the training median.
    """
    return df.ffill()


class SequenceData:
    """
    Holds one float32 array per station plus an index of (station, t) rows,
    and cuts windows out on demand.

    Build it once with `prepare()`, then ask for batches with `batch()`.
    """

    def __init__(self, arrays, station_ids, index, meta, scaler, medians,
                 seq_len=None, features=None, feature_scaler=None,
                 feature_medians=None, feature_names=None):
        self.arrays = arrays              # station -> (n_hours, n_channels)
        self.station_ids = station_ids    # station -> integer code
        self.index = index                # (n_rows, 2): station slot, row t
        self.meta = meta                  # DataFrame: station, time, targets
        self.scaler = scaler
        self.medians = medians
        self.seq_len = SEQ_LEN if seq_len is None else int(seq_len)
        # The step-2 engineered features at time t, already scaled, or None.
        # One row per row of `meta`, in the same order.
        self.features = features
        self.feature_scaler = feature_scaler
        self.feature_medians = feature_medians
        self.feature_names = feature_names
        self._slots = sorted(arrays)
        self._slot_of = {s: i for i, s in enumerate(self._slots)}
        self._stack = [arrays[s] for s in self._slots]
        self._codes = np.array([station_ids[s] for s in self._slots], dtype="int64")

    def __len__(self):
        return len(self.index)

    def batch(self, row_ids):
        """
        Windows for these rows, as (batch, seq_len, n_channels), the station
        codes, and the engineered features at time t (or None). Built with
        one vectorised gather per station, not a Python loop over samples.
        """
        row_ids = np.asarray(row_ids)
        slots = self.index[row_ids, 0]
        ends = self.index[row_ids, 1]
        L = self.seq_len
        x = np.empty((len(row_ids), L, N_CHANNELS), dtype="float32")
        for slot in np.unique(slots):
            here = np.flatnonzero(slots == slot)
            # offsets t-(L-1) .. t for every row of this station at once
            starts = ends[here] - L + 1
            take = starts[:, None] + np.arange(L)[None, :]
            x[here] = self._stack[slot][take]
        feats = None if self.features is None else self.features[row_ids]
        return x, self._codes[slots], feats

    def targets(self, row_ids, task, h):
        col = f"{'y_reg' if task == 'regression' else 'y_cls'}_{h}"
        return self.meta[col].to_numpy()[row_ids]

    def do_now(self, row_ids):
        """DO at time t, in the original units: the regression anchor."""
        return self.meta["do_now"].to_numpy()[row_ids]


def engineered_feature_columns():
    """The 211 step-2 features, in the same fixed order the tree models use."""
    import json
    with open(project_path(P["feature_groups"]), encoding="utf-8") as f:
        groups = json.load(f)
    return [c for g in ["water", "weather", "air", "time", "station"]
            for c in groups[g]]


def prepare(split, scaler=None, medians=None, stations=None, seq_len=None,
            with_features=False, feature_scaler=None, feature_medians=None,
            rows=None, panel=None):
    """
    Build the sequence data for one split.

    For the training split, pass scaler=None and medians=None: they are
    fitted here on the training rows and returned. For validation, pass the
    ones that came back from the training call - nothing is ever fitted on
    2024.

    `with_features=True` also loads the 211 engineered features at time t
    (step 5's hybrid input). They come straight out of
    data/processed/<split>.parquet, which step 2 built using only data at or
    before t, so they carry no future information. Their NaNs are filled
    with the TRAIN median and they are scaled by a StandardScaler fitted on
    TRAIN - a second scaler, separate from the one for the sequence.

    `rows` and `panel` let a caller hand in tables it already has in memory
    instead of naming a split in config.yaml. Step 12 uses this to build
    sequences for the 2026 and live windows through exactly this function,
    so the deep members' inputs are assembled the same way in deployment as
    in training. `split` is ignored when both are given.
    """
    seq_len = SEQ_LEN if seq_len is None else int(seq_len)
    feature_cols = engineered_feature_columns() if with_features else []
    needed = (["station", "time", "station_id", DO]
              + [f"y_{k}_{h}" for h in HORIZONS for k in ["reg", "cls"]]
              + [c for c in feature_cols if c not in ("station_id", DO)])
    if panel is None:
        panel = pd.read_parquet(project_path(P["hourly"]))
    if rows is None:
        rows = pd.read_parquet(project_path(P[split]), columns=needed)
    else:
        rows = rows[needed].copy()
        # a station that was dropped while building features has no row and
        # therefore no station code, so it must not stay in the panel either
        panel = panel[panel["station"].isin(set(rows["station"]))]
    if stations is not None:                     # --quick mode
        rows = rows[rows["station"].isin(stations)]
        panel = panel[panel["station"].isin(stations)]

    arrays, station_ids, positions = {}, {}, {}
    for station, g in panel.groupby("station", sort=True):
        tab = fill_past_only(station_table(g))
        arrays[station] = tab
        positions[station] = {t: i for i, t in enumerate(tab.index)}
    station_ids = (rows.groupby("station")["station_id"].first().astype(int)
                   .to_dict())

    # medians from TRAIN rows only
    if medians is None:
        train_rows = pd.concat([arrays[s] for s in arrays])
        medians = train_rows.median(numeric_only=True)
    for s in arrays:
        arrays[s] = arrays[s].fillna(medians)

    # scaler from TRAIN rows only
    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(pd.concat([arrays[s] for s in arrays]).to_numpy())

    for s in arrays:
        arrays[s] = scaler.transform(arrays[s].to_numpy()).astype("float32")

    # the row index: only rows that exist in the processed split AND have a
    # full window of history behind them
    slots = sorted(arrays)
    slot_of = {s: i for i, s in enumerate(slots)}
    keep, idx = [], []
    for i, r in enumerate(rows.itertuples()):
        pos = positions[r.station].get(r.time)
        if pos is None or pos < seq_len - 1:
            continue
        keep.append(i)
        idx.append((slot_of[r.station], pos))

    meta = rows.iloc[keep].reset_index(drop=True)

    # the hybrid input: the engineered features at time t
    features = None
    if with_features:
        f = meta[feature_cols].astype("float32")
        if feature_medians is None:          # TRAIN only
            feature_medians = f.median(numeric_only=True)
        f = f.fillna(feature_medians)
        # a column that is NaN for every training row has no median; 0 after
        # scaling means "no information", which is what it is
        f = f.fillna(0.0)
        if feature_scaler is None:           # TRAIN only
            from sklearn.preprocessing import StandardScaler
            feature_scaler = StandardScaler().fit(f.to_numpy())
        features = feature_scaler.transform(f.to_numpy()).astype("float32")

    meta = meta.rename(columns={DO: "do_now"})
    return SequenceData(arrays, station_ids, np.array(idx, dtype="int64"),
                        meta, scaler, medians, seq_len=seq_len,
                        features=features, feature_scaler=feature_scaler,
                        feature_medians=feature_medians,
                        feature_names=feature_cols or None)


# ---------------------------------------------------------------------------
# Saving / loading the scaler
# ---------------------------------------------------------------------------
def scaler_path():
    folder = project_path(DL["dirs"]["dl_models"])
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "scaler.joblib")


def save_scaler(scaler, medians, seq_len=None, feature_scaler=None,
                feature_medians=None, feature_names=None):
    """Everything needed to rebuild the inputs for a saved model."""
    joblib.dump({"scaler": scaler, "medians": medians,
                 "channels": CHANNELS,
                 "seq_len": SEQ_LEN if seq_len is None else seq_len,
                 "feature_scaler": feature_scaler,
                 "feature_medians": feature_medians,
                 "feature_names": feature_names}, scaler_path())
    return scaler_path()


def load_scaler():
    return joblib.load(scaler_path())
