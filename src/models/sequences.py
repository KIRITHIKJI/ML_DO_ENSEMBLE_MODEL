"""
STEP 4.1 - Turn the hourly table into sequences for the neural networks.

A tree model gets 211 engineered features. A recurrent network is supposed
to find the lags and rolling statistics itself, so it gets the raw hourly
history instead: for a forecast made at time t, the last 48 hours
(t-47 .. t) of

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

    def __init__(self, arrays, station_ids, index, meta, scaler, medians):
        self.arrays = arrays              # station -> (n_hours, n_channels)
        self.station_ids = station_ids    # station -> integer code
        self.index = index                # (n_rows, 2): station slot, row t
        self.meta = meta                  # DataFrame: station, time, targets
        self.scaler = scaler
        self.medians = medians
        self._slots = sorted(arrays)
        self._slot_of = {s: i for i, s in enumerate(self._slots)}
        self._stack = [arrays[s] for s in self._slots]
        self._codes = np.array([station_ids[s] for s in self._slots], dtype="int64")

    def __len__(self):
        return len(self.index)

    def batch(self, row_ids):
        """
        Windows for these rows, as (batch, SEQ_LEN, n_channels) plus the
        station codes. Built with one vectorised gather per station, not a
        Python loop over samples.
        """
        row_ids = np.asarray(row_ids)
        slots = self.index[row_ids, 0]
        ends = self.index[row_ids, 1]
        x = np.empty((len(row_ids), SEQ_LEN, N_CHANNELS), dtype="float32")
        for slot in np.unique(slots):
            here = np.flatnonzero(slots == slot)
            # offsets t-47 .. t for every row of this station at once
            starts = ends[here] - SEQ_LEN + 1
            take = starts[:, None] + np.arange(SEQ_LEN)[None, :]
            x[here] = self._stack[slot][take]
        return x, self._codes[slots]

    def targets(self, row_ids, task, h):
        col = f"{'y_reg' if task == 'regression' else 'y_cls'}_{h}"
        return self.meta[col].to_numpy()[row_ids]

    def do_now(self, row_ids):
        """DO at time t, in the original units: the regression anchor."""
        return self.meta["do_now"].to_numpy()[row_ids]


def prepare(split, scaler=None, medians=None, stations=None):
    """
    Build the sequence data for one split.

    For the training split, pass scaler=None and medians=None: they are
    fitted here on the training rows and returned. For validation, pass the
    ones that came back from the training call - nothing is ever fitted on
    2024.
    """
    panel = pd.read_parquet(project_path(P["hourly"]))
    rows = pd.read_parquet(project_path(P[split]),
                           columns=["station", "time", "station_id", DO]
                                   + [f"y_{k}_{h}" for h in HORIZONS
                                      for k in ["reg", "cls"]])
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
    # full 48-hour history behind them
    slots = sorted(arrays)
    slot_of = {s: i for i, s in enumerate(slots)}
    keep, idx = [], []
    for i, r in enumerate(rows.itertuples()):
        pos = positions[r.station].get(r.time)
        if pos is None or pos < SEQ_LEN - 1:
            continue
        keep.append(i)
        idx.append((slot_of[r.station], pos))

    meta = rows.iloc[keep].reset_index(drop=True)
    meta = meta.rename(columns={DO: "do_now"})
    return SequenceData(arrays, station_ids, np.array(idx, dtype="int64"),
                        meta, scaler, medians)


# ---------------------------------------------------------------------------
# Saving / loading the scaler
# ---------------------------------------------------------------------------
def scaler_path():
    folder = project_path(DL["dirs"]["dl_models"])
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "scaler.joblib")


def save_scaler(scaler, medians):
    joblib.dump({"scaler": scaler, "medians": medians,
                 "channels": CHANNELS, "seq_len": SEQ_LEN}, scaler_path())
    return scaler_path()


def load_scaler():
    return joblib.load(scaler_path())
