"""
Build do_demo_pack.npz - a small, REAL subset of the processed dataset for the
Colab training notebook (02_DO_End_to_End_Training_GPU_Colab.ipynb).

    python make_demo_pack.py            # every 4th hour of all 30 stations

Contents: the 211 engineered features + targets for every STEP-th hour of
2023 (train), 2024 (validation) and 2025 (test), and the raw hourly channel
history every station needs for the deep models' 72-hour windows. Nothing is
synthetic; rows are only thinned so training takes minutes instead of an hour.
"""
import sys, json
import numpy as np
import pandas as pd
from src.config import load_config, project_path
from src.models import common as C
from src.models import sequences as S

STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 4
CFG = load_config()
FEATURES = C.feature_columns()
H = CFG["horizons"]
TARGETS = [f"y_{k}_{h}" for h in H for k in ["reg", "cls"]]

panel = pd.read_parquet(project_path(CFG["paths"]["hourly"]))
arrays, start, stations = [], [], sorted(panel["station"].unique())
for st in stations:
    tab = S.fill_past_only(S.station_table(panel[panel.station == st]))      # project code: past-only fill
    assert list(tab.columns) == S.CHANNELS
    arrays.append(tab.to_numpy(dtype="float32")); start.append(tab.index[0])
lengths = [len(a) for a in arrays]
offsets = np.r_[0, np.cumsum(lengths)[:-1]]
channels = np.concatenate(arrays).astype("float32")   # float16 overflows on discharge

out = {"channels": channels, "channel_names": np.array(S.CHANNELS), "stations": np.array(stations),
       "offsets": offsets, "lengths": np.array(lengths), "start": np.array([str(s) for s in start]),
       "feature_names": np.array(FEATURES), "target_names": np.array(TARGETS)}
for split in ["train", "val", "test"]:
    d = C.load_split(split)
    d = d.sort_values(["station", "time"])
    d = d[d.groupby("station").cumcount() % STEP == 0].reset_index(drop=True)
    sidx = d["station"].map({s: i for i, s in enumerate(stations)}).to_numpy()
    st0 = pd.to_datetime(np.array([str(start[i]) for i in sidx]))
    pos = ((d["time"].to_numpy() - st0.to_numpy()) / np.timedelta64(1, "h")).astype(int)
    out[f"{split}_X"] = d[FEATURES].to_numpy(dtype="float32")
    out[f"{split}_y"] = d[TARGETS].astype("float64").to_numpy(dtype="float32")
    out[f"{split}_do_now"] = d[CFG["target"]["column"]].to_numpy(dtype="float32")
    out[f"{split}_station"] = sidx.astype("int16")
    out[f"{split}_pos"] = pos.astype("int32")
    out[f"{split}_time"] = d["time"].to_numpy().astype("datetime64[h]")
    print(f"{split}: {len(d):,} rows")
np.savez_compressed("do_demo_pack.npz", **out)
import os; print(f"do_demo_pack.npz: {os.path.getsize('do_demo_pack.npz') / 1e6:.0f} MB")
