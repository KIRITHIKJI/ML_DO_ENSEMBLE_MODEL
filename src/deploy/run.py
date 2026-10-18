"""
STEP 12.5 - a production run.

    python -m src.deploy.run

What it does, in order:

  1. fetch the live window (src.deploy.fetch_live --mode live): the last
     `fetch.live_days` days of USGS water, ERA5 where it reaches and the
     forecast API for the hours it does not, every hour tagged with its
     source;
  2. forecast every station at its most recent hour, for all four horizons,
     with models/final and the frozen warning thresholds;
  3. write results/live/forecast_<UTC timestamp>.csv and append the same
     rows to results/live/log.parquet;
  4. print which stations are under warning now and which were skipped,
     with the reason.

Nothing is trained and nothing is tuned. The forecast comes from
`predict_service.Forecaster`, the same object the app uses, so a row in the
csv is the same number the app would show. Every row carries the model
version id - the SHA-256 over the per-file hashes of models/final - so a
forecast can always be traced to the exact models that made it.

Schedule it hourly with Windows Task Scheduler; see run_live.bat and the
README.
"""

import argparse
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from src.config import project_path
from src.evaluation.metrics import to_class
from src.models.predict_service import Forecaster

from src.deploy import common as D
from src.deploy import fetch_live as F
from src.deploy import score_2026 as S

CFG = D.CFG
HORIZONS = D.HORIZONS
CLASS_NAMES = D.CLASS_NAMES
LOW_CLASS = D.LOW_CLASS
RUN = D.RUN
COLUMNS = list(RUN["columns"])


# ---------------------------------------------------------------------------
# The rows to forecast from
# ---------------------------------------------------------------------------
def latest_rows(features):
    """
    The most recent hour at each station - one forecast per station. A
    station whose newest reading is hours old still gets a forecast; the
    `data_age_hours` column says how stale it is, which is the operator's
    decision to make, not ours to hide.
    """
    idx = features.groupby("station")["time"].idxmax()
    return features.loc[sorted(idx)].reset_index(drop=True)


def source_at(sources, station, time):
    """The weather source tag for one station-hour."""
    if sources is None:
        return F.TAGS["missing"]
    hit = sources[(sources["station"] == station)
                  & (sources["time"] == time)]
    return str(hit["weather_source"].iloc[0]) if len(hit) \
        else F.TAGS["missing"]


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
def forecast(rows, panel, sources, issued_at, batch_size=None):
    """
    One row per station, horizon. Returns (frame, info) where `info` lists
    the stations that could not be forecast and why.
    """
    folder = D.deployed_folder()
    version = D.model_version()
    f = Forecaster(D.DEP["models"]["stage"])
    f.batch_size = int(batch_size or D.COMPUTE["deep_batch_size"])

    seq, keys = S.sequences_for(rows, panel, folder)
    info = []
    have = set(zip(keys["station"], keys["time"]))
    for r in rows.itertuples():
        if (r.station, r.time) not in have:
            info.append({"station": r.station, "stage": "sequence",
                         "reason": "fewer than 72 hours of history behind "
                                   "the latest reading"})
    use = rows[[(s, t) in have for s, t in zip(rows["station"],
                                               rows["time"])]]
    use = use.reset_index(drop=True)
    if not len(use):
        return pd.DataFrame(columns=COLUMNS), info

    out = f.predict_rows(use, seq)
    frames = []
    for j, h in enumerate(HORIZONS):
        d = out[h]
        p_low = np.asarray(d["p_low"], dtype="float64")
        tau = d["tau"]
        frames.append(pd.DataFrame({
            "station": use["station"].to_numpy(),
            "issued_at": issued_at,
            "horizon": h,
            "valid_at": use["time"].to_numpy()
                        + pd.Timedelta(hours=h),
            "pred_do_mgl": np.asarray(d["do"], dtype="float64"),
            "p_low": p_low,
            "tau": np.nan if tau is None else float(tau),
            "warning": np.asarray(d["warn"]).astype(bool),
            "model_version": version,
            "weather_source": [source_at(sources, s, t) for s, t
                               in zip(use["station"], use["time"])],
            "data_age_hours": [(issued_at - pd.Timestamp(t)).total_seconds()
                               / 3600.0 for t in use["time"]],
            # context, past the required schema: what the river is doing now
            "observed_at": use["time"].to_numpy(),
            "do_now_mgl": use[D.DO].to_numpy(dtype="float64"),
            "class_now": to_class(use[D.DO]),
            "method": d["method"],
        }))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(["station", "horizon"]).reset_index(drop=True)
    # the pre-registered columns first, the context after them
    extra = [c for c in frame.columns if c not in COLUMNS]
    return frame[COLUMNS + extra], info


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def forecast_dir():
    return D.ensure_dir(RUN["results_dir"])


def write_forecast(frame, issued_at):
    stamp = pd.Timestamp(issued_at).strftime(RUN["stamp_format"])
    path = os.path.join(forecast_dir(), RUN["forecast_file"].format(stamp=stamp))
    frame.to_csv(path, index=False)
    return path


def append_log(frame):
    """
    Append to results/live/log.parquet. Re-running the same hour replaces
    that hour's rows rather than doubling them, so an hourly schedule that
    fires twice cannot corrupt the monitoring input.
    """
    path = project_path(RUN["log_parquet"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = frame.copy()
    if os.path.exists(path):
        old = pd.read_parquet(path)
        keys = set(zip(new["station"], new["issued_at"], new["horizon"]))
        keep = [k not in keys for k in zip(old["station"], old["issued_at"],
                                           old["horizon"])]
        new = pd.concat([old[keep], new], ignore_index=True)
    new = new.sort_values(["issued_at", "station", "horizon"]) \
             .reset_index(drop=True)
    new.to_parquet(path, index=False)
    return path, len(new)


def latest_forecast_path():
    """The newest forecast file, or None."""
    import glob
    files = sorted(glob.glob(os.path.join(forecast_dir(),
                                          RUN["forecast_glob"])))
    return files[-1] if files else None


def load_latest():
    """The newest forecast file as a DataFrame, or None."""
    path = latest_forecast_path()
    if path is None:
        return None, None
    df = pd.read_csv(path, dtype={"station": str})
    for c in ("issued_at", "valid_at", "observed_at"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df, path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--no-fetch", action="store_true",
                    help="use the live data already on disk")
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    st = D.Stages()
    issued_at = pd.Timestamp.utcnow().tz_localize(None).floor("s")
    D.log(f"=== live run, issued {issued_at:%Y-%m-%d %H:%M:%S} UTC ===")
    D.report_device("deep inference")
    if not D.require_cuda_or_explain() and not args.allow_cpu:
        raise SystemExit(2)
    D.log(f"  model version {D.model_version()} "
          f"({D.DEP['models']['stage']})")

    skipped = []
    if not args.no_fetch:
        with st("fetch"):
            res = F.run("live", stations=args.stations)
        skipped += res.get("skipped", [])

    fpath = D.dpath(D.DATA["features"], mode="live")
    hpath = D.dpath(D.DATA["hourly"], mode="live")
    spath = D.dpath(D.DATA["sources"], mode="live")
    if not os.path.exists(fpath):
        D.log("\nNo live features on disk - nothing to forecast. "
              "Run without --no-fetch.")
        raise SystemExit(1)

    features = pd.read_parquet(fpath)
    panel = pd.read_parquet(hpath)
    sources = pd.read_parquet(spath) if os.path.exists(spath) else None
    if args.stations:
        keep = [str(s) for s in args.stations]
        features = features[features["station"].isin(keep)]
        panel = panel[panel["station"].isin(keep)]

    with st("forecast"):
        rows = latest_rows(features)
        frame, info = forecast(rows, panel, sources, issued_at,
                               args.batch_size)
    skipped += info

    with st("write"):
        path = write_forecast(frame, issued_at)
        log_path, n_log = append_log(frame)

    summarise(frame, skipped, path, log_path, n_log)
    st.report()
    return frame


def summarise(frame, skipped, path, log_path, n_log):
    D.log("")
    if not len(frame):
        D.log("No station could be forecast this run.")
    else:
        head = 24 if 24 in HORIZONS else HORIZONS[len(HORIZONS) // 2]
        warn_now = frame[frame["warning"]]
        stations = sorted(set(frame["station"]))
        D.log(f"{len(stations)} stations forecast, {len(frame)} rows "
              f"({len(HORIZONS)} horizons each)")
        if len(warn_now):
            D.log(f"\nUNDER WARNING NOW ({warn_now['station'].nunique()} "
                  f"stations, {len(warn_now)} station-horizons):")
            D.log(f"  {'station':<12}{'h':>4}{'pred DO':>9}{'P(Low)':>9}"
                  f"{'tau':>7}{'DO now':>9}{'age h':>7}  weather")
            for r in warn_now.sort_values(
                    ["p_low"], ascending=False).itertuples():
                D.log(f"  {r.station:<12}{r.horizon:>4}"
                      f"{r.pred_do_mgl:>9.2f}{r.p_low:>9.1%}"
                      f"{r.tau:>7.2f}{r.do_now_mgl:>9.2f}"
                      f"{r.data_age_hours:>7.1f}  {r.weather_source}")
        else:
            D.log("\nNo station is under warning at any horizon.")
        at_head = frame[frame["horizon"] == head]
        D.log(f"\nAt {head} h: "
              f"{int(at_head['warning'].sum())} of {len(at_head)} stations "
              f"warned; oldest reading "
              f"{at_head['data_age_hours'].max():.1f} h old; weather "
              + ", ".join(f"{k} {v}" for k, v in
                          at_head["weather_source"].value_counts().items()))
    if skipped:
        D.log(f"\nSKIPPED ({len(skipped)}):")
        for s in skipped:
            D.log(f"  {s['station']}  ({s['stage']}) {s['reason']}")
    else:
        D.log("\nNo station skipped.")
    D.log(f"\nwrote {D.shortpath(path)}")
    D.log(f"appended to {D.shortpath(log_path)} "
          f"({n_log:,} rows total)")


if __name__ == "__main__":
    main()
