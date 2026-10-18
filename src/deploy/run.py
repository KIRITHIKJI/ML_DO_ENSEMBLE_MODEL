"""
STEP 12.5 - a production run.

    python -m src.deploy.run

What it does, in order:

  1. fetch the live window (src.deploy.fetch_live --mode live): the last
     `fetch.live_days` days of USGS water, ERA5 where it reaches and the
     forecast API for the hours it does not, every hour tagged with its
     source;
  2. forecast every station whose reading is recent enough, at its most
     recent hour, for all four horizons, with models/final and the frozen
     warning thresholds;
  3. write results/live/forecast_<UTC timestamp>.csv and append the same
     rows to results/live/log.parquet;
  4. print the counts per alert type, the early warnings, the stations that
     are already Low, the stations whose sensor is offline, and whatever was
     skipped, with the reason.

Two things the rows carry that are worth knowing before reading them:

  `status`      "ok", or "stale - sensor offline" when the station's newest
                DO reading is older than `run.max_data_age_hours`. A stale
                station gets NO predicted DO, NO P(Low) and `warning` False
                - it is never handed to the models at all, because every
                input they read is a lag of the DO the sensor last sent.
  `alert_type`  early_warning (safe water predicted to go Low - the point of
                the system), ongoing_low (already below the threshold and
                the rule fires, which an operator can read off the sensor),
                none, or stale.
  `class_now`   the class of the DO reading the forecast was made from,
                written as "Low", "Medium" or "Good" rather than 0.0, 1.0
                or 2.0. Nobody reading a csv should have to remember which
                end of the scale 0 is. Files written before this change
                carry the numbers, and everything that reads the column
                (`class_code` below, the monitor, the API and the app)
                accepts either.

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
MAX_AGE_HOURS = float(RUN["max_data_age_hours"])
ALERT = RUN["alert_types"]


# ---------------------------------------------------------------------------
# The rows to forecast from
# ---------------------------------------------------------------------------
def latest_rows(features):
    """
    The most recent hour at each station - one forecast per station.
    """
    idx = features.groupby("station")["time"].idxmax()
    return features.loc[sorted(idx)].reset_index(drop=True)


def data_age_hours(times, issued_at):
    """How old each reading is at the moment the forecast is issued."""
    issued_at = pd.Timestamp(issued_at)
    return np.array([(issued_at - pd.Timestamp(t)).total_seconds() / 3600.0
                     for t in times], dtype="float64")


def split_by_age(rows, issued_at, max_age_hours=None):
    """
    (fresh, stale, ages) - which stations have a recent enough reading to
    forecast from.

    A station whose sensor stopped reporting days ago is not a forecasting
    problem, it is an outage. Every feature the models read is a lag or a
    rolling statistic of the DO the sensor last sent, so a stale reading
    produces a confident answer about a river nobody has looked at since,
    and that answer is indistinguishable from a fresh one. The row still
    goes out - silence is worse - but it carries the outage instead of a
    number.
    """
    limit = MAX_AGE_HOURS if max_age_hours is None else float(max_age_hours)
    ages = data_age_hours(rows["time"], issued_at)
    stale = ages > limit
    return (rows[~stale].reset_index(drop=True),
            rows[stale].reset_index(drop=True),
            ages)


# ---------------------------------------------------------------------------
# The class of the water right now, in words
# ---------------------------------------------------------------------------
# A live forecast file is read by people, not only by the scorer, so
# `class_now` goes out as "Low" / "Medium" / "Good". Every forecast written
# before this change carries 0.0 / 1.0 / 2.0 instead, and those files must
# keep working: `class_code` reads both, and it is the only place that
# knows the difference.
# ---------------------------------------------------------------------------
_LABEL_OF_CODE = {i: name for i, name in enumerate(CLASS_NAMES)}
_CODE_OF_LABEL = {name.lower(): i for i, name in enumerate(CLASS_NAMES)}


def class_code(value):
    """
    A DO class as 0 / 1 / 2, from either format, or None when there is no
    class there.

    Accepts the word ("Low", case-insensitively), the code (0, 0.0, and
    "0.0" as a csv read back without a dtype), and every flavour of
    missing. Nothing downstream should test the column for itself.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _CODE_OF_LABEL:
            return _CODE_OF_LABEL[text.lower()]
        try:
            value = float(text)
        except ValueError:
            return None
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def class_label(value):
    """One DO class as a word, from a code, a word, or nothing."""
    code = class_code(value)
    return None if code is None else _LABEL_OF_CODE[code]


def class_labels(do_values):
    """
    DO readings in mg/L -> ["Low", "Medium", "Good", ...]: the `class_now`
    column as it is written to a forecast file.

    A missing reading stays missing - None, which a csv writes as an empty
    field - rather than a word that would claim to know something about a
    river nobody measured.
    """
    return [None if c != c else _LABEL_OF_CODE[int(c)]
            for c in to_class(do_values)]


def classify_alert(class_now, warning, stale=False):
    """
    Which of the four kinds of row this is.

    `class_now` is the class of the DO reading the forecast was made from,
    as a word ("Low") or as the old code (0); `warning` whether the frozen
    rule fired.

      stale          no prediction was made at all
      ongoing_low    the water is ALREADY below the threshold and the rule
                     fires. True, and useful, but it is not an early
                     warning: the operator can read it off the sensor.
      early_warning  the water is currently Medium or Good and the rule
                     says it is likely to go Low. This is the thing the
                     system exists for.
      none           the rule did not fire.

    A warning whose current class is unknown is reported as an early
    warning: between hiding a warning and over-stating one, a warning
    system errs towards saying something.
    """
    if stale:
        return ALERT["stale"]
    if not bool(warning):
        return ALERT["none"]
    if class_code(class_now) == LOW_CLASS:
        return ALERT["ongoing_low"]
    return ALERT["early_warning"]


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
def stale_frame(stale, sources, issued_at, version, max_age_hours=None):
    """
    The rows for stations whose reading is too old to forecast from: one per
    station and horizon, carrying the outage and the data age, with no
    predicted DO, no P(Low), and `warning` False.

    The warning column is False by construction, not by arithmetic. There is
    no probability here to compare with tau, so there is nothing a stale
    row could ever be thresholded into.
    """
    limit = MAX_AGE_HOURS if max_age_hours is None else float(max_age_hours)
    if not len(stale):
        return pd.DataFrame(columns=COLUMNS + ["observed_at", "do_now_mgl",
                                               "class_now", "method"])
    ages = data_age_hours(stale["time"], issued_at)
    frames = []
    for h in HORIZONS:
        frames.append(pd.DataFrame({
            "station": stale["station"].to_numpy(),
            "issued_at": issued_at,
            "horizon": h,
            "valid_at": stale["time"].to_numpy() + pd.Timedelta(hours=h),
            "pred_do_mgl": np.nan,
            "p_low": np.nan,
            "tau": np.nan,
            "warning": False,
            "model_version": version,
            "weather_source": [source_at(sources, s, t) for s, t
                               in zip(stale["station"], stale["time"])],
            "data_age_hours": ages,
            "status": RUN["stale_status"],
            "alert_type": ALERT["stale"],
            "observed_at": stale["time"].to_numpy(),
            "do_now_mgl": stale[D.DO].to_numpy(dtype="float64"),
            "class_now": class_labels(stale[D.DO]),
            "method": f"none - reading older than {limit:g} h",
        }))
    return pd.concat(frames, ignore_index=True)


def forecast(rows, panel, sources, issued_at, batch_size=None,
             max_age_hours=None):
    """
    One row per station, horizon. Returns (frame, info) where `info` lists
    the stations that could not be forecast and why.

    Stations whose newest reading is older than `run.max_data_age_hours` are
    never handed to the models at all - they get their rows from
    `stale_frame` instead. That is deliberately done BEFORE the sequence is
    built, so a dead sensor costs no GPU work and cannot produce a number.
    """
    version = D.model_version()
    limit = MAX_AGE_HOURS if max_age_hours is None else float(max_age_hours)
    fresh, stale, _ = split_by_age(rows, issued_at, limit)
    info = []
    for r in stale.itertuples():
        age = (pd.Timestamp(issued_at)
               - pd.Timestamp(r.time)).total_seconds() / 3600.0
        info.append({"station": r.station, "stage": "stale",
                     "reason": f"last DO reading is {age:.1f} h old, over "
                               f"the {limit:g} h limit - no forecast issued"})
    stale_rows = stale_frame(stale, sources, issued_at, version, limit)

    if not len(fresh):
        frame = stale_rows.sort_values(["station", "horizon"]) \
                          .reset_index(drop=True)
        extra = [c for c in frame.columns if c not in COLUMNS]
        return frame[COLUMNS + extra], info

    folder = D.deployed_folder()
    f = Forecaster(D.DEP["models"]["stage"])
    f.batch_size = int(batch_size or D.COMPUTE["deep_batch_size"])

    seq, keys = S.sequences_for(fresh, panel, folder)
    have = set(zip(keys["station"], keys["time"]))
    for r in fresh.itertuples():
        if (r.station, r.time) not in have:
            info.append({"station": r.station, "stage": "sequence",
                         "reason": "fewer than 72 hours of history behind "
                                   "the latest reading"})
    use = fresh[[(s, t) in have for s, t in zip(fresh["station"],
                                                fresh["time"])]]
    use = use.reset_index(drop=True)
    if not len(use):
        frame = stale_rows.sort_values(["station", "horizon"]) \
                          .reset_index(drop=True)
        extra = [c for c in frame.columns if c not in COLUMNS]
        return (frame[COLUMNS + extra] if len(frame)
                else pd.DataFrame(columns=COLUMNS)), info

    out = f.predict_rows(use, seq)
    class_now = class_labels(use[D.DO])
    ages = data_age_hours(use["time"], issued_at)
    frames = []
    for h in HORIZONS:
        d = out[h]
        warn = np.asarray(d["warn"]).astype(bool)
        frames.append(pd.DataFrame({
            "station": use["station"].to_numpy(),
            "issued_at": issued_at,
            "horizon": h,
            "valid_at": use["time"].to_numpy()
                        + pd.Timedelta(hours=h),
            "pred_do_mgl": np.asarray(d["do"], dtype="float64"),
            "p_low": np.asarray(d["p_low"], dtype="float64"),
            "tau": np.nan if d["tau"] is None else float(d["tau"]),
            "warning": warn,
            "model_version": version,
            "weather_source": [source_at(sources, s, t) for s, t
                               in zip(use["station"], use["time"])],
            "data_age_hours": ages,
            "status": RUN["fresh_status"],
            "alert_type": [classify_alert(c, w)
                           for c, w in zip(class_now, warn)],
            # context, past the required schema: what the river is doing now
            "observed_at": use["time"].to_numpy(),
            "do_now_mgl": use[D.DO].to_numpy(dtype="float64"),
            "class_now": class_now,
            "method": d["method"],
        }))
    frame = pd.concat(frames + ([stale_rows] if len(stale_rows) else []),
                      ignore_index=True)
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


def as_labels(frame):
    """
    A copy of `frame` with `class_now` written as words, whichever format
    it arrived in. Used on the existing log before anything is appended to
    it: a parquet column holding 1.0 in the old rows and "Medium" in the
    new ones has no type, and pyarrow would refuse the file.
    """
    if "class_now" not in frame.columns:
        return frame
    out = frame.copy()
    out["class_now"] = [class_label(v) for v in out["class_now"]]
    return out


def append_log(frame):
    """
    Append to results/live/log.parquet. Re-running the same hour replaces
    that hour's rows rather than doubling them, so an hourly schedule that
    fires twice cannot corrupt the monitoring input.

    A log written before `class_now` became readable is converted as it is
    read, so one file never carries both formats.
    """
    path = project_path(RUN["log_parquet"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = as_labels(frame)
    if os.path.exists(path):
        old = as_labels(pd.read_parquet(path))
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


def counts_by_alert(frame):
    """Rows and stations per alert type, in the order that matters."""
    order = [ALERT["early_warning"], ALERT["ongoing_low"], ALERT["stale"],
             ALERT["none"]]
    out = {}
    for name in order:
        g = frame[frame["alert_type"] == name]
        out[name] = {"rows": int(len(g)),
                     "stations": int(g["station"].nunique())}
    return out


def _warning_table(rows, title):
    D.log(f"\n{title}")
    D.log(f"  {'station':<12}{'h':>4}{'pred DO':>9}{'P(Low)':>9}"
          f"{'tau':>7}{'DO now':>9}{'age h':>7}  weather")
    for r in rows.sort_values(["p_low"], ascending=False).itertuples():
        D.log(f"  {r.station:<12}{r.horizon:>4}"
              f"{r.pred_do_mgl:>9.2f}{r.p_low:>9.1%}"
              f"{r.tau:>7.2f}{r.do_now_mgl:>9.2f}"
              f"{r.data_age_hours:>7.1f}  {r.weather_source}")


def summarise(frame, skipped, path, log_path, n_log):
    D.log("")
    if not len(frame):
        D.log("No station could be forecast this run.")
    else:
        head = 24 if 24 in HORIZONS else HORIZONS[len(HORIZONS) // 2]
        stations = sorted(set(frame["station"]))
        counts = counts_by_alert(frame)
        forecast_rows = frame[frame["alert_type"] != ALERT["stale"]]
        D.log(f"{len(stations)} stations, {len(frame)} rows "
              f"({len(HORIZONS)} horizons each); "
              f"{forecast_rows['station'].nunique()} forecast, "
              f"{counts[ALERT['stale']]['stations']} stale")

        D.log("\nBY ALERT TYPE")
        D.log(f"  {'alert_type':<16}{'rows':>7}{'stations':>10}")
        for name, c in counts.items():
            D.log(f"  {name:<16}{c['rows']:>7}{c['stations']:>10}")

        early = frame[frame["alert_type"] == ALERT["early_warning"]]
        ongoing = frame[frame["alert_type"] == ALERT["ongoing_low"]]
        stale = frame[frame["alert_type"] == ALERT["stale"]]

        # Early warnings first and on their own: safe water predicted to go
        # Low is the only thing here an operator could not read straight off
        # the sensor.
        if len(early):
            _warning_table(
                early,
                f"EARLY WARNINGS - safe water predicted to go Low "
                f"({early['station'].nunique()} stations, {len(early)} "
                f"station-horizons):")
        else:
            D.log("\nNo early warning: no station with safe water now is "
                  "predicted to go Low.")

        if len(ongoing):
            _warning_table(
                ongoing,
                f"ALREADY LOW - the rule fires, but the sensor already says "
                f"so ({ongoing['station'].nunique()} stations, "
                f"{len(ongoing)} station-horizons):")

        if len(stale):
            D.log(f"\nSENSOR OFFLINE - no forecast issued "
                  f"({stale['station'].nunique()} stations, limit "
                  f"{MAX_AGE_HOURS:g} h):")
            D.log(f"  {'station':<12}{'age h':>9}  last reading")
            for r in (stale[stale["horizon"] == HORIZONS[0]]
                      .sort_values("data_age_hours", ascending=False)
                      .itertuples()):
                D.log(f"  {r.station:<12}{r.data_age_hours:>9.1f}  "
                      f"{pd.Timestamp(r.observed_at):%Y-%m-%d %H:%M} UTC")

        at_head = forecast_rows[forecast_rows["horizon"] == head]
        if len(at_head):
            D.log(f"\nAt {head} h: "
                  f"{int(at_head['warning'].sum())} of {len(at_head)} "
                  f"forecast stations warned "
                  f"({int((at_head['alert_type'] == ALERT['early_warning']).sum())}"
                  f" early, "
                  f"{int((at_head['alert_type'] == ALERT['ongoing_low']).sum())}"
                  f" already low); oldest reading used "
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
