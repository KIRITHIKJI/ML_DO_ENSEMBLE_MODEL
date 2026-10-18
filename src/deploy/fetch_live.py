"""
STEP 12.2 - data for 2026 and for live running.

    python -m src.deploy.fetch_live --mode history      # the 2026 test window
    python -m src.deploy.fetch_live --mode live         # the last N days
    python -m src.deploy.fetch_live --mode penalty      # ERA5 vs forecast

Three modes, one code path:

  history   2026-01-01 .. the last complete ERA5 day. The SAME sources as
            training - USGS instantaneous values, the Open-Meteo ERA5
            archive, CAMS - fetched with `src/data/download`'s own fetchers,
            so the data the 2026 test is scored on is made the same way the
            training data was.

  penalty   the last `test_2026.penalty_days` days of the 2026 window, built
            TWICE - once on ERA5 weather and once on the forecast API's
            recent past - so the live-data penalty can be measured on the
            same hours.

  live      the last `fetch.live_days` days up to now: enough history for a
            72-hour sequence and for every lag and rolling window (the
            longest is a 168-hour rain sum). ERA5 runs about five days
            behind, so the hours it does not cover come from the Open-Meteo
            forecast API's recent past (model ecmwf_ifs025, the one that
            carries soil_moisture_0_to_7cm) and the CAMS forecast feed.
            EVERY HOUR IS TAGGED with the source it came from.

Everything is written under data/live/ - never into data/raw or
data/processed, which belong to the paper.

Robustness: a station whose sensor is offline, whose API call fails, or
whose file is missing a column is SKIPPED with a logged reason and the run
carries on. Requests are retried with exponential backoff and jitter, and
at most `fetch.max_workers` stations are in flight at once so the free USGS
and Open-Meteo endpoints are not hammered. Everything already on disk and
already covering the window is reused, so a re-run only fetches what is
missing.
"""

import argparse
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

from src.config import project_path
from src.data import clean as CLEAN
from src.data import download as DLD
from src.features.build_features import build_station

from src.deploy import common as D

CFG = D.CFG
FETCH = D.FETCH
DATA = D.DATA
TAGS = FETCH["source_tags"]

WATER_VARS = list(CFG["download"]["usgs_params"].values())
WEATHER_VARS = CFG["download"]["weather_vars"]
AIR_VARS = CFG["download"]["air_vars"]
MASK = CFG["cleaning"]["mask_suffix"]
WARMUP = int(CFG["features"]["warmup_hours"])

_local = threading.local()
_pace = threading.Lock()
_last_call = [0.0]


# ---------------------------------------------------------------------------
# HTTP, politely
# ---------------------------------------------------------------------------
def session():
    """One requests session per worker thread."""
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = CFG["download"]["user_agent"]
        _local.session = s
    return s


def _pace_calls():
    """Keep a minimum gap between calls across all workers."""
    gap = float(FETCH["min_seconds_between_calls"])
    if gap <= 0:
        return
    with _pace:
        wait = gap - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


def get(url, params):
    """
    GET with retries and exponential backoff plus jitter. Returns the
    response or None. 400 and 404 mean "no such data", so they come straight
    back rather than being retried.
    """
    tries = int(FETCH["retries"])
    delay = float(FETCH["backoff_seconds"])
    cap = float(FETCH["backoff_max_seconds"])
    for k in range(tries):
        _pace_calls()
        try:
            r = session().get(url, params=params, timeout=FETCH["timeout"])
            if r.status_code == 200:
                return r
            if r.status_code in (400, 404):
                return r
            why = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            why = f"{type(e).__name__}: {e}"
        if k == tries - 1:
            D.log(f"      giving up after {tries} tries ({why}): {url}")
            return None
        sleep = min(cap, delay * (2 ** k)) * (0.5 + random.random())
        D.log(f"      {why} - retry {k + 1}/{tries} in {sleep:.1f}s")
        time.sleep(sleep)
    return None


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------
def era5_cutoff(now=None):
    """
    The last day for which the ERA5 archive can be trusted: today (UTC)
    minus `test_2026.era5_lag_days`. Open-Meteo's archive endpoint will
    happily serve the days after it from a forecast model, which is exactly
    what must not end up in the 2026 test.
    """
    now = pd.Timestamp.utcnow().tz_localize(None) if now is None else now
    lag = int(D.TEST26["era5_lag_days"])
    return (pd.Timestamp(now).normalize() - pd.Timedelta(days=lag)).normalize()


def window(mode, now=None):
    """
    (start, end, fetch_start, water_end) for this mode, all UTC, `end`
    inclusive at 23:00.

    history   the pre-registered 2026 window, never past `end_frozen`
    live      the last `live_days` days up to the current hour
    """
    now = pd.Timestamp.utcnow().tz_localize(None).floor("h") if now is None \
        else pd.Timestamp(now).floor("h")
    if mode == "history":
        start = pd.Timestamp(D.TEST26["start"])
        frozen = pd.Timestamp(D.TEST26["end_frozen"])
        end_day = min(era5_cutoff(now), frozen)
        end = end_day + pd.Timedelta("23h")
    elif mode == "live":
        start = (now - pd.Timedelta(days=int(FETCH["live_days"]))).normalize()
        end = now
    elif mode == "penalty":
        # the last `penalty_days` days of the 2026 test window, which is the
        # stretch the forecast feed can still reach back over
        _, end, _, _ = window("history", now)
        start = (end.normalize()
                 - pd.Timedelta(days=int(D.TEST26["penalty_days"]) - 1))
    else:
        raise ValueError(f"mode must be 'history', 'live' or 'penalty', "
                         f"got {mode!r}")
    fetch_start = start - pd.Timedelta(hours=int(FETCH["preroll_hours"]))
    water_end = end + pd.Timedelta(days=int(FETCH["water_tail_days"]))
    return start, end, fetch_start, min(water_end, now + pd.Timedelta("1D"))


# ---------------------------------------------------------------------------
# Cached raw files
# ---------------------------------------------------------------------------
def raw_path(kind, site):
    return os.path.join(project_path(DATA[kind]), f"{site}.csv")


def read_cached(path):
    if not os.path.exists(path):
        return None
    try:
        df = DLD.read_csv_time(path)
        return df if len(df) else None
    except Exception:
        return None


def missing_ranges(have, start, end):
    """
    Which parts of [start, end] a cached file does not cover. Only the two
    ends are considered - an interior hole is a real gap in the source, and
    re-asking for it every run would be a wasted call every hour.
    """
    if have is None:
        return [(start, end)]
    lo, hi = have.index.min(), have.index.max()
    gaps = []
    if lo > start + pd.Timedelta("1h"):
        gaps.append((start, min(lo, end)))
    if hi < end - pd.Timedelta("1h"):
        gaps.append((max(hi, start), end))
    return [(a, b) for a, b in gaps if b > a]


def chunks(start, end):
    """Split a long range into requests of at most `chunk_days` days."""
    step = pd.Timedelta(days=int(FETCH["chunk_days"]))
    out, a = [], pd.Timestamp(start)
    end = pd.Timestamp(end)
    while a <= end:
        b = min(end, a + step)
        out.append((a, b))
        a = b + pd.Timedelta("1h")
    return out


def save(df, kind, site, order=None):
    path = raw_path(kind, site)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if order:
        for c in order:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[order]
    df.to_csv(path)
    return path


# ---------------------------------------------------------------------------
# The fetchers. The archive ones are the project's own.
# ---------------------------------------------------------------------------
def fetch_water(site, start, end):
    """USGS instantaneous values - download.py's own fetcher."""
    frames = []
    for a, b in chunks(start, end):
        frames.append(DLD.fetch_water(site, a, b))
    return DLD.merge_frames(frames)


def fetch_archive(kind, lat, lon, start, end):
    """ERA5 weather or CAMS air quality - download.py's own fetcher."""
    frames = []
    for a, b in chunks(start, end):
        frames.append(DLD.fetch_open_meteo(kind, lat, lon, a, b))
    return DLD.merge_frames(frames)


def fetch_forecast(kind, lat, lon, past_days):
    """
    The recent past from the forecast feeds, for the hours ERA5 does not
    cover yet. A new endpoint, so it is configured in config_deploy.yaml and
    not in config.yaml.
    """
    if kind == "weather":
        url = FETCH["urls"]["open_meteo_forecast"]
        params = {"latitude": lat, "longitude": lon,
                  "hourly": ",".join(WEATHER_VARS),
                  "models": FETCH["weather_model_recent"],
                  "past_days": int(past_days), "forecast_days": 1,
                  "timezone": "GMT"}
    else:
        url = FETCH["urls"]["open_meteo_air"]
        params = {"latitude": lat, "longitude": lon,
                  "hourly": ",".join(AIR_VARS),
                  "past_days": int(past_days), "forecast_days": 1,
                  "timezone": "GMT"}
    r = get(url, params)
    if r is None or r.status_code != 200:
        return None
    try:
        df = pd.DataFrame(r.json()["hourly"])
    except Exception:
        return None
    if "time" not in df.columns:
        return None
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    # the forecast model names a column after itself when `models` is set
    suffix = "_" + str(FETCH["weather_model_recent"])
    df.columns = [c[:-len(suffix)] if c.endswith(suffix) else c
                  for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


# ---------------------------------------------------------------------------
# One station
# ---------------------------------------------------------------------------
class Skip(Exception):
    """This station cannot be used; the reason travels with the exception."""


def fetch_station(site, lat, lon, mode, win):
    """
    Everything for one station. Returns a dict with the station's hourly
    table, the per-hour source tags, and a log line. Raises Skip with a
    reason when the station cannot be used - the caller logs it and carries
    on with the others.
    """
    start, end, fetch_start, water_end = win
    t0 = time.perf_counter()
    notes = []

    # ---- water -----------------------------------------------------------
    have = read_cached(raw_path("water", site))
    gaps = missing_ranges(have, fetch_start, water_end)
    new = [fetch_water(site, a, b) for a, b in gaps] if gaps else []
    water = DLD.merge_frames(([have] if have is not None else []) + new)
    if water is None or not len(water):
        raise Skip("no USGS water data returned for the window")
    water = water[(water.index >= fetch_start) & (water.index <= water_end)]
    if gaps:
        save(water, "water", site, WATER_VARS)
        notes.append(f"water +{sum(len(f) for f in new if f is not None)} rows")
    else:
        notes.append("water cached")
    for col in CFG["download"]["required_params"]:
        name = CFG["download"]["usgs_params"][col] \
            if col in CFG["download"]["usgs_params"] else str(col)
        if name not in water.columns:
            raise Skip(f"required column {name} missing from the USGS reply")
    do = water[D.DO] if D.DO in water.columns else None
    if do is None or do.notna().sum() == 0:
        raise Skip("the DO sensor reported nothing in the window")

    # ---- weather and air, from the archive -------------------------------
    for kind in ("weather", "air"):
        have = read_cached(raw_path(kind, site))
        # the archive is only asked for hours it can actually have
        limit = min(end, era5_cutoff() + pd.Timedelta("23h"))
        gaps = missing_ranges(have, fetch_start, limit)
        new = [fetch_archive(kind, lat, lon, a, b) for a, b in gaps] \
            if gaps and limit > fetch_start else []
        merged = DLD.merge_frames(([have] if have is not None else []) + new)
        if merged is None:
            raise Skip(f"the {kind} archive returned nothing")
        save(merged, kind, site)
        notes.append(f"{kind} {'cached' if not gaps else 'fetched'}")

    # ---- the forecast tail, for the hours ERA5 has not reached -----------
    if mode in ("live", "penalty") or end > era5_cutoff() + pd.Timedelta("23h"):
        past = int(FETCH["penalty_past_days" if mode == "penalty"
                         else "forecast_past_days"])
        for kind, key in (("weather", "weather_fc"), ("air", "air_fc")):
            fc = fetch_forecast(kind, lat, lon, past)
            if fc is None:
                notes.append(f"{kind} forecast unavailable")
                continue
            have = read_cached(raw_path(key, site))
            # the forecast feed is re-issued constantly, so the NEW values
            # win rather than being averaged with yesterday's issue
            merged = fc if have is None else fc.combine_first(have)
            save(merged.sort_index(), key, site)
            notes.append(f"{kind} forecast {len(fc)} rows")

    return {"station": site, "seconds": round(time.perf_counter() - t0, 1),
            "note": " | ".join(notes)}


# ---------------------------------------------------------------------------
# Cleaning and features, reusing the project's own code
# ---------------------------------------------------------------------------
def frozen_drops():
    """
    The per-station variable drops step 2 decided on the training years. A
    deployed model must be fed the columns it was trained on.
    """
    path = project_path(CFG["paths"]["dropped_vars"])
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path, dtype={"site_no": str})
    return {s: sorted(g["variable"]) for s, g in d.groupby("site_no")}


def blend_weather(site, index, mode, prefer="era5"):
    """
    One station's weather and air tables on `index`, plus the per-hour
    source tag.

    prefer="era5"      ERA5 archive first, the forecast feed only where ERA5
                       has nothing. This is what a live run has: ERA5 for
                       everything older than about five days, the forecast
                       API for the tail.
    prefer="forecast"  the forecast feed wherever it reaches, ERA5 only as a
                       fallback. Used by the live-data penalty to score the
                       same hours a second time on forecast weather.

    Returns (weather, air, tags) where `tags` carries `weather_source` and
    `air_source` on the same index.
    """
    out, tags = {}, {}
    for kind, key, cols in (("weather", "weather_fc", WEATHER_VARS),
                            ("air", "air_fc", AIR_VARS)):
        arc = read_cached(raw_path(kind, site))
        fc = read_cached(raw_path(key, site))
        frames, names = [], []
        for table, tag in ((arc, TAGS["era5"]), (fc, TAGS["forecast"])):
            if table is None:
                continue
            frames.append(table.reindex(index).reindex(columns=cols))
            names.append(tag)
        if prefer == "forecast":
            frames, names = frames[::-1], names[::-1]
        if not frames:
            out[kind] = pd.DataFrame(index=index, columns=cols, dtype="float64")
            tags[f"{kind}_source"] = pd.Series(TAGS["missing"], index=index)
            continue
        best = frames[0].copy()
        src = pd.Series(np.where(best.notna().any(axis=1), names[0],
                                 TAGS["missing"]), index=index)
        for frame, tag in zip(frames[1:], names[1:]):
            use = best.isna().all(axis=1) & frame.notna().any(axis=1)
            best = best.where(~use, frame)
            src = src.where(~use, tag)
        out[kind] = best
        tags[f"{kind}_source"] = src
    return out["weather"], out["air"], pd.DataFrame(tags, index=index)


def clean_one(site, drops, index, mode, prefer="era5"):
    """
    One station's cleaned hourly table, through `clean.clean_station`. The
    weather and air tables are blended first and handed over as temporary
    csv files, because that is the interface clean_station has and the point
    of this module is to reuse it rather than to copy it.
    """
    import tempfile
    wx, aq, tags = blend_weather(site, index, mode, prefer)
    tmp = tempfile.mkdtemp(prefix=f"live_{site}_")
    try:
        wpath = os.path.join(tmp, "weather.csv")
        apath = os.path.join(tmp, "air.csv")
        wx.rename_axis("time").to_csv(wpath)
        aq.rename_axis("time").to_csv(apath)
        df, dropped, _, _ = CLEAN.clean_station(
            site, raw_path("water", site), wpath, apath,
            index=index, drop_variables=drops.get(site, []))
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    # A loud check, because the quiet version of this going wrong is a
    # 2026 table whose weather and air columns are silently all-NaN - the
    # models would be fed water only and every metric would still look
    # plausible. It happened once; it fails here now.
    if not df.index.equals(index):
        raise Skip(f"the cleaned table has {len(df)} rows for a "
                   f"{len(index)}-hour window - the water, weather and air "
                   f"indices did not line up")
    empty = [v for v in WEATHER_VARS + AIR_VARS if df[v].notna().sum() == 0]
    if len(empty) == len(WEATHER_VARS) + len(AIR_VARS):
        raise Skip("no weather or air data at all for this window")
    df.insert(0, "station", site)
    tags.insert(0, "station", site)
    return df.reset_index(), tags.rename_axis("time").reset_index()


def features_one(args):
    """
    One station's 211 engineered features, through the project's own
    `build_station`. Runs in a worker process, so it takes and returns plain
    objects.
    """
    site, panel, station_id, lat, lon, start, end = args
    g = panel.sort_values("time").set_index("time")
    feat = build_station(g, site, station_id, lat, lon)
    t = feat["time"]
    keep = (t >= start) & (t <= end) & feat[D.DO].notna()
    # the first hours of the fetched history have unfilled rolling windows
    keep &= t >= t.min() + pd.Timedelta(hours=WARMUP)
    feat = feat[keep].reset_index(drop=True)
    # the same dtypes data/processed uses, so the models see what they saw
    # in training and the file stays small
    float_cols = [c for c in feat.columns
                  if c not in ("station", "time")
                  and not c.startswith("y_cls_")]
    feat[float_cols] = feat[float_cols].astype("float32")
    for h in D.HORIZONS:
        feat[f"y_cls_{h}"] = feat[f"y_cls_{h}"].astype("Int8")
    feat["station_id"] = feat["station_id"].astype("int16")
    feat["month"] = feat["month"].astype("int8")
    return feat


def build_panel(mode, stations, win, workers=None, prefer="era5"):
    """
    Clean every station and build its features, in parallel across stations.
    Returns (hourly panel, features, source tags, skipped rows).
    """
    start, end, fetch_start, water_end = win
    index = pd.date_range(fetch_start, water_end, freq="h")
    drops = frozen_drops()
    codes = D.station_codes()
    meta = D.stations_table().set_index("site_no")

    hourly, tags, skipped = [], [], []
    for site in stations:
        try:
            df, tg = clean_one(site, drops, index, mode, prefer)
            hourly.append(df)
            tags.append(tg)
        except Exception as e:
            skipped.append({"station": site, "stage": "clean",
                            "reason": f"{type(e).__name__}: {e}"})
    if not hourly:
        return None, None, None, pd.DataFrame(skipped)

    panel = pd.concat(hourly, ignore_index=True)
    tagframe = pd.concat(tags, ignore_index=True)

    jobs = []
    for site, g in panel.groupby("station", sort=True):
        if site not in codes or site not in meta.index:
            skipped.append({"station": site, "stage": "features",
                            "reason": "station not in stations.csv"})
            continue
        jobs.append((site, g, codes[site], float(meta.loc[site, "lat"]),
                     float(meta.loc[site, "lon"]), start, end))

    n = workers or D.feature_workers()
    feats = []
    if n > 1 and len(jobs) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=min(n, len(jobs)))(
            delayed(features_one)(j) for j in jobs)
    else:
        results = [features_one(j) for j in jobs]
    for j, f in zip(jobs, results):
        if f is None or not len(f):
            skipped.append({"station": j[0], "stage": "features",
                            "reason": "no scoreable rows in the window"})
            continue
        feats.append(f)

    features = pd.concat(feats, ignore_index=True) if feats else None
    if features is not None:
        features = features.sort_values(["station", "time"]) \
                           .reset_index(drop=True)
    return panel, features, tagframe, pd.DataFrame(skipped)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(mode, stations=None, workers=None, now=None):
    """
    Fetch, clean and build features for one mode. Returns a dict with the
    paths written and the stations skipped. Never raises for a single
    station's failure.
    """
    st = D.Stages()
    win = window(mode, now)
    start, end, fetch_start, water_end = win
    table = D.stations_table()
    if stations:
        table = table[table["site_no"].astype(str).isin([str(s) for s in stations])]
    sites = table["site_no"].astype(str).tolist()

    D.log(f"\n=== fetch_live --mode {mode} ===")
    D.log(f"  window      {start:%Y-%m-%d %H:%M} .. {end:%Y-%m-%d %H:%M} UTC")
    D.log(f"  fetched     {fetch_start:%Y-%m-%d} .. {water_end:%Y-%m-%d} "
          f"(water tail for the 48 h targets)")
    D.log(f"  ERA5 cutoff {era5_cutoff(now):%Y-%m-%d} "
          f"(archive lag {D.TEST26['era5_lag_days']} days)")
    D.log(f"  {len(sites)} stations, {FETCH['max_workers']} at a time\n")

    ok, skipped, log_rows = [], [], []
    with st("fetch"):
        with ThreadPoolExecutor(max_workers=int(FETCH["max_workers"])) as pool:
            futures = {pool.submit(fetch_station, s,
                                   float(table.loc[table["site_no"] == s,
                                                   "lat"].iloc[0]),
                                   float(table.loc[table["site_no"] == s,
                                                   "lon"].iloc[0]),
                                   mode, win): s
                       for s in sites}
            for i, fut in enumerate(as_completed(futures), 1):
                site = futures[fut]
                try:
                    row = fut.result()
                    ok.append(site)
                    log_rows.append(row)
                    D.log(f"  [{i}/{len(sites)}] {site}  {row['seconds']:>5.1f}s  "
                          f"{row['note']}")
                except Skip as e:
                    skipped.append({"station": site, "stage": "fetch",
                                    "reason": str(e)})
                    D.log(f"  [{i}/{len(sites)}] {site}  SKIPPED: {e}")
                except Exception as e:
                    skipped.append({"station": site, "stage": "fetch",
                                    "reason": f"{type(e).__name__}: {e}"})
                    D.log(f"  [{i}/{len(sites)}] {site}  SKIPPED: "
                          f"{type(e).__name__}: {e}")

    if not ok:
        D.log("\nNo station returned usable data - nothing to build.")
        out = {"mode": mode, "stations": [], "skipped": skipped}
        _write_skipped(mode, pd.DataFrame(skipped))
        return out

    # In penalty mode the SAME hours are built twice: once on ERA5 weather
    # and once on the forecast feed, so the two can be scored against each
    # other hour for hour.
    variants = ([("era5", f"{mode}_era5"), ("forecast", f"{mode}_forecast")]
                if mode == "penalty" else [("era5", mode)])

    paths = {}
    features = tags = panel = None
    for prefer, label in variants:
        with st("clean+features"):
            panel, features, tags, skip2 = build_panel(mode, sorted(ok), win,
                                                       workers, prefer=prefer)
        if len(skip2):
            skipped += skip2.to_dict("records")
        with st("write"):
            if panel is not None:
                paths[f"hourly_{label}"] = D.dpath(DATA["hourly"], mode=label)
                panel.to_parquet(paths[f"hourly_{label}"], index=False)
            if tags is not None:
                paths[f"sources_{label}"] = D.dpath(DATA["sources"], mode=label)
                tags.to_parquet(paths[f"sources_{label}"], index=False)
            if features is not None:
                paths[f"features_{label}"] = D.dpath(DATA["features"],
                                                     mode=label)
                features.to_parquet(paths[f"features_{label}"], index=False)
    with st("write"):
        pd.DataFrame(log_rows).to_csv(D.dpath(DATA["fetch_log"], mode=mode),
                                      index=False)
        _write_skipped(mode, pd.DataFrame(skipped))

    used = sorted(set(features["station"])) if features is not None else []
    D.log(f"\n  stations with usable rows: {len(used)} of {len(sites)}")
    if features is not None:
        D.log(f"  {len(features):,} feature rows "
              f"{features['time'].min():%Y-%m-%d} .. "
              f"{features['time'].max():%Y-%m-%d}")
    if tags is not None:
        share = (tags[tags["time"].between(start, end)]["weather_source"]
                 .value_counts(normalize=True))
        D.log("  weather source: "
              + ", ".join(f"{k} {v:.1%}" for k, v in share.items()))
    if skipped:
        D.log(f"  skipped {len(skipped)}:")
        for s in skipped:
            D.log(f"    {s['station']}  ({s['stage']}) {s['reason']}")
    for k, v in paths.items():
        D.log(f"  wrote {D.shortpath(v)}")
    st.report()
    return {"mode": mode, "stations": used, "skipped": skipped,
            "paths": paths, "window": (start, end), "timings": st.times}


def _write_skipped(mode, df):
    path = D.dpath(DATA["skipped"], mode=mode)
    (df if len(df) else pd.DataFrame(columns=["station", "stage", "reason"])
     ).to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--mode", choices=["history", "live", "penalty"],
                    default="history")
    ap.add_argument("--stations", nargs="*", default=None,
                    help="only these stations (default: all 30)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    run(args.mode, stations=args.stations, workers=args.workers)


if __name__ == "__main__":
    main()
