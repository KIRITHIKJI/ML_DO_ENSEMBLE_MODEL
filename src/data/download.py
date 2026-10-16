"""
STEP 1 - Get the raw data for 2023-01-01 .. 2025-12-31.

Run from the project root:

    python -m src.data.download

What it does, in order:

  0. IMPORT: copies the data already downloaded by the old project
     (A:/ML_MAIN, read-only) into data/raw/ and merges the 2023-2024 files
     with the 2025 files per station, de-duplicating timestamps.
  1. TOP UP: downloads whatever part of the period is still missing for those
     stations (mainly Oct-Dec 2025) and merges it in.
  2. GROW: searches USGS for more stations and adds up to `target_stations`
     in total, downloading the full 2023-2025 period for each new one.
  3. Writes data/raw/stations.csv.

Three sources, all free and with no API key:
  * water   - USGS instantaneous values  (15-min, UTC)
  * weather - ERA5 via Open-Meteo        (hourly, UTC)
  * air     - CAMS via Open-Meteo        (hourly, UTC)

Safe to re-run: anything already on disk and already covering the full period
is skipped. Network errors are retried; a station that keeps failing is
skipped and the run continues.
"""

import io
import os
import shutil
import time

import pandas as pd
import requests

from src.config import load_config, project_path, set_seeds

# ---------------------------------------------------------------------------
# Settings, all from config.yaml
# ---------------------------------------------------------------------------
CFG = load_config()
DL = CFG["download"]
P = CFG["paths"]

START = CFG["period"]["start"]
END = CFG["period"]["end"]
YEARS = [2023, 2024, 2025]

PARAMS = {str(k): v for k, v in DL["usgs_params"].items()}   # code -> column
WATER_COLS = list(PARAMS.values())                           # canonical order
REQUIRED = [str(c) for c in DL["required_params"]]
DO_COL = CFG["target"]["column"]
LOW = CFG["target"]["thresholds"]["low"]

RAW = project_path(P["data_raw"])
DIRS = {"water": project_path(P["water"]),
        "weather": project_path(P["weather"]),
        "air": project_path(P["air"])}
STATIONS_CSV = project_path(P["stations"])
# Candidate stations we screen but may not keep live under data/interim, so
# data/raw only ever holds stations we actually use.
SCREEN_DIR = project_path(P["data_interim"], "candidates")
REJECT_CSV = project_path(P["data_interim"], "rejected_candidates.csv")

session = requests.Session()
session.headers["User-Agent"] = DL["user_agent"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def get(url, params, tries=None, timeout=None):
    """GET with retries. Returns the response, or None if it never worked."""
    tries = tries or DL["retries"]
    timeout = timeout or DL["timeout"]
    for k in range(tries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            # 400/404 mean "no such data" - retrying will not help.
            if r.status_code in (400, 404):
                return r
            print(f"      HTTP {r.status_code} (try {k + 1}/{tries})")
        except requests.RequestException as e:
            print(f"      network error: {e} (try {k + 1}/{tries})")
        time.sleep(3 * (k + 1))
    print(f"      giving up on {url}")
    return None


def read_csv_time(path):
    """Read one of our raw csv files, with `time` parsed and as the index."""
    df = pd.read_csv(path, parse_dates=["time"])
    return df.set_index("time").sort_index()


def merge_frames(frames):
    """
    Stack several tables of the same kind and de-duplicate timestamps.
    Where two sources give the same timestamp we average them (identical
    values stay identical; a NaN in one source is ignored).
    """
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return None
    out = pd.concat(frames)
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.groupby(level=0).mean().sort_index()
    out.index.name = "time"
    return out


def save_table(df, kind, site, column_order=None):
    """Write one station's table to data/raw/<kind>/<site>.csv."""
    if column_order:
        # keep a stable column order; add any missing column as empty
        for c in column_order:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[column_order]
    path = os.path.join(DIRS[kind], f"{site}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    return path


def covered_range(path):
    """First and last timestamp in an existing file, or (None, None)."""
    if not os.path.exists(path):
        return None, None
    try:
        df = read_csv_time(path)
    except Exception:
        return None, None
    if not len(df):
        return None, None
    return df.index.min(), df.index.max()


def do_stats(water):
    """
    Per-year DO statistics used to judge a station.
    DO is averaged into hourly means first, so stations with different sensor
    intervals are compared fairly.
    """
    if water is None or DO_COL not in water.columns:
        return None
    do = water[DO_COL].resample("h").mean()
    stats = {}
    for y in YEARS:
        full = pd.date_range(f"{y}-01-01", f"{y}-12-31 23:00", freq="h")
        s = do.reindex(full)
        stats[y] = {"coverage": round(float(s.notna().mean()), 3),
                    "hours_below_low": int((s < LOW).sum())}
    return stats


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------
def fetch_water(site, start, end):
    """
    USGS instantaneous values for one station, one date range.
    Returns a wide table (time index, one column per parameter) or None.
    Long ranges are requested one calendar year at a time, which is what the
    service copes with best.
    """
    frames = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        y_start = max(pd.Timestamp(start), pd.Timestamp(f"{year}-01-01"))
        y_end = min(pd.Timestamp(end), pd.Timestamp(f"{year}-12-31"))
        if y_start > y_end:
            continue
        r = get(DL["urls"]["usgs_iv"], {
            "format": "json", "sites": site,
            "parameterCd": ",".join(PARAMS),
            "startDT": y_start.strftime("%Y-%m-%d"),
            "endDT": y_end.strftime("%Y-%m-%d"),
            "siteStatus": "all"})
        if r is None or r.status_code != 200:
            continue
        try:
            series = r.json()["value"]["timeSeries"]
        except Exception:
            continue
        for ts in series:
            code = ts["variable"]["variableCode"][0]["value"]
            if code not in PARAMS:
                continue
            nodata = ts["variable"].get("noDataValue", -999999)
            for block in ts["values"]:
                if not block["value"]:
                    continue
                d = pd.DataFrame(block["value"])[["dateTime", "value"]]
                d["value"] = pd.to_numeric(d["value"], errors="coerce")
                d.loc[d["value"] == nodata, "value"] = None
                d["time"] = pd.to_datetime(d["dateTime"], utc=True, format="mixed")
                d["param"] = PARAMS[code]
                frames.append(d[["time", "param", "value"]])
    if not frames:
        return None
    long = pd.concat(frames)
    # one parameter can have several sensors at a site -> average them
    wide = (long.groupby(["time", "param"])["value"].mean()
                .unstack("param").sort_index())
    wide.index = wide.index.tz_convert("UTC").tz_localize(None)
    wide.index.name = "time"
    return wide


def fetch_open_meteo(kind, lat, lon, start, end):
    """ERA5 weather or CAMS air quality for one point and date range."""
    url = DL["urls"]["open_meteo_weather" if kind == "weather" else "open_meteo_air"]
    vars_ = DL["weather_vars" if kind == "weather" else "air_vars"]
    r = get(url, {"latitude": lat, "longitude": lon,
                  "start_date": pd.Timestamp(start).strftime("%Y-%m-%d"),
                  "end_date": pd.Timestamp(end).strftime("%Y-%m-%d"),
                  "hourly": ",".join(vars_), "timezone": "GMT"})
    if r is None or r.status_code != 200:
        print(f"      {kind} download failed")
        return None
    try:
        df = pd.DataFrame(r.json()["hourly"])
    except Exception:
        return None
    df["time"] = pd.to_datetime(df["time"])
    time.sleep(DL["sleep_between_calls"])        # be polite to the free API
    return df.set_index("time").sort_index()


# ---------------------------------------------------------------------------
# Step 0 - import what the old project already downloaded
# ---------------------------------------------------------------------------
def import_legacy():
    """
    Copy (never move) the old project's files into data/raw, merging the
    2023-2024 and the 2025 download per station. Returns the old stations
    table, or None if the old project is not there.
    """
    old_a = CFG["legacy"]["raw_2023_2024"]
    old_b = CFG["legacy"]["raw_2025"]
    src_stations = os.path.join(old_a, "stations.csv")
    if not os.path.exists(src_stations):
        print(f"No old project at {old_a} - nothing to import.")
        return None

    stations = pd.read_csv(src_stations, dtype={"site_no": str})
    print(f"\n=== Step 0: importing {len(stations)} stations from the old project ===")
    os.makedirs(RAW, exist_ok=True)
    # keep a copy of the screening list the old project produced
    for extra in ["station_candidates.csv"]:
        s = os.path.join(old_a, extra)
        if os.path.exists(s) and not os.path.exists(os.path.join(RAW, extra)):
            shutil.copy(s, os.path.join(RAW, extra))

    for i, s in stations.iterrows():
        site = str(s["site_no"])
        done = []
        for kind in ["water", "weather", "air"]:
            target = os.path.join(DIRS[kind], f"{site}.csv")
            if os.path.exists(target):
                done.append(f"{kind}=already here")
                continue
            parts = []
            for folder in [old_a, old_b]:
                p = os.path.join(folder, kind, f"{site}.csv")
                if os.path.exists(p):
                    parts.append(read_csv_time(p))
            if not parts:
                done.append(f"{kind}=MISSING")
                continue
            merged = merge_frames(parts)
            save_table(merged, kind, site,
                       WATER_COLS if kind == "water" else None)
            done.append(f"{kind}={len(merged)} rows "
                        f"{merged.index.min():%Y-%m-%d}..{merged.index.max():%Y-%m-%d}")
        print(f"  [{i + 1}/{len(stations)}] {site}  " + " | ".join(done))
    return stations


# ---------------------------------------------------------------------------
# Step 1 - top up the imported stations to the full period
# ---------------------------------------------------------------------------
def topup_station(site, lat, lon):
    """
    Download whatever is missing at the end (and start) of each of the three
    files for one station, and merge it in. Nothing is re-downloaded if the
    file already reaches the end of the period.
    """
    period_start, period_end = pd.Timestamp(START), pd.Timestamp(END)
    for kind in ["water", "weather", "air"]:
        path = os.path.join(DIRS[kind], f"{site}.csv")
        have_lo, have_hi = covered_range(path)
        existing = read_csv_time(path) if have_lo is not None else None

        gaps = []
        if have_lo is None:
            gaps.append((period_start, period_end))
        else:
            # more than a day short at either end counts as a gap worth filling
            if have_lo > period_start + pd.Timedelta("1D"):
                gaps.append((period_start, have_lo))
            if have_hi < period_end - pd.Timedelta("1D"):
                gaps.append((have_hi, period_end))
        if not gaps:
            print(f"      {kind}: complete, skipped")
            continue

        new = []
        for g0, g1 in gaps:
            print(f"      {kind}: fetching {g0:%Y-%m-%d}..{g1:%Y-%m-%d}")
            if kind == "water":
                new.append(fetch_water(site, g0, g1))
            else:
                new.append(fetch_open_meteo(kind, lat, lon, g0, g1))
        merged = merge_frames([existing] + new)
        if merged is None:
            print(f"      {kind}: nothing downloaded, left as it was")
            continue
        merged = merged[(merged.index >= period_start) &
                        (merged.index <= period_end + pd.Timedelta("1D"))]
        save_table(merged, kind, site, WATER_COLS if kind == "water" else None)
        print(f"      {kind}: now {len(merged)} rows "
              f"{merged.index.min():%Y-%m-%d}..{merged.index.max():%Y-%m-%d}")


# ---------------------------------------------------------------------------
# Step 2 - find more stations
# ---------------------------------------------------------------------------
def find_candidates(exclude):
    """
    Ask USGS, state by state, for river stations whose DO / temperature /
    conductivity sensors ran for the whole 2023-2025 period. This is a cheap
    metadata query - no measurements are downloaded yet.
    """
    path = project_path(P["data_interim"], "new_candidates.csv")
    if os.path.exists(path):
        print(f"  using the candidate list already saved at {path}")
        return pd.read_csv(path, dtype={"site_no": str})

    rows = []
    for st in DL["states"]:
        r = get(DL["urls"]["usgs_site"], {
            "format": "rdb", "stateCd": st, "siteType": "ST",
            "parameterCd": ",".join(PARAMS), "hasDataTypeCd": "iv",
            "seriesCatalogOutput": "true", "siteStatus": "all"})
        if r is None or r.status_code != 200:
            print(f"  {st}: lookup failed, skipping this state")
            continue
        lines = [l for l in r.text.splitlines() if not l.startswith("#")]
        if len(lines) < 3:
            print(f"  {st}: 0 candidates")
            continue
        # the rdb format has a header line, then a line of column widths
        df = pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[2:])),
                         sep="\t", dtype=str)
        df = df[df["data_type_cd"].isin(["uv", "iv"]) & df["parm_cd"].isin(PARAMS)]
        df["begin_date"] = pd.to_datetime(df["begin_date"], errors="coerce")
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
        # the sensor must have been running before the period starts and
        # still running at the end of it
        df = df[(df["begin_date"] <= START) & (df["end_date"] >= END)]
        found = 0
        for site, g in df.groupby("site_no"):
            if site in exclude:
                continue
            have = set(g["parm_cd"])
            if not all(p in have for p in REQUIRED):
                continue
            rows.append({"site_no": site, "state": st,
                         "name": g["station_nm"].iloc[0],
                         "lat": float(g["dec_lat_va"].iloc[0]),
                         "lon": float(g["dec_long_va"].iloc[0]),
                         "n_params": len(have & set(PARAMS)),
                         "params": ",".join(sorted(have & set(PARAMS)))})
            found += 1
        print(f"  {st}: {found} candidates")

    if not rows:
        return pd.DataFrame(columns=["site_no", "state", "name", "lat", "lon",
                                     "n_params", "params"])
    cand = pd.DataFrame(rows)
    # Keep the screening list spread across states: at most
    # `candidates_per_state` per state, then take one state at a time in
    # rotation (FL, GA, LA, ... , FL, GA, ...). Within a state the stations
    # with the most parameters come first. That way we never spend the whole
    # budget on the first couple of states.
    cand["order"] = cand["state"].map({s: i for i, s in enumerate(DL["states"])})
    cand = cand.sort_values(["order", "n_params"], ascending=[True, False])
    cand = cand.groupby("state", sort=False).head(DL["candidates_per_state"])
    cand["rank_in_state"] = cand.groupby("state").cumcount()
    cand = (cand.sort_values(["rank_in_state", "order"])
                .drop(columns=["order", "rank_in_state"])
                .head(DL["max_new_candidates"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cand.to_csv(path, index=False)
    return cand


def screen_candidate(site):
    """
    Download a candidate's water data (into data/interim, not data/raw) and
    return it. Cached, so re-runs are free.
    """
    path = os.path.join(SCREEN_DIR, "water", f"{site}.csv")
    if os.path.exists(path):
        return read_csv_time(path)
    water = fetch_water(site, START, END)
    if water is None:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    water.to_csv(path)
    return water


def passes_rules(water):
    """
    The station rules from the project brief:
      * has DO, water temperature and conductivity
      * DO coverage >= min_do_coverage in EACH of 2023, 2024, 2025
      * some hours of DO below the Low threshold in at least two of the years
    Returns (ok, reason, stats).
    """
    needed = [PARAMS[c] for c in REQUIRED]
    missing = [c for c in needed
               if c not in water.columns or water[c].notna().sum() == 0]
    if missing:
        return False, f"no {', '.join(missing)}", None
    stats = do_stats(water)
    thin = [y for y in YEARS if stats[y]["coverage"] < DL["min_do_coverage"]]
    if thin:
        cov = ", ".join(f"{y}:{stats[y]['coverage']:.0%}" for y in thin)
        return False, f"DO coverage too low ({cov})", stats
    years_with_low = sum(stats[y]["hours_below_low"] > 0 for y in YEARS)
    if years_with_low < 2:
        return False, f"DO below {LOW} in only {years_with_low} of 3 years", stats
    return True, "ok", stats


def summarise(row, stats):
    """One row of stations.csv."""
    out = {"site_no": str(row["site_no"]), "state": row["state"],
           "name": row["name"], "lat": row["lat"], "lon": row["lon"],
           "n_params": row.get("n_params"), "params": row.get("params")}
    total_low = 0
    for y in YEARS:
        out[f"do_coverage_{y}"] = stats[y]["coverage"]
        out[f"hours_do_below_5_{y}"] = stats[y]["hours_below_low"]
        total_low += stats[y]["hours_below_low"]
    out["hours_do_below_5_total"] = total_low
    out["do_coverage_min"] = min(stats[y]["coverage"] for y in YEARS)
    return out


def grow_stations(kept, existing_sites):
    """
    Screen new candidates until we have `target_stations` in total.
    Accepted stations get their water file copied into data/raw and their
    weather + air downloaded.
    """
    target = DL["target_stations"]
    need = target - len(kept)
    rejected = []
    if need <= 0:
        print(f"\n=== Step 2: already have {len(kept)} stations, none needed ===")
        return kept, [], rejected

    print(f"\n=== Step 2: looking for {need} more stations "
          f"(target {target} in total) ===")
    cand = find_candidates(exclude=set(existing_sites))
    if not len(cand):
        print("  no candidates found")
        return kept, [], rejected
    print(f"  {len(cand)} candidates to screen, best first\n")

    # how many NEW stations we allow per state, so the set stays spread out
    new_per_state = {}
    screened = []
    for i, (_, row) in enumerate(cand.iterrows(), 1):
        # Stop once we have a few more passing stations than we need: enough
        # choice to prefer the ones with the most low-oxygen hours, without
        # downloading three years of data for stations we will never use.
        if len(screened) >= need + 5:
            break
        site = str(row["site_no"])
        st = row["state"]
        if new_per_state.get(st, 0) >= DL["per_state_cap"]:
            continue
        print(f"  [{i}/{len(cand)}] screening {site} ({st}) {str(row['name'])[:50]}")
        try:
            water = screen_candidate(site)
        except Exception as e:
            print(f"      error: {e}, skipped")
            rejected.append({"site_no": site, "state": st, "reason": f"error: {e}"})
            continue
        if water is None:
            print("      no water data, skipped")
            rejected.append({"site_no": site, "state": st, "reason": "no water data"})
            continue
        ok, reason, stats = passes_rules(water)
        if not ok:
            print(f"      rejected: {reason}")
            rejected.append({"site_no": site, "state": st, "reason": reason})
            continue
        low_txt = "/".join(str(stats[y]["hours_below_low"]) for y in YEARS)
        print(f"      PASSES - hours DO<{LOW} per year (2023/24/25): {low_txt}")
        screened.append((row, stats))
        new_per_state[st] = new_per_state.get(st, 0) + 1

    # Prefer the stations with the most low-oxygen hours - those are the
    # events the early-warning model has to catch.
    screened.sort(key=lambda t: -sum(t[1][y]["hours_below_low"] for y in YEARS))

    added = []
    for row, stats in screened:
        if len(kept) >= target:
            break
        site = str(row["site_no"])
        print(f"\n  adding {site} ({row['state']}) - weather + air")
        w = os.path.join(DIRS["weather"], f"{site}.csv")
        a = os.path.join(DIRS["air"], f"{site}.csv")
        wx = (read_csv_time(w) if os.path.exists(w) else
              fetch_open_meteo("weather", row["lat"], row["lon"], START, END))
        aq = (read_csv_time(a) if os.path.exists(a) else
              fetch_open_meteo("air", row["lat"], row["lon"], START, END))
        if wx is None or aq is None:
            print("      weather/air failed, skipping this station")
            rejected.append({"site_no": site, "state": row["state"],
                             "reason": "weather/air download failed"})
            continue
        save_table(wx, "weather", site)
        save_table(aq, "air", site)
        shutil.copy(os.path.join(SCREEN_DIR, "water", f"{site}.csv"),
                    os.path.join(DIRS["water"], f"{site}.csv"))
        rec = summarise(row, stats)
        kept.append(rec)
        added.append(rec)
        print(f"      kept ({len(kept)}/{target})")

    return kept, added, rejected


# ---------------------------------------------------------------------------
# Step 3 - keep data/raw in step with stations.csv
# ---------------------------------------------------------------------------
def tidy_raw(keep_sites):
    """
    Move the files of stations we are not using out of data/raw and into
    data/interim/unused/. Nothing is deleted - a station dropped today can be
    brought back by hand - but every file left in data/raw belongs to a
    station in stations.csv, so later steps can simply read the folder.
    """
    moved = []
    for kind in ["water", "weather", "air"]:
        folder = DIRS[kind]
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".csv"):
                continue
            site = fname[:-4]
            if site in keep_sites:
                continue
            dest_dir = project_path(P["data_interim"], "unused", kind)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, fname)
            if os.path.exists(dest):
                os.remove(dest)
            shutil.move(os.path.join(folder, fname), dest)
            moved.append(f"{kind}/{site}")
    if moved:
        sites = sorted({m.split("/")[1] for m in moved})
        print(f"\nMoved {len(sites)} unused stations out of data/raw "
              f"(now in data/interim/unused/): {', '.join(sites)}")
    return moved


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    set_seeds(CFG["seed"])
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)
    print(f"Period: {START} .. {END}   target stations: {DL['target_stations']}")

    # Step 0 + 1: the stations we inherit from the old project
    old = import_legacy()
    kept, dropped = [], []
    if old is not None:
        print(f"\n=== Step 1: topping up {len(old)} imported stations to {END} ===")
        for i, s in old.iterrows():
            site = str(s["site_no"])
            print(f"  [{i + 1}/{len(old)}] {site} {str(s['name'])[:50]}")
            try:
                topup_station(site, s["lat"], s["lon"])
            except Exception as e:
                print(f"      ERROR topping up {site}: {e} - keeping what we have")
            wpath = os.path.join(DIRS["water"], f"{site}.csv")
            if not os.path.exists(wpath):
                print("      no water file, dropped")
                dropped.append({"site_no": site, "state": s["state"],
                                "reason": "no water file"})
                continue
            water = read_csv_time(wpath)
            # The same station rules apply to inherited stations: a station
            # whose DO sensor stopped part-way cannot be part of an honest
            # 2025 test, so it is dropped here and replaced in step 2.
            if DL.get("apply_rules_to_legacy", True):
                ok, reason, stats = passes_rules(water)
                if not ok:
                    print(f"      dropped: {reason}")
                    dropped.append({"site_no": site, "state": s["state"],
                                    "reason": f"inherited but {reason}"})
                    continue
            else:
                stats = do_stats(water)
            if stats is None:
                print("      no DO column, dropped")
                dropped.append({"site_no": site, "state": s["state"],
                                "reason": "no DO column"})
                continue
            kept.append(summarise(s, stats))

    # Step 2: add new stations up to the target. Stations we already have and
    # stations we just dropped are both excluded from the search.
    existing = ({r["site_no"] for r in kept}
                | {r["site_no"] for r in dropped})
    kept, added, rejected = grow_stations(kept, existing)

    # Step 3: stations.csv, plus a note of everything we skipped
    df = pd.DataFrame(kept).sort_values("hours_do_below_5_total", ascending=False)
    df.to_csv(STATIONS_CSV, index=False)
    tidy_raw(set(df["site_no"]))
    skipped = dropped + rejected
    if skipped:
        os.makedirs(os.path.dirname(REJECT_CSV), exist_ok=True)
        pd.DataFrame(skipped).to_csv(REJECT_CSV, index=False)

    print(f"\n=== Done: {len(df)} stations -> {STATIONS_CSV} ===")
    cols = ["site_no", "state", "do_coverage_min", "hours_do_below_5_total"]
    print(df[cols].to_string(index=False))
    print(f"\nNewly added this run: {len(added)}   skipped: {len(skipped)}"
          + (f" (see {REJECT_CSV})" if skipped else ""))
    print("Next: python -m src.data.check_data")


if __name__ == "__main__":
    main()
