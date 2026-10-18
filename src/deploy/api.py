"""
STEP 12.7 - the API.

    python -m src.deploy.api                 # or: uvicorn src.deploy.api:app

Three read-only endpoints over the latest forecast file:

    GET /health                 is the service up, and how fresh is its data
    GET /warnings               the stations under warning now
    GET /forecast/{station}     the latest forecast for one station,
                                all horizons

It reads `results/live/forecast_<timestamp>.csv` - whatever
`src/deploy/run.py` wrote last - and nothing else. It never trains, never
fetches and never writes. If no forecast exists yet, /health says so and
the other two return 503 rather than inventing an answer.

Two rules about what comes out of /warnings:

  * **Stale stations are excluded.** A station whose sensor stopped
    reporting has no prediction at all (see `run.max_data_age_hours`), so
    there is nothing for it to be warned about. /health lists them by name
    and /forecast/{station} returns the outage, so they are visible - just
    not as warnings.
  * **Early warnings are separated from ongoing lows.** `alert_type` says
    which: `early_warning` is safe water predicted to go Low, which is the
    thing this system is for; `ongoing_low` is water already below the
    threshold, which an operator can read off the sensor. Both are returned,
    counted separately, and `?alert_type=early_warning` narrows to the
    first.
"""

import os

import numpy as np
import pandas as pd

try:
    from fastapi import FastAPI, HTTPException
except ImportError as e:                      # pragma: no cover
    raise SystemExit(
        "fastapi is not installed. It is pinned in requirements.txt:\n"
        "    pip install -r requirements.txt") from e

from src.config import project_path

from src.deploy import common as D
from src.deploy import run as RUNNER

API = D.DEP["api"]
HORIZONS = D.HORIZONS
ALERT = D.RUN["alert_types"]


def _stale(df):
    """The stale rows: no prediction was made for these."""
    if "alert_type" in df.columns:
        return df[df["alert_type"].astype(str) == ALERT["stale"]]
    # a forecast file written before the stale guard existed
    return df.iloc[0:0]


def _live(df):
    """The rows a prediction was actually made for."""
    return df.drop(_stale(df).index)


def _stale_stations(df):
    return sorted(set(_stale(df)["station"].astype(str)))


def _num(value):
    """A float for JSON, or null when there is no number to report."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


app = FastAPI(title=API["title"],
              description=__doc__.strip().splitlines()[0],
              version=str(D.CFG["project"].get("name", "")) or "1")


# ---------------------------------------------------------------------------
# Reading the latest forecast
# ---------------------------------------------------------------------------
def latest():
    """(DataFrame, path) for the newest forecast file, or (None, None)."""
    return RUNNER.load_latest()


def _rows(df):
    """
    The forecast rows as plain JSON-able dicts, in the schema order.

    NaN becomes null. A stale row has no predicted DO, no P(Low) and no tau,
    and NaN is not valid JSON - left alone it makes the whole response a
    500, which is the one way an outage must not be reported.
    """
    cols = [c for c in RUNNER.COLUMNS if c in df.columns]
    extra = [c for c in ("observed_at", "do_now_mgl", "method")
             if c in df.columns]
    out = []
    for r in df[cols + extra].to_dict("records"):
        row = {}
        for k, v in r.items():
            if isinstance(v, pd.Timestamp):
                row[k] = v.isoformat()
            elif v is None or (isinstance(v, float) and not np.isfinite(v)):
                row[k] = None
            elif hasattr(v, "item"):
                item = v.item()
                row[k] = (None if isinstance(item, float)
                          and not np.isfinite(item) else item)
            else:
                row[k] = v
        out.append(row)
    return out


def _need_forecast():
    df, path = latest()
    if df is None:
        raise HTTPException(
            status_code=503,
            detail="no forecast has been produced yet - run "
                   "`python -m src.deploy.run` (or schedule run_live.bat)")
    return df, path


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """
    Up, and how fresh. `age_minutes` is how long ago the latest forecast was
    issued; a scheduler running hourly should keep it under 60.
    """
    df, path = latest()
    now = pd.Timestamp.utcnow().tz_localize(None)
    if df is None:
        return {"status": "no_forecast", "now": now.isoformat(),
                "model_version": D.model_version(),
                "models": D.DEP["models"]["stage"],
                "horizons": HORIZONS,
                "detail": "no forecast file in "
                          f"{D.RUN['results_dir']} yet"}
    issued = pd.Timestamp(df["issued_at"].max())
    live, stale = _live(df), _stale(df)
    warned = live[live["warning"].astype(bool)]

    def n_stations(d):
        return int(d["station"].nunique()) if len(d) else 0

    def by_alert(name):
        if "alert_type" not in df.columns:
            return None
        return n_stations(df[df["alert_type"].astype(str) == name])

    return {
        "status": "ok",
        "now": now.isoformat(),
        "latest_forecast": os.path.basename(path),
        "issued_at": issued.isoformat(),
        "age_minutes": round((now - issued).total_seconds() / 60.0, 1),
        "stations": int(df["station"].nunique()),
        "rows": int(len(df)),
        "horizons": sorted(int(h) for h in df["horizon"].unique()),
        "model_version": sorted(df["model_version"].astype(str).unique()),
        "models": D.DEP["models"]["stage"],
        # stale stations have no prediction, so they are not counted among
        # the warned ones - and they are named, because a silent sensor is
        # the one thing an operator must not have to go looking for.
        "stations_forecast": n_stations(live),
        "stations_warned": n_stations(warned),
        "stations_early_warning": by_alert(ALERT["early_warning"]),
        "stations_ongoing_low": by_alert(ALERT["ongoing_low"]),
        "stations_stale": _stale_stations(df),
        "max_data_age_hours_allowed": float(D.RUN["max_data_age_hours"]),
        "max_data_age_hours": _num(round(float(
            df["data_age_hours"].max()), 1)),
        "max_data_age_hours_forecast": (
            _num(round(float(live["data_age_hours"].max()), 1)) if len(live)
            else None),
    }


@app.get("/warnings")
def warnings(horizon: int = None, alert_type: str = None):
    """
    Every station under warning in the latest forecast, worst first.

    Stale stations are **excluded**: no prediction was made for them, so
    there is nothing to warn about. They are listed by /health instead.

    `?horizon=24` narrows it to one horizon; `?alert_type=early_warning`
    narrows it to safe water predicted to go Low, dropping the stations that
    are already below the threshold.
    """
    df, path = _need_forecast()
    d = _live(df)
    d = d[d["warning"].astype(bool)]
    if horizon is not None:
        if int(horizon) not in list(HORIZONS):
            raise HTTPException(status_code=400,
                                detail=f"horizon must be one of {HORIZONS}")
        d = d[d["horizon"] == int(horizon)]
    if alert_type is not None:
        allowed = [ALERT["early_warning"], ALERT["ongoing_low"]]
        if alert_type not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"alert_type must be one of {allowed} - stale "
                       f"stations are never warnings and are listed by "
                       f"/health")
        d = d[d["alert_type"].astype(str) == alert_type]
    d = d.sort_values(["p_low", "station"], ascending=[False, True])

    def n_of(name):
        if "alert_type" not in d.columns:
            return None
        return int(d[d["alert_type"].astype(str) == name]["station"].nunique())

    return {
        "issued_at": pd.Timestamp(df["issued_at"].max()).isoformat(),
        "latest_forecast": os.path.basename(path),
        "horizon": horizon,
        "alert_type": alert_type,
        "n_stations": int(d["station"].nunique()),
        "n_rows": int(len(d)),
        "n_stations_early_warning": n_of(ALERT["early_warning"]),
        "n_stations_ongoing_low": n_of(ALERT["ongoing_low"]),
        "excluded_stale_stations": _stale_stations(df),
        "warnings": _rows(d),
    }


@app.get("/forecast/{station}")
def forecast(station: str):
    """The latest forecast for one station, all horizons."""
    df, path = _need_forecast()
    d = df[df["station"].astype(str) == str(station)]
    if not len(d):
        known = sorted(df["station"].astype(str).unique())
        raise HTTPException(
            status_code=404,
            detail=f"station {station} is not in the latest forecast. "
                   f"Stations present: {', '.join(known)}")
    d = d.sort_values("horizon")
    first = d.iloc[0]
    stale = bool(len(_stale(d)) == len(d)) and len(d) > 0
    alerts = (sorted(set(d["alert_type"].astype(str)))
              if "alert_type" in d.columns else None)
    return {
        "station": str(station),
        "issued_at": pd.Timestamp(first["issued_at"]).isoformat(),
        "latest_forecast": os.path.basename(path),
        "observed_at": (pd.Timestamp(first["observed_at"]).isoformat()
                        if "observed_at" in d.columns else None),
        "do_now_mgl": (_num(first["do_now_mgl"])
                       if "do_now_mgl" in d.columns else None),
        "data_age_hours": _num(first["data_age_hours"]),
        "weather_source": str(first["weather_source"]),
        "model_version": str(first["model_version"]),
        # the outage, stated rather than implied by a row full of nulls
        "status": (str(first["status"]) if "status" in d.columns else None),
        "stale": stale,
        "max_data_age_hours_allowed": float(D.RUN["max_data_age_hours"]),
        "alert_types": alerts,
        "any_warning": bool(d["warning"].astype(bool).any()),
        "any_early_warning": (
            bool((d["alert_type"].astype(str) == ALERT["early_warning"]).any())
            if "alert_type" in d.columns else None),
        "detail": (f"no forecast was issued: the station's last DO reading "
                   f"is {_num(first['data_age_hours']):.1f} h old, over the "
                   f"{float(D.RUN['max_data_age_hours']):g} h limit"
                   if stale else None),
        "horizons": _rows(d),
    }


def main():
    import uvicorn
    D.log(f"serving {API['title']} on http://{API['host']}:{API['port']}")
    D.log(f"  endpoints: {', '.join(API['endpoints'])}")
    D.log(f"  reading {D.RUN['results_dir']}/{D.RUN['forecast_glob']}")
    uvicorn.run(app, host=API["host"], port=int(API["port"]))


if __name__ == "__main__":
    main()
