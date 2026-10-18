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
"""

import os

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
    """The forecast rows as plain JSON-able dicts, in the schema order."""
    cols = [c for c in RUNNER.COLUMNS if c in df.columns]
    extra = [c for c in ("observed_at", "do_now_mgl", "method")
             if c in df.columns]
    out = []
    for r in df[cols + extra].to_dict("records"):
        row = {}
        for k, v in r.items():
            if isinstance(v, pd.Timestamp):
                row[k] = v.isoformat()
            elif hasattr(v, "item"):
                row[k] = v.item()
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
        "stations_warned": int(df[df["warning"].astype(bool)]["station"]
                               .nunique()),
        "max_data_age_hours": round(float(df["data_age_hours"].max()), 1),
    }


@app.get("/warnings")
def warnings(horizon: int = None):
    """
    Every station under warning in the latest forecast, worst first.
    `?horizon=24` narrows it to one horizon.
    """
    df, path = _need_forecast()
    d = df[df["warning"].astype(bool)]
    if horizon is not None:
        if int(horizon) not in list(HORIZONS):
            raise HTTPException(status_code=400,
                                detail=f"horizon must be one of {HORIZONS}")
        d = d[d["horizon"] == int(horizon)]
    d = d.sort_values(["p_low", "station"], ascending=[False, True])
    return {
        "issued_at": pd.Timestamp(df["issued_at"].max()).isoformat(),
        "latest_forecast": os.path.basename(path),
        "horizon": horizon,
        "n_stations": int(d["station"].nunique()),
        "n_rows": int(len(d)),
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
    return {
        "station": str(station),
        "issued_at": pd.Timestamp(first["issued_at"]).isoformat(),
        "latest_forecast": os.path.basename(path),
        "observed_at": (pd.Timestamp(first["observed_at"]).isoformat()
                        if "observed_at" in d.columns else None),
        "do_now_mgl": (float(first["do_now_mgl"])
                       if "do_now_mgl" in d.columns else None),
        "data_age_hours": float(first["data_age_hours"]),
        "weather_source": str(first["weather_source"]),
        "model_version": str(first["model_version"]),
        "any_warning": bool(d["warning"].astype(bool).any()),
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
