"""
STEP 1 - Check the raw data and write a report.

Run from the project root:

    python -m src.data.check_data

Reads data/raw/stations.csv and the water files, then writes
reports/data_report.md with, for every station and every year:

  * rows                 - how many sensor readings there are
  * % missing            - per water variable (share of hours with no value)
  * hours DO < 5         - hours whose mean DO is below the Low threshold
  * low-DO events        - an event starts when DO drops below 5 after at
                           least 24 h at or above 5, so one long sag counts
                           once instead of hundreds of times
  * Low / Medium / Good  - share of hours in each DO class

It also prints a totals row per year to the terminal. Nothing is modified:
this script only reads.
"""

import os

import pandas as pd

from src.config import load_config, project_path

CFG = load_config()
P = CFG["paths"]
YEARS = sorted([y for ys in (CFG["split"]["train_years"],
                             CFG["split"]["val_years"],
                             CFG["split"]["test_years"]) for y in ys])
SPLIT_OF = {}
for name, key in [("train", "train_years"), ("val", "val_years"),
                  ("test", "test_years")]:
    for y in CFG["split"][key]:
        SPLIT_OF[y] = name

DO_COL = CFG["target"]["column"]
LOW = CFG["target"]["thresholds"]["low"]
GOOD = CFG["target"]["thresholds"]["good"]
CLASS_NAMES = CFG["target"]["class_names"]          # Low, Medium, Good
RECOVERY_H = CFG["event"]["recovery_hours"]

WATER_DIR = project_path(P["water"])
STATIONS_CSV = project_path(P["stations"])
REPORT = project_path(P["data_report"])

# The water variables we report missingness for, in a fixed order.
WATER_VARS = list(CFG["download"]["usgs_params"].values())


def hourly(water):
    """Average the 15-minute readings into hourly means."""
    return water.resample("h").mean()


def count_events(do_hourly):
    """
    Number of low-DO events. An event is the first hour below LOW that follows
    at least RECOVERY_H hours at or above LOW (missing hours do not count as
    recovery). The first low hour of the year also counts as an event start.
    """
    below = do_hourly < LOW
    # hours that are definitely not low (a value exists and it is >= LOW)
    recovered = do_hourly.notna() & (do_hourly >= LOW)
    # how many of the previous RECOVERY_H hours were recovered
    run = recovered.rolling(RECOVERY_H, min_periods=1).sum()
    prev_run = run.shift(1).fillna(RECOVERY_H)      # start of series counts as recovered
    starts = below & (prev_run >= RECOVERY_H)
    return int(starts.sum())


def class_shares(do_hourly):
    """Share of the hours WITH a reading that fall in Low / Medium / Good."""
    s = do_hourly.dropna()
    if not len(s):
        return {c: float("nan") for c in CLASS_NAMES}, 0
    low = (s < LOW).sum()
    good = (s > GOOD).sum()
    med = len(s) - low - good
    return ({CLASS_NAMES[0]: 100 * low / len(s),
             CLASS_NAMES[1]: 100 * med / len(s),
             CLASS_NAMES[2]: 100 * good / len(s)}, len(s))


def station_year_rows(site, state, water):
    """One dict per year for one station."""
    rows = []
    for y in YEARS:
        w = water[water.index.year == y]
        full = pd.date_range(f"{y}-01-01", f"{y}-12-31 23:00", freq="h")
        h = hourly(w).reindex(full) if len(w) else pd.DataFrame(index=full)
        row = {"site_no": site, "state": state, "year": y,
               "split": SPLIT_OF.get(y, "-"), "rows": len(w),
               "hours_total": len(full)}
        for v in WATER_VARS:
            col = h[v] if v in h.columns else pd.Series(index=full, dtype=float)
            row[f"miss_{v}"] = 100 * float(col.isna().mean())
        do = h[DO_COL] if DO_COL in h.columns else pd.Series(index=full, dtype=float)
        row["hours_do_below_5"] = int((do < LOW).sum())
        row["events"] = count_events(do)
        shares, n_obs = class_shares(do)
        row["hours_with_do"] = n_obs
        for c in CLASS_NAMES:
            row[f"pct_{c}"] = shares[c]
        rows.append(row)
    return rows


def totals_per_year(df):
    """
    Totals row for each year. Hours and events are summed over stations;
    the class shares are over all stations' hours pooled together, which is
    why we kept the per-station hour counts.
    """
    out = []
    for y in YEARS:
        d = df[df["year"] == y]
        n_obs = d["hours_with_do"].sum()
        row = {"year": y, "split": SPLIT_OF.get(y, "-"), "stations": len(d),
               "rows": int(d["rows"].sum()),
               "hours_with_do": int(n_obs),
               "hours_do_below_5": int(d["hours_do_below_5"].sum()),
               "events": int(d["events"].sum())}
        for c in CLASS_NAMES:
            # weighted mean of the per-station shares
            row[f"pct_{c}"] = (float((d[f"pct_{c}"].fillna(0) * d["hours_with_do"]).sum()
                                     / n_obs) if n_obs else float("nan"))
        # average missingness per water variable, across stations
        for v in WATER_VARS:
            row[f"miss_{v}"] = float(d[f"miss_{v}"].mean())
        out.append(row)
    return pd.DataFrame(out)


def md_table(header, rows):
    """A markdown table from a header list and a list of string-cell lists."""
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def main():
    if not os.path.exists(STATIONS_CSV):
        raise SystemExit(f"{STATIONS_CSV} not found - run "
                         f"`python -m src.data.download` first.")
    stations = pd.read_csv(STATIONS_CSV, dtype={"site_no": str})
    print(f"Checking {len(stations)} stations from {STATIONS_CSV}")

    rows, missing_files = [], []
    for i, s in stations.iterrows():
        site = str(s["site_no"])
        path = os.path.join(WATER_DIR, f"{site}.csv")
        if not os.path.exists(path):
            missing_files.append(site)
            continue
        water = pd.read_csv(path, parse_dates=["time"]).set_index("time").sort_index()
        rows += station_year_rows(site, s["state"], water)
        print(f"  [{i + 1}/{len(stations)}] {site} ok")

    df = pd.DataFrame(rows)
    tot = totals_per_year(df)

    # ---------------- report ----------------
    out = []
    out.append("# Raw data report\n")
    out.append(f"- Period: **{CFG['period']['start']} .. {CFG['period']['end']}**")
    out.append(f"- Stations: **{df['site_no'].nunique()}**")
    out.append(f"- Split: train {CFG['split']['train_years']}, "
               f"validation {CFG['split']['val_years']}, "
               f"**test {CFG['split']['test_years']} (hold-out, do not look at "
               f"it until the final step)**")
    out.append(f"- DO classes: {CLASS_NAMES[0]} < {LOW} mg/L, "
               f"{CLASS_NAMES[1]} {LOW}-{GOOD} mg/L, {CLASS_NAMES[2]} > {GOOD} mg/L")
    out.append(f"- A *low-DO event* = DO drops below {LOW} mg/L after at least "
               f"{RECOVERY_H} h at or above it")
    out.append("- `rows` counts raw sensor readings (usually every 15 min); "
               "everything else is computed on hourly means.")
    out.append("- `% missing` is the share of the year's hours with no value "
               "for that variable.\n")
    if missing_files:
        out.append(f"> **Warning:** no water file for {', '.join(missing_files)}\n")

    out.append("## Totals per year\n")
    head = (["year", "split", "stations", "rows", "hours with DO",
             f"hours DO<{LOW}", "events"]
            + [f"% {c}" for c in CLASS_NAMES])
    out.append(md_table(head, [
        [str(r["year"]), r["split"], str(r["stations"]), f"{r['rows']:,}",
         f"{r['hours_with_do']:,}", f"{r['hours_do_below_5']:,}", str(r["events"])]
        + [f"{r[f'pct_{c}']:.1f}" for c in CLASS_NAMES]
        for _, r in tot.iterrows()]))

    out.append("\n## Average % missing per year (mean over stations)\n")
    out.append(md_table(["year"] + WATER_VARS, [
        [str(r["year"])] + [f"{r[f'miss_{v}']:.1f}" for v in WATER_VARS]
        for _, r in tot.iterrows()]))

    out.append("\n## Per station and year\n")
    head = (["site", "state", "year", "split", "rows"]
            + [f"% miss {v}" for v in WATER_VARS]
            + [f"hours DO<{LOW}", "events"] + [f"% {c}" for c in CLASS_NAMES])
    body = []
    for _, r in df.sort_values(["site_no", "year"]).iterrows():
        body.append([r["site_no"], r["state"], str(r["year"]), r["split"],
                     f"{r['rows']:,}"]
                    + [f"{r[f'miss_{v}']:.1f}" for v in WATER_VARS]
                    + [f"{r['hours_do_below_5']:,}", str(r["events"])]
                    + [("-" if pd.isna(r[f'pct_{c}']) else f"{r[f'pct_{c}']:.1f}")
                       for c in CLASS_NAMES])
    out.append(md_table(head, body))
    out.append("")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    # ---------------- terminal ----------------
    print(f"\nWrote {REPORT}\n")
    print("TOTALS PER YEAR")
    show = ["year", "split", "stations", "rows", "hours_with_do",
            "hours_do_below_5", "events"] + [f"pct_{c}" for c in CLASS_NAMES]
    print(tot[show].to_string(index=False,
                              float_format=lambda x: f"{x:.1f}"))
    if missing_files:
        print(f"\nWARNING: no water file for {', '.join(missing_files)}")


if __name__ == "__main__":
    main()
