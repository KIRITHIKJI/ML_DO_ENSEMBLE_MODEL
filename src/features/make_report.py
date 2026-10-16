"""
STEP 2.6 - Report on the features and the split.

Run from the project root (after src.features.build_features):

    python -m src.features.make_report

Writes reports/features_report.md:
  * rows per split and number of features per group
  * per split and horizon: how many rows have a usable target, and the
    Low / Medium / Good shares of those targets
  * missingness after feature building, per group and the 10 worst columns
  * the water variables that cleaning dropped at each station

The test-year numbers are descriptive only. They are here so the data is
documented in one place; no feature or model decision may be made from them.
"""

import json
import os

import pandas as pd

from src.config import load_config, project_path

CFG = load_config()
P = CFG["paths"]
HORIZONS = CFG["horizons"]
CLASS_NAMES = CFG["target"]["class_names"]
SPLITS = ["train", "val", "test"]
SPLIT_YEARS = {"train": CFG["split"]["train_years"],
               "val": CFG["split"]["val_years"],
               "test": CFG["split"]["test_years"]}


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def main():
    with open(project_path(P["feature_groups"]), encoding="utf-8") as f:
        groups = json.load(f)
    feature_cols = [c for g in ["water", "weather", "air", "time", "station"]
                    for c in groups[g]]

    out = ["# Feature report\n",
           f"- Built from `{P['hourly']}` by `src/features/build_features.py`",
           f"- Horizons: **{HORIZONS} hours ahead**",
           f"- Classes: {CLASS_NAMES[0]} < {CFG['target']['thresholds']['low']} "
           f"mg/L, {CLASS_NAMES[1]} "
           f"{CFG['target']['thresholds']['low']}-{CFG['target']['thresholds']['good']}, "
           f"{CLASS_NAMES[2]} > {CFG['target']['thresholds']['good']}",
           "- Every feature at time t uses only data at times <= t, computed "
           "per station. No statistic is computed over the whole dataset.",
           "- Rows with no DO reading at t are dropped; rows with a missing "
           "target are kept but must be skipped for that horizon.",
           "- **The test columns are descriptive only. 2025 must not "
           "influence any choice until the final step.**\n"]

    # ---------------- rows per split ----------------
    frames, rows = {}, []
    for name in SPLITS:
        df = pd.read_parquet(project_path(P[name]))
        frames[name] = df
        rows.append([name, SPLIT_YEARS[name], f"{len(df):,}",
                     df["station"].nunique(),
                     f"{df['time'].min():%Y-%m-%d} .. {df['time'].max():%Y-%m-%d}"])
    out.append("## Rows per split\n")
    out.append(md_table(["split", "years", "rows", "stations", "time range"], rows))

    # ---------------- features per group ----------------
    out.append("\n## Features per group\n")
    grows = [[g, len(groups[g]), ", ".join(groups[g][:3]) + (" ..." if len(groups[g]) > 3 else "")]
             for g in ["water", "weather", "air", "time", "station"]]
    grows.append(["**total features**", len(feature_cols), ""])
    grows.append(["targets", len(groups["targets"]), ", ".join(groups["targets"][:4]) + " ..."])
    out.append(md_table(["group", "columns", "examples"], grows))

    # ---------------- targets per split and horizon ----------------
    out.append("\n## Usable targets, and their Low / Medium / Good shares\n")
    head = ["split", "horizon", "rows with target", "% of split"] + \
           [f"% {c}" for c in CLASS_NAMES]
    trows = []
    for name in SPLITS:
        df = frames[name]
        for h in HORIZONS:
            y = df[f"y_cls_{h}"].dropna()
            share = [100 * (y == k).mean() for k in range(3)]
            trows.append([name, f"{h} h", f"{len(y):,}",
                          f"{100 * len(y) / len(df):.1f}"]
                         + [f"{s:.1f}" for s in share])
    out.append(md_table(head, trows))

    # ---------------- missingness ----------------
    out.append("\n## Missing values after feature building\n")
    out.append("Share of rows with no value, averaged over the columns of "
               "each group (train split):\n")
    tr = frames["train"]
    mrows = []
    for g in ["water", "weather", "air", "time", "station"]:
        miss = tr[groups[g]].isna().mean()
        mrows.append([g, len(groups[g]), f"{100 * miss.mean():.1f}",
                      f"{100 * miss.max():.1f}"])
    out.append(md_table(["group", "columns", "mean % missing", "worst column %"],
                        mrows))

    out.append("\n### The 10 worst columns (train split)\n")
    worst = tr[feature_cols].isna().mean().sort_values(ascending=False).head(10)
    out.append(md_table(["column", "group", "% missing"],
                        [[c, next(g for g in ["water", "weather", "air", "time", "station"]
                                  if c in groups[g]), f"{100 * v:.1f}"]
                         for c, v in worst.items()]))
    out.append("\nTurbidity and discharge are the worst: several stations have "
               "no such sensor at all, and the cleaning step blanks a variable "
               "that is more than half missing in the training year. Tree "
               "models handle these NaNs natively; the deep-learning models "
               "will need them imputed, fitted on train only.\n")

    # ---------------- variables dropped by cleaning ----------------
    dropped = pd.read_csv(project_path(P["dropped_vars"]), dtype={"site_no": str})
    out.append("## Water variables dropped per station (cleaning step)\n")
    out.append(f"A variable more than "
               f"{100 * CFG['cleaning']['drop_variable_if_missing_above']:.0f}% "
               f"missing in the training years is blanked for that station. "
               f"Measured on {CFG['split']['train_years']} only, so the "
               f"validation and test years never affect the choice.\n")
    if len(dropped):
        out.append(md_table(["station", "variable", "% missing in train"],
                            [[r.site_no, r.variable, f"{100 * r.missing_in_train:.0f}"]
                             for r in dropped.itertuples()]))
    else:
        out.append("None.\n")

    out.append("\n## Known caveat: short-gap interpolation\n")
    out.append(f"Cleaning fills water-variable gaps of up to "
               f"{CFG['cleaning']['max_gap_fill_hours']} h by linear "
               "interpolation, as specified. Linear interpolation looks at the "
               "reading on *both* sides of a gap, so a filled input value at "
               "time t can contain up to "
               f"{CFG['cleaning']['max_gap_fill_hours']} h of future sensor "
               "information. This affects inputs only: the targets are taken "
               "from observed values (`*_was_missing == 0`), and the "
               "`<var>_was_missing` columns mark every filled hour, so the "
               "effect can be measured or the filling switched off "
               "(`cleaning.max_gap_fill_hours: 0`). It touches about "
               f"{100 * tr['do_mgl' + CFG['cleaning']['mask_suffix']].mean():.1f}% "
               "of training rows for DO.\n")

    out.append("\n## Note on negative discharge\n")
    out.append("The cleaning rule `discharge_cfs >= 0` removes about 72,000 "
               "hourly values, almost all of them at tidal stations (Savannah "
               "River, Middle River, Waccamaw, Murderkill), where a negative "
               "value is the real tide pushing water upstream rather than a "
               "sensor fault. Those hours are marked by "
               "`discharge_cfs_was_missing = 1`. To keep them instead, set "
               "`cleaning.valid_ranges.discharge_cfs.min: null` in config.yaml "
               "and re-run the cleaning step.\n")

    path = project_path(P["features_report"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Wrote {path}\n")

    # ---------------- terminal summary ----------------
    print("ROWS PER SPLIT")
    for name in SPLITS:
        print(f"  {name:<6}{len(frames[name]):>10,} rows   "
              f"{frames[name]['time'].min():%Y-%m-%d} .. "
              f"{frames[name]['time'].max():%Y-%m-%d}")
    print("\nFEATURES PER GROUP")
    for g in ["water", "weather", "air", "time", "station"]:
        print(f"  {g:<9}{len(groups[g]):>4}")
    print(f"  {'TOTAL':<9}{len(feature_cols):>4}")
    print(f"\nTARGET CLASS SHARES, horizon 24 h")
    print(f"  {'split':<7}{'rows':>10}  {'% Low':>7}{'% Medium':>10}{'% Good':>8}")
    for name in SPLITS:
        y = frames[name]["y_cls_24"].dropna()
        print(f"  {name:<7}{len(y):>10,}  "
              + "".join(f"{100 * (y == k).mean():>7.1f}   " for k in range(3)))


if __name__ == "__main__":
    main()
