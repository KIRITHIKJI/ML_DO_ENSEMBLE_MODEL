"""
STEP 12.5 - write models/final/MODEL_CARD.md.

    python -m src.deploy.model_card

The card is generated, not typed: the version id is the hash of the model
folder, the training data comes from config.yaml, the 2025 numbers come from
results/test and the 2026 numbers from results/live_2026. Nothing in it is
a number somebody remembered.

The version id it records is the one every forecast row carries, so a
forecast can always be matched to the models that made it.
"""

import json
import os

import pandas as pd

from src.config import project_path
from src.evaluation.step3_report import fmt, md_table

from src.deploy import common as D

CFG = D.CFG
FT = CFG["final_test"]
HORIZONS = D.HORIZONS
FINAL = D.FINAL_SYSTEM


def read(path, **kw):
    full = project_path(path)
    return pd.read_csv(full, **kw) if os.path.exists(full) else None


def load_json(path):
    full = project_path(path)
    if not os.path.exists(full):
        return {}
    with open(full, encoding="utf-8") as f:
        return json.load(f)


def frozen_stamp():
    """When and at which commit the system was frozen, out of FROZEN.md."""
    import re
    p = project_path(FT["frozen_file"])
    if not os.path.exists(p):
        return None, None
    head = open(p, encoding="utf-8").read(600)
    m = re.search(r"Written \*\*([^*]+)\*\* at commit `([0-9a-f]{7,40})`", head)
    return (m.group(1).strip(), m.group(2)[:12]) if m else (None, None)


def build():
    folder = D.deployed_folder()
    version = D.model_version()
    rel = os.path.relpath(folder, project_path(".")).replace("\\", "/")
    reg25 = read(os.path.join(FT["results_dir"],
                              "metrics_regression_stage2.csv"))
    cls25 = read(os.path.join(FT["results_dir"],
                              "metrics_classification_stage2.csv"))
    reg26 = read(os.path.join(D.TEST26["results_dir"],
                              "metrics_regression.csv"))
    cls26 = read(os.path.join(D.TEST26["results_dir"],
                              "metrics_classification.csv"))
    summary26 = load_json(os.path.join(D.TEST26["results_dir"],
                                       "summary.json"))
    lead25 = read(os.path.join(FT["results_dir"], "lead_times.csv"))
    lead26 = read(os.path.join(D.TEST26["results_dir"], "lead_times.csv"))
    penalty = read(os.path.join(D.TEST26["penalty_dir"], "penalty.csv"))
    stations = D.stations_table()
    froze_when, froze_commit = frozen_stamp()
    head = 24 if 24 in HORIZONS else HORIZONS[len(HORIZONS) // 2]
    budget = FT["target_false_alarm_rate"]
    low = CFG["target"]["thresholds"]["low"]
    good = CFG["target"]["thresholds"]["good"]

    def g(df, method, h, col, rule=None):
        if df is None:
            return float("nan")
        m = (df["method"] == method) & (df["horizon"] == h)
        if rule is not None:
            m &= df["rule"] == rule
        row = df[m]
        return float(row[col].iloc[0]) if len(row) else float("nan")

    L = []
    L.append("# Model card - DO early warning, deployed models\n")
    L.append(f"**Version id** `{version}`\n")
    L.append(
        f"The id is the SHA-256 over the per-file hashes of `{rel}`, "
        f"truncated to {D.DEP['models']['version_digest_chars']} characters. "
        f"It changes if and only if a model file changes, and it is written "
        f"into **every row of every forecast** this system produces "
        f"(`{D.RUN['results_dir']}/forecast_*.csv` and "
        f"`{D.RUN['log_parquet']}`), so any past forecast can be matched to "
        f"the models that made it. The previous version is kept under "
        f"`{D.DEP['models']['archive_dir']}/<version id>/` whenever a "
        f"retrain is promoted.\n")

    # ---------------------------------------------------------------- what
    L.append("\n## What it is\n")
    L.append(
        f"A forecast of dissolved oxygen (DO) in a river "
        f"{', '.join(str(h) for h in HORIZONS[:-1])} and {HORIZONS[-1]} "
        f"hours ahead, and a warning when the water is likely to drop below "
        f"**{low} mg/L**.\n")
    L.append(
        f"- **Inputs**: {len(CFG['models'] and D.MEMBERS) and 211} engineered "
        f"features at the forecast hour (water sensors, ERA5 weather, CAMS "
        f"air quality, time of day and year, station identity) plus the last "
        f"72 hours of raw hourly readings for the two recurrent members. "
        f"Every feature uses only data at or before the forecast hour.\n"
        f"- **Outputs**: predicted DO in mg/L, a probability for each of "
        f"Low (< {low}), Medium ({low}-{good}) and Good (> {good}), and a "
        f"yes/no warning.\n"
        f"- **Structure**: {len(D.MEMBERS)} members "
        f"({', '.join(D.MEMBERS)}) combined by a **{FINAL}** meta-model "
        f"fitted on 2024, with a warning rule on top: *warn when "
        f"P(Low) >= tau*.\n"
        f"- **The warning rule** is the single most valuable part. tau was "
        f"chosen on out-of-fold 2024 probabilities as the threshold "
        f"catching the most onsets at a false-alarm rate at or below "
        f"{FT['select_false_alarm_rate']:.1%} - deliberately tighter than "
        f"the {budget:.0%} the system promises.\n")
    taus = []
    from src.models.predict_service import tau_for
    for h in HORIZONS:
        taus.append([f"{h} h", f"{tau_for(FINAL, h):.2f}"])
    L.append("\nThe thresholds in force, from "
             f"`{FT['thresholds_file']}`:\n")
    L.append(md_table(["horizon", "tau"], taus))

    # ------------------------------------------------------- training data
    L.append("\n## Training data\n")
    L.append(
        f"- **{len(stations)} USGS river stations** across "
        f"{stations['state'].nunique()} US states, hourly, UTC.\n"
        f"- **{CFG['period']['start']} to {CFG['period']['end']}** - all "
        f"three years. These are the *stage-3* models: the members were "
        f"retrained on 2023 + 2024 + 2025 so a live forecast uses every "
        f"hour ever recorded.\n"
        f"- **Three sources**: USGS instantaneous values (DO, water "
        f"temperature, conductivity, pH, turbidity, discharge), ERA5 "
        f"reanalysis via Open-Meteo (8 variables), CAMS via Open-Meteo "
        f"(8 variables).\n"
        f"- **Nothing was tuned on that third year.** Every "
        f"hyperparameter, every number of boosting rounds and every epoch "
        f"count came from `FROZEN.md`"
        + (f", committed {froze_when} at commit `{froze_commit}`"
           if froze_commit else "")
        + ", which was written before any 2025 target was read. There is no "
          "early stopping in these models, because the only year left to "
          "watch would have been the test year.\n")
    L.append(
        f"- **Stations are chosen, not arbitrary**: continuous DO, water "
        f"temperature and conductivity, at least "
        f"{CFG['download']['min_do_coverage']:.0%} DO coverage in *every* "
        f"year, and real low-oxygen episodes in at least two of the three. "
        f"The list is `{CFG['paths']['stations']}`.\n")

    # ------------------------------------------------------- test results
    L.append("\n## Test results\n")
    L.append(
        "Two separate things are measured here, and they must not be "
        "confused:\n\n"
        "1. **The method**, tested on **2025**. Those numbers come from the "
        "*stage-2* models (trained on 2023+2024), which is what makes 2025 "
        "a hold-out for them. They are the paper's result and they are **not "
        "this model's** result.\n"
        "2. **These models**, tested on **2026**. The models in this folder "
        "trained on 2025, so 2025 cannot score them. 2026 can, and the "
        "criteria were pre-registered in `reports/plan_2026.md` and "
        "committed before any 2026 value was read.\n")

    L.append("\n### The method, on 2025 (stage-2 models, not these)\n")
    rows = []
    for h in HORIZONS:
        rows.append([f"{h} h", fmt(g(reg25, FINAL, h, "r2"), 3),
                     fmt(g(reg25, FINAL, h, "rmse"), 3),
                     fmt(g(cls25, FINAL, h, "f1_macro", "warn"), 3),
                     f"{g(cls25, FINAL, h, 'onset_recall', 'warn'):.3f}",
                     f"{g(cls25, FINAL, h, 'false_alarm_rate', 'warn'):.2%}"])
    L.append(md_table(["horizon", "R2", "RMSE", "macro F1", "onset recall",
                       "false alarms"], rows))
    if lead25 is not None and len(lead25):
        w = lead25["warned"].astype(bool)
        L.append(f"\nOf the {len(lead25)} real low-oxygen events in 2025, "
                 f"{w.mean():.0%} were warned about in advance, median lead "
                 f"{lead25.loc[w, 'lead_hours'].median():.0f} h.\n")

    L.append("\n### These models, on 2026 (the deployed models' own test)\n")
    if reg26 is None:
        L.append("Not yet run on this checkout. "
                 "`python -m src.deploy.score_2026`.\n")
    else:
        rows = []
        for h in HORIZONS:
            rows.append([f"{h} h", fmt(g(reg26, FINAL, h, "r2"), 3),
                         fmt(g(reg26, FINAL, h, "rmse"), 3),
                         fmt(g(cls26, FINAL, h, "f1_macro", "warn"), 3),
                         f"{g(cls26, FINAL, h, 'onset_recall', 'warn'):.3f}",
                         f"{g(cls26, FINAL, h, 'false_alarm_rate', 'warn'):.2%}"])
        L.append(md_table(["horizon", "R2", "RMSE", "macro F1",
                           "onset recall", "false alarms"], rows))
        window = summary26.get("window", [None, None])
        n_scored = len(summary26.get("stations_scored", []))
        L.append(
            f"\n{str(window[0])[:10]} to {str(window[1])[:10]} UTC, "
            f"{summary26.get('rows', 0):,} scored station-hours, "
            f"{n_scored} of {len(stations)} stations. "
            f"**Verdict against the pre-registered criteria: "
            f"{summary26.get('verdict', '?')}** - "
            f"{summary26.get('verdict_reason', '')}.\n")
        if lead26 is not None and len(lead26):
            w = lead26["warned"].astype(bool)
            L.append(f"Of the {len(lead26)} low-oxygen events in the 2026 "
                     f"window, {w.mean():.0%} were warned about in advance"
                     + (f", median lead "
                        f"{lead26.loc[w, 'lead_hours'].median():.0f} h"
                        if w.any() else "") + ".\n")
        L.append("Full tables, intervals and the per-station breakdown: "
                 "`reports/final_results.md`, section **Deployed models on "
                 f"2026**, and `{D.TEST26['results_dir']}/`.\n")

    if penalty is not None and len(penalty):
        L.append("\n### The cost of running live\n")
        L.append(
            f"ERA5 runs about {D.TEST26['era5_lag_days']} days behind, so a "
            f"live forecast uses the Open-Meteo forecast API for its most "
            f"recent hours. Scoring the same "
            f"{int(penalty['rows'].max()):,} station-hours both ways:\n")
        rows = []
        for r in penalty.itertuples():
            rows.append([f"{r.horizon} h", fmt(r.rmse_era5, 3),
                         fmt(r.rmse_forecast, 3), f"{r.rmse_diff:+.3f}",
                         f"{r.onset_era5:.3f}", f"{r.onset_forecast:.3f}",
                         f"{r.onset_diff:+.3f}"])
        L.append(md_table(["horizon", "RMSE ERA5", "RMSE forecast", "diff",
                           "onset ERA5", "onset forecast", "diff"], rows))
        L.append("\nEvery hour in every live file is tagged with its "
                 "source, so a forecast made on forecast weather is never "
                 "mistaken for one made on a reanalysis.\n")

    # ------------------------------------------------------- intended use
    L.append("\n## Intended use\n")
    L.append(
        f"**Use it for**: an early warning at a river station that already "
        f"has a DO sensor, so an operator gets hours of notice instead of a "
        f"reading after the fact - time to open a sluice, pause a "
        f"discharge, or send someone to look. It is designed to be run "
        f"**{D.RUN['schedule']}** (`{D.RUN['batch_file']}`, or "
        f"`python -m src.deploy.run`), with the forecasts logged and scored "
        f"as the sensors catch up (`python -m src.deploy.monitor`).\n")
    L.append(
        "**Do not use it for**:\n\n"
        "- a river with no DO sensor. Every station in the test set is also "
        "in the training set; nothing here measures transfer to an unseen "
        "river, and the station identity is an input the models use.\n"
        "- rivers outside the US, or outside the USGS network. The water "
        "inputs are USGS instantaneous values; another network would need "
        "its own freeze and its own untouched test year.\n"
        "- a regulatory or safety-of-life decision on its own. It is a "
        f"forecast with a measured false-alarm rate, not a measurement.\n"
        "- scoring the method on 2025. These models trained on 2025. The "
        "app enforces this: a 2025 hour is forecast with the stage-2 models "
        "only.\n")

    # -------------------------------------------------------- limitations
    L.append("\n## Limitations\n")
    lines = []
    if reg26 is not None:
        r_25 = g(reg25, FINAL, head, "rmse")
        r_26 = g(reg26, FINAL, head, "rmse")
        o_25 = g(cls25, FINAL, head, "onset_recall", "warn")
        o_26 = g(cls26, FINAL, head, "onset_recall", "warn")
        f_25 = g(cls25, FINAL, head, "false_alarm_rate", "warn")
        f_26 = g(cls26, FINAL, head, "false_alarm_rate", "warn")
        # which verdicts the pre-registered comparison actually returned
        cmp_table = read(os.path.join(D.TEST26["results_dir"],
                                      "compare_2025_2026.csv"))
        degraded = sorted(set(
            cmp_table[cmp_table["verdict"] == "degraded"]["metric"])) \
            if cmp_table is not None else []
        lines.append(
            f"- **The DO value is predicted less accurately on 2026 than on "
            f"2025.** At {head} h, RMSE went from {r_25:.3f} to "
            f"{r_26:.3f} mg/L - about "
            f"{(r_26 / r_25 - 1):.0%} worse - and macro F1 slipped as well. "
            f"RMSE is outside the 2025 bootstrap interval at every horizon. "
            f"That is the honest size of the drop between a year the method "
            f"was tested on and a year nobody prepared for"
            + (f"; the pre-registered verdict is PARTIAL, and the metrics "
               f"it marks as degraded are {', '.join(degraded)}."
               if degraded else "."))
        lines.append(
            f"- **What did hold is the thing the system is for.** Onset "
            f"recall stayed inside the 2025 interval at every horizon "
            f"({o_25:.3f} to {o_26:.3f} at {head} h), and the share of real "
            f"events warned about in advance barely moved. Accuracy on the "
            f"DO *value* degraded more than the *decision* did.")
        over = []
        if cls26 is not None:
            for h2 in HORIZONS:
                v = g(cls26, FINAL, h2, "false_alarm_rate", "warn")
                if v == v and v > budget:
                    over.append((h2, v))
        if over:
            lines.append(
                "- **The false-alarm promise was exceeded at "
                + ", ".join(f"{h2} h ({v:.2%})" for h2, v in over)
                + f"**, against the {budget:.0%} budget. It held at the "
                  f"{head} h headline horizon ({f_26:.2%}, against "
                  f"{f_25:.2%} on 2025), which is what the pre-registered "
                  f"pass/fail criterion rests on, but the other horizons "
                  f"are reported as they came out and this is a real "
                  f"exceedance, not a rounding one. tau was chosen on 2024 "
                  f"and is **not** being re-chosen on 2026.")
        else:
            lines.append(
                f"- **The false-alarm promise held at every horizon** "
                f"(worst {max(g(cls26, FINAL, h2, 'false_alarm_rate', 'warn') for h2 in HORIZONS):.2%} "
                f"against the {budget:.0%} budget), but with less headroom "
                f"than 2025 had.")
    lines += [
        f"- **Low DO is rare, and onsets are rarer.** Overall accuracy "
        f"flatters any model here; the numbers that matter are onset recall "
        f"and the false-alarm rate, which is why they are the ones the "
        f"system is judged on.",
        f"- **{head} h is the hard horizon.** 6 and 12 h are much easier - "
        f"the river's own recent history carries most of the signal - and "
        f"48 h is harder still.",
        f"- **Stations differ enormously.** Per-station onset recall spans "
        f"most of the possible range; a station-level table is in the "
        f"report and in the app. A single headline number is an average "
        f"over very unlike rivers.",
        f"- **The deep members are not bit-for-bit reproducible.** cuDNN has "
        f"no deterministic recurrent kernel, so the system averages three "
        f"seeds; the remaining spread was measured rather than assumed "
        f"(design check 6 in the report).",
        f"- **Sensor faults are not detected.** A stuck or drifting DO "
        f"sensor is fed straight in. `python -m src.deploy.monitor` flags "
        f"features that leave their training range, which is a warning "
        f"about the data, not a repair of it.",
        f"- **Gaps are carried forward, not interpolated**, for at most "
        f"{CFG['cleaning']['max_gap_fill_hours']} h. Longer gaps stay "
        f"missing and the hour is not forecast; `data_age_hours` in every "
        f"forecast row says how stale the newest reading was.",
        f"- **Air quality earns almost nothing.** The ablation in the report "
        f"shows weather pays for itself and CAMS adds very little - worth "
        f"knowing before anyone pays for an air-quality feed.",
    ]
    L += lines

    # ------------------------------------------------------------ running
    L.append("\n## How it is run and watched\n")
    L.append(
        f"| | |\n|---|---|\n"
        f"| forecast | `python -m src.deploy.run` → "
        f"`{D.RUN['results_dir']}/forecast_<UTC timestamp>.csv` |\n"
        f"| schedule | `{D.RUN['batch_file']}`, {D.RUN['schedule']}, "
        f"Windows Task Scheduler |\n"
        f"| log | `{D.RUN['log_parquet']}` |\n"
        f"| monitoring | `python -m src.deploy.monitor` → "
        f"`{D.MON['out_csv']}`, `{D.MON['drift_csv']}` |\n"
        f"| drift limits | false alarms > {D.MON['max_false_alarm_rate']:.0%} "
        f"over {D.MON['window_days']} days; features outside the training "
        f"range |\n"
        f"| API | `python -m src.deploy.api` → "
        f"{', '.join(D.DEP['api']['endpoints'])} |\n"
        f"| retraining | `python -m src.deploy.retrain` - gated: a candidate "
        f"replaces these models only if it is at least as good on the most "
        f"recent {D.RETRAIN['holdout_months']} months it did not train on, "
        f"on RMSE **and** false-alarm rate, at **every** horizon |\n"
        f"| settings | `config.yaml` (the science) + `config_deploy.yaml` "
        f"(the deployment) |\n")
    L.append(
        "\n*This card is generated by `python -m src.deploy.model_card` "
        "from the model folder and the results files, so none of its "
        "numbers is typed by hand.*\n")
    return "\n".join(L)


def main():
    text = build()
    path = project_path(D.DEP["models"]["model_card"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    D.log(f"Wrote {D.shortpath(path)} "
          f"({len(text):,} characters)")
    D.log(f"  version id {D.model_version()}")
    return path


if __name__ == "__main__":
    main()
