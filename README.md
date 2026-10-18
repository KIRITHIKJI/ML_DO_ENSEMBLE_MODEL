# Early warning for low dissolved oxygen in US rivers

Forecasting dissolved oxygen (DO) 6, 12, 24 and 48 hours ahead at 30 USGS
river stations, and raising a warning before the water becomes dangerous.
Built from three data sources - water-quality sensors, weather and air
quality - and tested **once** on a year the system had never seen.

Below 5 mg/L, dissolved oxygen is stressful for fish and can be lethal. By
the time a sensor records it, the harm is done. A day's notice is the
difference between reacting and preparing.

## Headline results, 2025 test year

The system is a **stacking ensemble** over five models, with a warning rule
on top. Twenty-four hours ahead, on a year used for nothing but this test:

| | result |
|---|---|
| R² | **0.951** |
| RMSE | **0.672 mg/L** (11.1% better than assuming no change) |
| Macro F1 (Low / Medium / Good) | **0.886** |
| Low-class recall | **0.941** |
| Onsets caught (safe water turning Low) | **49.6%** |
| False alarm rate | **1.50%** (budget was 2%) |
| Real low-DO events warned in advance | **51.7% of 230**, median lead **20 h** |

The single most valuable component is not the ensemble - it is the
**warning rule**. Asking "is P(Low) above a threshold?" instead of "which
class is most likely?" roughly doubles the share of onsets caught, for
about one extra false alarm per hundred safe hours.

Full numbers: [reports/final_results.md](reports/final_results.md).

## How the evaluation was kept honest

| year | role |
|---|---|
| **2023** | training |
| **2024** | every choice: hyperparameters, ensemble, thresholds |
| **2025** | the test, scored **once** |

Before a single 2025 target was read, the entire system was written into
[FROZEN.md](FROZEN.md) - hyperparameters, the number of boosting rounds and
epochs, the ensemble, the warning thresholds, and a SHA-256 for all 77
model files - and committed to git. A test is only a test if it cannot be
repeated until it passes. `tests/test_final.py` re-checks those hashes.

Two ensemble methods were built, tested and **rejected** by rules written
before their results were known: ATAE (step 6) matched a plain average, and
CAE (step 7) failed its pre-registered decision rule outright. Both are
still scored in the results for the record.

## Data

| source | what | via |
|---|---|---|
| USGS water sensors | DO, temperature, conductivity, pH, turbidity, discharge | waterservices.usgs.gov |
| ERA5 reanalysis | temperature, humidity, rain, pressure, wind, radiation, soil moisture, evapotranspiration | Open-Meteo |
| CAMS | PM10, PM2.5, CO, NO₂, SO₂, ozone, aerosol optical depth, dust | Open-Meteo |

30 stations across 15 US states, hourly, 2023-01-01 to 2025-12-31
(789,120 station-hours). Stations were chosen to have continuous DO,
temperature and conductivity, at least 80% DO coverage in **every** year,
and real low-oxygen episodes in at least two of the three years.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

The deep models train on a GPU if one is present and fall back to CPU
otherwise. For an NVIDIA card, replace the CPU build of torch with the
CUDA one **after** installing the requirements:

```bash
pip uninstall -y torch
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```

`cu121` is correct for any driver from CUDA 12.1 upwards - a newer driver
runs it, because the CUDA driver is backward compatible. This project was
trained on an RTX 3050 Laptop GPU (6 GB) with `torch 2.2.2+cu121`.

## Running the pipeline

Every script runs from the project root as a module, and reads all its
settings from [config.yaml](config.yaml). Nothing is hard-coded.

```bash
# Step 1 - data
python -m src.data.download              # ~2 h, resumable, skips existing files
python -m src.data.check_data            # -> reports/data_report.md

# Step 2 - cleaning and features
python -m src.data.clean                 # -> data/interim/hourly.parquet
python -m src.features.build_features    # -> data/processed/{train,val,test}.parquet
python -m src.features.make_report       # -> reports/features_report.md

# Step 3 - baselines and tree models (validation only)
python -m src.models.baselines
python -m src.models.train_ml            # ~55 min
python -m src.evaluation.step3_report

# Steps 4-5 - deep models, then tuning them (GPU, validation only)
python -m src.models.train_dl --quick    # smoke test first
python -m src.models.train_dl
python -m src.evaluation.step4_report
python -m src.models.search_dl           # 10-run variant search
python -m src.models.finalize_dl         # 3 seeds, averaged
python -m src.evaluation.step5_report
python -m src.models.fix_dl_cls          # step 5b: one attempt at the classifier
python -m src.evaluation.step5b_report

# Step 6 - the adaptive threshold-aware ensemble (tested, not adopted)
python -m src.models.ensemble
python -m src.evaluation.step6_report

# Step 7 - cross-fitted comparison, context-aware ensemble, warning rule
python -m src.models.run_step7           # ~20 min
python -m src.evaluation.step7_report    # writes final_system to config.yaml

# Step 8 - FREEZE, then the one and only 2025 test
python -m src.models.freeze              # -> FROZEN.md   (commit this first)
python -m src.models.stage2              # retrain members on 2023+2024
python -m src.evaluation.test_2025       # score 2025, once
python -m src.evaluation.final_report    # -> reports/final_results.md
python -m src.models.stage2 --stage 3    # models for the app (never scored)
python -m src.models.sanity_stage3       # stage-3 sanity check

# Step 10 - additional analyses on what was already produced.
# Nothing here changes the frozen system; it re-scores existing predictions.
python -m src.evaluation.step10_checks   # design checks -> results/test/step10_*
python -m src.evaluation.final_report    # re-render the report with them

# Step 12 - deployment, and the 2026 test of the deployed models.
# Settings live in config_deploy.yaml; config.yaml is never touched again.
python -m src.deploy.fetch_live --mode history   # 2026 data, same sources
python -m src.deploy.score_2026                  # -> results/live_2026/
python -m src.deploy.fetch_live --mode penalty   # ERA5 vs the forecast API
python -m src.deploy.score_2026 --penalty        # the live-data penalty
python -m src.deploy.walk_forward                # 2024 / 2025 / 2026 table
python -m src.deploy.model_card                  # -> models/final/MODEL_CARD.md
python -m src.evaluation.final_report            # adds the 2026 section

pytest -q                                # the whole test suite
```

## Running it live

The deployment is a layer on top of the frozen system. Every setting it
needs - dates, paths, the schedule, the API, the drift limits, the
promotion gate - is in [config_deploy.yaml](config_deploy.yaml), loaded
alongside `config.yaml`. Nothing in `config.yaml`, `FROZEN.md`,
`models/stage2/` or any 2025 result is changed by any of it, and nothing is
retuned on new data.

```bash
python -m src.deploy.run         # fetch -> forecast every station -> csv + log
python -m src.deploy.monitor     # score the forecasts the sensors caught up with
python -m src.deploy.api         # GET /health /warnings /forecast/{station}
python -m src.deploy.retrain --dry-run   # exercise the promotion gate
```

A run writes `results/live/forecast_<UTC timestamp>.csv` - one row per
station and horizon, carrying the predicted DO, P(Low), the frozen
threshold tau, the warning flag, the **model version id**, the **weather
source** for that hour and how old the newest reading was - and appends the
same rows to `results/live/log.parquet`. It prints which stations are
under warning and which were skipped, with the reason. A station offline,
an API down or a missing column costs that station and nothing else.

### Scheduling it hourly on Windows

[`run_live.bat`](run_live.bat) does one run and then updates the
monitoring csv, logging both to `results\live\run_live.log`. Register it
with Task Scheduler (one line, from an **Administrator** prompt, with the
path to your checkout):

```bat
schtasks /create /tn "DO early warning" /tr "A:\ML_MAIN_V2\run_live.bat" ^
         /sc hourly /mo 1 /st 00:05 /rl LIMITED
```

Then:

```bat
schtasks /run    /tn "DO early warning"      REM run it once now
schtasks /query  /tn "DO early warning" /v   REM when it last ran, and its result
schtasks /change /tn "DO early warning" /disable
schtasks /delete /tn "DO early warning" /f
```

Or through the GUI: **Task Scheduler → Create Task**, Triggers → *Daily*,
repeat task every *1 hour* for *1 day*; Actions → *Start a program*,
Program `A:\ML_MAIN_V2\run_live.bat`, Start in `A:\ML_MAIN_V2`; General →
*Run whether user is logged on or not*.

The batch file prefers `.venv\Scripts\python.exe` and falls back to
whatever `python` is on PATH. It does **not** pass `--allow-cpu`: if the
CUDA build of torch is missing, the run stops and says so rather than
quietly forecasting on the CPU.

`GET /health` reports how many minutes ago the latest forecast was issued,
so an hourly schedule that has stopped is visible from the API as well as
from the app's **Live** tab.

### Monitoring and retraining

`python -m src.deploy.monitor` joins the logged forecasts to the DO that
actually arrived and writes `results/live/monitor.csv` (rolling 30-day
RMSE, macro F1, onset recall and false-alarm rate, per station and
overall) plus `results/live/drift.csv`. Two things get flagged: **data
drift**, a watched feature outside the range it had in training, and
**performance drift**, a false-alarm rate above the 2% the system promises
over the window.

`python -m src.deploy.retrain` retrains every member and the stacking layer
on all data up to a cut-off, with the frozen settings and round counts -
nothing is tuned. It has a **promotion gate**: the candidate replaces
`models/final` only if it is at least as good as the deployed models on the
most recent two months it did not train on, on RMSE *and* on false-alarm
rate, at *every* horizon. The old version moves to
`models/archive/<version id>/` first, and that version id is the one in
every forecast row it ever produced. `--dry-run` exercises the whole path,
including a deliberately worse candidate, without training anything.

## The app

```bash
streamlit run app/app.py
```

Six tabs: **Overview** (the problem and the headline numbers), **Predict**
(pick a station and hour, or upload a window of hourly readings, and get a
forecast plus a WARNING badge), **Monitor** (a station through a month of
2025 with the warning periods shaded), **Live** (the deployed system right
now: when it last ran, every station sorted by risk, P(Low) against tau,
data age and weather source, the stations it skipped, and the monitoring
numbers), **Results** (the 2025 tables, the onset/false-alarm trade-off,
lead times, the ablation and the per-station breakdown) and **Method** (how
it works, what was rejected, the design checks, and the scope).

Two example inputs are in `app/sample_data/` - one window taken from just
before a real low-oxygen event, one from healthy water. The app reads
everything from `results/`, `reports/` and `models/`; no number in it is
hard-coded, and the Live tab shows a clear empty state - with the commands
to fix it - when there has been no run.

### Which models the app uses

There are two sets, and the app does not let them be confused:

- **`models/stage2/`** - the members trained on **2023 + 2024**. These are
  the models the reported 2025 numbers come from, and a **2025 hour can
  only be forecast with these**. Both 2025 modes print what the oxygen
  actually did next to the forecast, and `models/final` trained on that
  hour, so a comparison using it would not be a real one.
- **`models/final/`** - the members retrained on **2023 + 2024 + 2025**,
  which is what every live forecast and the 2026 test use. These are the
  deployed models; see
  [models/final/MODEL_CARD.md](models/final/MODEL_CARD.md).

## Two test years

| year | role |
|---|---|
| **2023** | training |
| **2024** | every choice: hyperparameters, ensemble, thresholds |
| **2025** | the test of the **method**, scored once against `FROZEN.md` |
| **2026** | the test of the **deployed models**, scored once against [`reports/plan_2026.md`](reports/plan_2026.md) |

`models/final/` could not be scored on 2025, because 2025 was inside its
training data. 2026 is a year it did not train on, so it can be. The test
period, the stations, the metrics and the pass criteria - including the
2025 bootstrap intervals the result is judged against - were written into
`reports/plan_2026.md` and **committed on their own, before any 2026
dissolved-oxygen value was downloaded**. `git log` shows that order, and
`tests/test_deploy.py` checks it.

The result, the walk-forward table across all three years, and the
live-data penalty are in
[reports/final_results.md](reports/final_results.md) under **Deployed
models on 2026**, with the raw numbers in `results/live_2026/`. It is
reported as it came out.

## Project structure

```
config.yaml              every path, date, threshold, horizon and seed
FROZEN.md                the system as frozen before the 2025 test
requirements.txt

data/
  raw/                   water/ weather/ air/ + stations.csv   (not in git)
  interim/               hourly.parquet, cross-fitting caches  (not in git)
  processed/             train/val/test.parquet + feature_groups.json

src/
  config.py              loads config.yaml
  data/                  download.py, check_data.py, clean.py
  features/              build_features.py, make_report.py
  models/                baselines.py, train_ml.py                (step 3)
                         sequences.py, dl_models.py, train_dl.py  (steps 4-5)
                         search_dl.py, finalize_dl.py, fix_dl_cls.py
                         ensemble.py                              (ATAE, step 6)
                         crossfit.py, context_ensemble.py,
                         run_step7.py                             (step 7)
                         freeze.py, stage2.py, sanity_stage3.py   (step 8)
                         predict_service.py                       (used by the app)
  evaluation/            metrics.py, calibration.py and one report
                         module per step, plus step10_checks.py
                         (additional analyses, step 10)

models/
  ml/ dl/                step-3 and step-5 members (trained on 2023)
  ensemble/              stacking meta-models + frozen thresholds
  stage2/                members retrained on 2023+2024  (what was tested)
  final/                 members retrained on all years  (what the app uses)
  ablation/              the water / +weather / +air models

results/
  val/                   everything measured on 2024
  test/                  the 2025 results, predictions and plots,
                         and the step10_* design checks

reports/                 one markdown report per step
app/                     app.py + sample_data/
tests/                   85+ tests, mostly about leakage and the freeze
```

## Results summary

### 2025, 24 hours ahead

| method | R² | RMSE | macro F1 | onset recall | false alarms |
|---|---|---|---|---|---|
| persistence (no change) | 0.938 | 0.755 | 0.881 | 0.000 | 0.00% |
| best single (LightGBM) | 0.949 | 0.682 | 0.887 | 0.499 | 1.56% |
| simple average of 5 | 0.950 | 0.674 | 0.886 | 0.515 | 1.61% |
| **stacking (the system)** | **0.951** | **0.672** | **0.886** | **0.496** | **1.50%** |

Classification figures use the frozen warning rule. Persistence scores
exactly 0 on onset recall by definition: it never predicts a change.

### What each data source is worth (LightGBM, 24 h RMSE)

| inputs | features | RMSE |
|---|---|---|
| water + time + station | 156 | 0.700 |
| + weather | 195 | 0.683 |
| + air quality | 211 | 0.681 |

Weather earns its place. Air quality adds almost nothing - worth knowing
before anyone pays for an air-quality feed.

## Design choices and how they were checked

Each of these was a real design decision, and each one has a check behind
it. The checks are **additional analyses**: they re-score predictions that
already existed, and none of them was used to choose or change the
deployed system, which was fixed in [FROZEN.md](FROZEN.md) before any 2025
target was read. Full numbers and tables:
[reports/final_results.md](reports/final_results.md).

1. **Hyperparameters were chosen on 2024.** The standard three-way split:
   2023 trains, 2024 decides, 2025 judges. What 2025 tests is the whole
   frozen pipeline, which is what an operator actually deploys - and the
   freeze commit makes that claim checkable rather than asserted.
2. **The stacking meta-model was fitted on 2024 predictions of 2023-only
   members, then applied on top of members retrained on 2023+2024.** A
   second chain has no such mismatch (2023-only members throughout), and it
   is the *worse* of the two: at 24 h, RMSE 0.687 against 0.672, with onset
   recall 0.495 against 0.496 - a difference of -0.001 [-0.014, +0.013],
   which contains zero. The headline leads because its members saw an extra
   year, not because of the mismatch.
3. **One station stops reporting on 2025-11-19.** Re-scoring the same
   predictions without that station, and again on all 30 stations up to
   2025-11-18, moves 24 h onset recall only within 0.496-0.504 and RMSE
   within 0.664-0.672. The false-alarm rate stays inside the 2% promise in
   every subset at every horizon.
4. **The deep models are the weakest members and stay in for diversity.**
   Their 24 h errors correlate 0.905-0.955 with the trees', against
   0.963-0.976 among the trees themselves. Lower correlation means the
   members miss different hours, so combining them cancels more error than
   combining models that fail together.
5. **The warning rule is the main contribution of the decision layer.** At
   24 h it lifts onset recall from 0.285 to 0.496 - 1.7x as many onsets
   caught - while false alarms go from 0.46% to 1.50%, inside the 2%
   promise. Choosing stacking over the best single model, by comparison, is
   worth 0.010 mg/L of RMSE and nothing detectable on macro F1.
6. **GPU LSTM training is not bit-for-bit reproducible** (cuDNN has no
   deterministic recurrent kernel), so the frozen system averages three
   seeds and the remaining wobble was measured: RMSE +-0.0023 and macro F1
   +-0.0041 across seeds for a single deep model. The gaps between the
   ensembles are of the same order (0.0104 and 0.0019 mg/L of RMSE), which
   is why stacking was picked by a rule written in advance and why the
   report quotes bootstrap intervals. The margins over persistence, and the
   margin the warning rule adds, are an order of magnitude larger.

## Scope and future work

- **Evaluated at monitored stations** - the intended use: a warning for a
  river where a sensor already sits. Every test station also appears in
  training, so testing on rivers not seen in training is future work.
- **US rivers, USGS gauges.** Applying the pipeline to Indian rivers (the
  CPCB real-time stations, for example) is future work, and would need its
  own freeze and its own untouched test year.
- **One test year.** 2025 was scored once, which is what makes it honest,
  but one year is one draw; a further untouched year would narrow every
  interval here.
- **Adaptive weighting (ATAE) is worth revisiting.** Rejected on validation
  by a rule written in advance, it then came out level with or just ahead
  of stacking on 2025. That is a hint, not evidence - the comparison
  happened after the test year was opened - so only a new, unseen year can
  settle it.

## Reports

| | |
|---|---|
| [reports/data_report.md](reports/data_report.md) | the raw data, per station and year |
| [reports/features_report.md](reports/features_report.md) | cleaning and the 211 features |
| [reports/step3_ml_report.md](reports/step3_ml_report.md) | baselines and tree models |
| [reports/step4_dl_report.md](reports/step4_dl_report.md) | BiLSTM and CNN-LSTM |
| [reports/step5_dl_tuning_report.md](reports/step5_dl_tuning_report.md) | tuning the deep models |
| [reports/step5b_dl_cls_fix_report.md](reports/step5b_dl_cls_fix_report.md) | one attempt at their classifier |
| [reports/step6_ensemble_report.md](reports/step6_ensemble_report.md) | ATAE - tested, not adopted |
| [reports/step7_cae_report.md](reports/step7_cae_report.md) | CAE and the warning rule |
| [**reports/final_results.md**](reports/final_results.md) | **the 2025 test** |
