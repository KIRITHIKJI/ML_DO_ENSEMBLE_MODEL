# Dissolved-Oxygen Early Warning (V2)

Early warning of low dissolved oxygen (DO) in US rivers from multimodal data:
USGS water-quality sensors + ERA5 weather + CAMS air quality (both via Open-Meteo).

**Two tasks**, each at horizons **6, 12, 24 and 48 hours ahead**:

| Task | Target | Metrics |
|---|---|---|
| Regression | DO value in mg/L | R², RMSE, MAE, NSE |
| Classification | Low (<5) / Medium (5–7) / Good (>7) mg/L | accuracy, balanced accuracy, macro P/R/F1, ROC-AUC (OvR), confusion matrix |

**Data split (fixed, chronological):** train = 2023, validation = 2024, **test = 2025**.
2025 is a hold-out and must not influence any decision until the final step.

## Setup

```bash
pip install -r requirements.txt
```

Every script runs from the project root as a module, and reads all its
settings from [config.yaml](config.yaml):

```bash
python -m src.data.download      # fetch / top up the raw data
python -m src.data.check_data    # write reports/data_report.md
```

## Layout

```
config.yaml        all paths, dates, thresholds, horizons, seeds
data/raw/          water/ weather/ air/ + stations.csv   (not in git)
data/interim/      intermediate tables                   (not in git)
data/processed/    model-ready features                  (not in git)
src/data/          download.py, check_data.py
src/features/      feature engineering          (step 2)
src/models/        RF, XGBoost, LightGBM, BiLSTM, CNN-LSTM, CEEMDAN ensemble (step 3+)
src/evaluation/    metrics and plots            (step 4+)
results/ reports/  tables, figures, written reports
app/               Streamlit demo               (last step)
tests/ notebooks/
```

## Status

**Step 1 complete:** project skeleton, config, and raw data for 2023–2025.
Nothing is preprocessed, trained or evaluated yet.
