"""
Dissolved-oxygen early warning - the app.

    streamlit run app/app.py

Everything shown here is read from disk: the saved models in models/, the
results in results/, and the reports in reports/. Nothing is trained,
tuned or re-scored, and no number is hard-coded - if a figure appears on a
slide, it came out of a file that the pipeline wrote.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, project_path          # noqa: E402
from src.models import predict_service as PS              # noqa: E402

CFG = load_config()
FT = CFG["final_test"]
HORIZONS = CFG["horizons"]
CLASS_NAMES = CFG["target"]["class_names"]
LOW = CFG["target"]["thresholds"]["low"]
GOOD = CFG["target"]["thresholds"]["good"]
MEMBERS = CFG["ensemble"]["members"]
RESULTS = project_path(FT["results_dir"])
PLOTS = project_path(FT["plots_dir"])
CLASS_COLOUR = {"Low": "#d62728", "Medium": "#ff7f0e", "Good": "#2ca02c"}

st.set_page_config(page_title="DO Early Warning", page_icon="💧",
                   layout="wide")


# ---------------------------------------------------------------------------
# Loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path, str_cols=("station",)):
    """
    Station ids are strings with leading zeros ("02110701"); pandas would
    read them as integers and lose the zero, so they are forced to str.
    """
    p = os.path.join(RESULTS, path) if not os.path.isabs(path) else path
    if not os.path.exists(p):
        return None
    head = pd.read_csv(p, nrows=0)
    dtype = {c: str for c in str_cols if c in head.columns}
    return pd.read_csv(p, dtype=dtype or None)


@st.cache_data(show_spinner=False)
def load_stations():
    d = pd.read_csv(project_path(CFG["paths"]["stations"]), dtype={"site_no": str})
    return d.sort_values(["state", "site_no"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_pred(method, h):
    p = os.path.join(RESULTS, f"pred_{method}_h{h}.parquet")
    if not os.path.exists(p):
        return None
    d = pd.read_parquet(p)
    d["time"] = pd.to_datetime(d["time"])
    return d


@st.cache_resource(show_spinner=False)
def get_forecaster(stage):
    return PS.Forecaster(stage)


@st.cache_data(show_spinner=False)
def report_section(heading, stop_at=None):
    """
    One section of reports/final_results.md, verbatim. The report is the
    single source of truth for every number the app shows, so the Method
    tab quotes it rather than keeping its own copy that could drift.
    """
    path = project_path("reports", "final_results.md")
    if not os.path.exists(path):
        return None
    text = open(path, encoding="utf-8").read()
    if heading not in text:
        return None
    body = text.split(heading, 1)[1]
    if stop_at:
        for marker in ([stop_at] if isinstance(stop_at, str) else stop_at):
            if marker in body:
                body = body.split(marker, 1)[0]
    return body.strip()


def metric_row(reg, cls, method, h, rule="warn"):
    r = reg[(reg["method"] == method) & (reg["horizon"] == h)]
    c = cls[(cls["method"] == method) & (cls["horizon"] == h)
            & (cls["rule"] == rule)]
    return (r.iloc[0] if len(r) else None), (c.iloc[0] if len(c) else None)


REG = load_csv("metrics_regression_stage2.csv")
CLS = load_csv("metrics_classification_stage2.csv")
ABL = load_csv("ablation.csv")
LEAD = load_csv("lead_times.csv")
STATIONS = load_stations()
PER_STATION = load_csv("per_station_h24.csv")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("💧 Dissolved-oxygen early warning for US rivers")
st.caption("Forecasting low-oxygen conditions 6 to 48 hours ahead, from "
           "water-quality sensors, weather and air quality. "
           "Tested once on 2025, against a system frozen beforehand.")

tabs = st.tabs(["Overview", "Predict", "Monitor", "Results", "Method"])


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("The problem")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"""
Fish and river life need dissolved oxygen (DO). Below **{LOW} mg/L** water
is stressful and can become lethal; the damage is done by the time anyone
measures it. A forecast a day ahead is the difference between reacting and
preparing - opening a sluice, pausing a discharge, or simply sending
someone to look.

This system forecasts DO at **6, 12, 24 and 48 hours ahead** and raises a
warning when the probability of dropping below {LOW} mg/L is high enough to
act on.
        """)
        st.markdown("**The data**")
        st.markdown(f"""
- **{len(STATIONS)} USGS stations** across {STATIONS['state'].nunique()} US states, hourly, 2023-2025
- **Water sensors**: dissolved oxygen, temperature, conductivity, pH, turbidity, discharge
- **Weather** (ERA5 via Open-Meteo): temperature, humidity, rain, pressure, wind, radiation, soil moisture
- **Air quality** (CAMS via Open-Meteo): PM10, PM2.5, CO, NO₂, SO₂, ozone, aerosol, dust
        """)
    with c2:
        st.markdown("**The pipeline**")
        st.markdown("""
```
 USGS + ERA5 + CAMS
         │
   clean to hourly        (ranges, spikes,
         │                 forward-fill ≤3 h)
  211 features at t       (lags, rolling stats,
         │                 rain sums, saturation)
   ┌─────┴─────┐
   │ 5 models  │  RF · XGBoost · LightGBM
   │           │  BiLSTM · CNN-LSTM
   └─────┬─────┘
    stacking ensemble
         │
   warning rule          P(Low) ≥ τ
         │
   DO forecast + WARNING
```
        """)

    st.divider()
    st.header("Headline results on the 2025 test year")
    if REG is not None and CLS is not None:
        r, c = metric_row(REG, CLS, "stacking", 24)
        k = st.columns(5)
        k[0].metric("R² (24 h)", f"{r['r2']:.3f}")
        k[1].metric("RMSE (24 h)", f"{r['rmse']:.3f} mg/L",
                    f"{r['skill_vs_persistence']:+.1%} vs no-change")
        k[2].metric("Macro F1 (24 h)", f"{c['f1_macro']:.3f}")
        k[3].metric("Onsets caught", f"{c['onset_recall']:.1%}",
                    help="Share of the hours when safe water turns Low that "
                         "the system warned about, 24 h ahead.")
        k[4].metric("False alarms", f"{c['false_alarm_rate']:.2%}",
                    help=f"Budget was {FT['target_false_alarm_rate']:.0%}.")
        if LEAD is not None:
            warned = LEAD["warned"].mean()
            med = LEAD.loc[LEAD["warned"], "lead_hours"].median()
            st.success(f"**Of the {len(LEAD)} real low-oxygen events in 2025, "
                       f"{warned:.0%} were warned about in advance**, with a "
                       f"median lead time of {med:.0f} hours.")
    else:
        st.warning("Run `python -m src.evaluation.test_2025` to populate "
                   "the results.")


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("Make a forecast")
    stages = PS.available_stages()
    if not stages:
        st.error("No trained models found. Run `python -m src.models.stage2`.")
        st.stop()

    left, right = st.columns([1, 2])
    with left:
        default = "final" if "final" in stages else stages[0]
        stage = st.radio("Models to use", stages,
                         index=stages.index(default),
                         format_func=lambda s: {"final": "models/final (all data)",
                                                "stage2": "models/stage2 (tested)"}[s])
        st.caption(PS.STAGE_LABEL[stage])
        mode = st.radio("Input", ["Pick a station and time (2025)",
                                  "Upload a CSV"])

    forecaster = get_forecaster(stage)
    result = None

    if mode.startswith("Pick"):
        with left:
            labels = {f"{r.site_no} - {r['name'][:38]} ({r.state})": r.site_no
                      for _, r in STATIONS.iterrows()}
            pick = st.selectbox("Station", list(labels))
            station = labels[pick]
            day = st.date_input("Date in 2025",
                                value=pd.Timestamp("2025-07-15").date(),
                                min_value=pd.Timestamp("2025-01-01").date(),
                                max_value=pd.Timestamp("2025-12-31").date())
            hour = st.slider("Hour (UTC)", 0, 23, 12)
            go_btn = st.button("Forecast", type="primary")
        if go_btn:
            when = pd.Timestamp(day) + pd.Timedelta(hours=hour)
            try:
                with st.spinner("Running the five models and the ensemble..."):
                    res, row = forecaster.predict_station(station, when)
                result = (res, station, when, float(row["do_mgl"].iloc[0]),
                          {h: float(row[f"y_reg_{h}"].iloc[0])
                           if not pd.isna(row[f"y_reg_{h}"].iloc[0]) else None
                           for h in HORIZONS})
            except KeyError:
                st.error(f"No reading for {station} at {when}. That hour may "
                         f"be missing from the sensor record - try another.")
    else:
        with left:
            st.caption(f"At least 72 hourly rows; 480 h gives the best "
                       f"results because some features look back 7 days. "
                       f"Columns must match the template.")
            up = st.file_uploader("CSV file", type=["csv"])
            sample = st.selectbox("...or use a sample",
                                  ["-", "before a real low-DO event",
                                   "a healthy window"])
            st_pick = st.selectbox("Station identity (for its embedding)",
                                   ["auto"] + STATIONS["site_no"].tolist())
            go_btn = st.button("Forecast", type="primary", key="upload_btn")
        df = None
        if up is not None:
            df = pd.read_csv(up)
        elif sample != "-":
            f = ("sample_before_low_do_event" if "event" in sample
                 else "sample_healthy")
            df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                          "sample_data", f"{f}.csv"))
        if go_btn and df is not None:
            problems = PS.check_upload(df)
            if problems:
                for p in problems:
                    st.error(p)
                with st.expander("Required columns"):
                    st.code(", ".join(PS.template_columns()))
            else:
                with st.spinner("Building features and running the models..."):
                    out, s, t, do_now = PS.predict_from_hourly(
                        forecaster, df,
                        station=None if st_pick == "auto" else st_pick)
                result = ({h: {k: (v[0] if isinstance(v, np.ndarray) else v)
                               for k, v in d.items()
                               if k not in ("member_reg", "member_p_low")}
                           | {"member_reg": {m: float(a[0]) for m, a in
                                             d["member_reg"].items()},
                              "member_p_low": {m: float(a[0]) for m, a in
                                               d["member_p_low"].items()}}
                           for h, d in out.items()}, s, t, do_now, {})
        elif go_btn:
            st.info("Choose a file or a sample first.")

    with right:
        if result is None:
            st.info("Pick a station and press **Forecast**, or upload a "
                    "window of hourly readings.")
        else:
            res, station, when, do_now, truth = result
            st.subheader(f"Station {station} - forecast made at "
                         f"{pd.Timestamp(when):%Y-%m-%d %H:%M} UTC")
            st.caption(f"Dissolved oxygen right now: **{do_now:.2f} mg/L** "
                       f"({CLASS_NAMES[int(PS.classify(do_now))]})  ·  "
                       f"combined by **{res[HORIZONS[0]]['method']}**")
            cols = st.columns(len(HORIZONS))
            for col, h in zip(cols, HORIZONS):
                d = res[h]
                warn = bool(d["warn"])
                with col:
                    st.markdown(f"**{h} hours ahead**")
                    if warn:
                        st.markdown("<div style='background:#d62728;color:white;"
                                    "padding:8px 10px;border-radius:6px;"
                                    "font-weight:700;text-align:center'>"
                                    "⚠ WARNING</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='background:#2ca02c;color:white;"
                                    "padding:8px 10px;border-radius:6px;"
                                    "font-weight:700;text-align:center'>"
                                    "✓ NO WARNING</div>", unsafe_allow_html=True)
                    st.metric("predicted DO", f"{float(d['do']):.2f} mg/L",
                              f"{float(d['do']) - do_now:+.2f} vs now")
                    p = np.asarray(d["proba"], dtype=float).ravel()
                    for i, n in enumerate(CLASS_NAMES):
                        st.caption(f"{n}: {p[i]:.1%}")
                    st.caption(f"threshold τ = {d['tau']:.2f}"
                               if d["tau"] is not None else "rule: argmax")
                    if truth.get(h) is not None:
                        st.caption(f"actual was **{truth[h]:.2f} mg/L**")

            st.divider()
            st.markdown("**What each model said** (predicted DO, and its own "
                        "probability of Low)")
            tbl = pd.DataFrame({
                "model": list(res[24]["member_reg"]),
                **{f"{h} h DO": [res[h]["member_reg"][m]
                                 for m in res[24]["member_reg"]]
                   for h in HORIZONS},
                **{f"{h} h P(Low)": [res[h]["member_p_low"][m]
                                     for m in res[24]["member_reg"]]
                   for h in HORIZONS}})
            st.dataframe(
                tbl.style.format({c: "{:.2f}" for c in tbl.columns
                                  if "DO" in c}
                                 | {c: "{:.1%}" for c in tbl.columns
                                    if "P(Low)" in c}),
                use_container_width=True, hide_index=True)
            st.caption("The ensemble is a stacking model fitted on 2024, not "
                       "a simple average, so its answer need not sit between "
                       "the members'.")


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("Monitor a station through 2025")
    d24 = load_pred("stacking", 24)
    if d24 is None:
        st.warning("Run `python -m src.evaluation.test_2025` first.")
    else:
        c1, c2 = st.columns([1, 1])
        labels = {f"{r.site_no} - {r['name'][:38]} ({r.state})": r.site_no
                  for _, r in STATIONS.iterrows()}
        with c1:
            pick = st.selectbox("Station", list(labels), key="mon_station")
        station = labels[pick]
        months = sorted(d24[d24["station"] == station]["time"]
                        .dt.to_period("M").astype(str).unique())
        with c2:
            month = st.selectbox("Month of 2025", months,
                                 index=min(6, len(months) - 1))
        g = d24[(d24["station"] == station)
                & (d24["time"].dt.to_period("M").astype(str) == month)]
        g = g.sort_values("time")

        if not len(g):
            st.info("No rows for that station and month.")
        else:
            fig = go.Figure()
            warn = g["pred_warn"] == 0
            if warn.any():
                blocks, start = [], None
                times = g["time"].tolist()
                for i, w in enumerate(warn.tolist()):
                    if w and start is None:
                        start = times[i]
                    if (not w or i == len(times) - 1) and start is not None:
                        blocks.append((start, times[i]))
                        start = None
                for a, b in blocks:
                    fig.add_vrect(x0=a, x1=b, fillcolor="#d62728",
                                  opacity=0.13, line_width=0)
            fig.add_hline(y=LOW, line_dash="dash", line_color="#ff7f0e",
                          annotation_text=f"Low threshold ({LOW} mg/L)")
            fig.add_trace(go.Scatter(x=g["time"], y=g["y_true_reg"],
                                     name="observed DO (at t+24 h)",
                                     line=dict(color="black", width=1.6)))
            fig.add_trace(go.Scatter(x=g["time"], y=g["pred_reg"],
                                     name="forecast (24 h ahead)",
                                     line=dict(color="#1f77b4", width=1.4)))
            fig.update_layout(height=430, hovermode="x unified",
                              yaxis_title="DO (mg/L)", xaxis_title="time (UTC)",
                              margin=dict(l=10, r=10, t=30, b=10),
                              legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Red bands are hours when the system issued a warning "
                       "for 24 hours later.")

            ok = g["y_true_reg"].notna()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("hours shown", f"{len(g):,}")
            m2.metric("hours warned", f"{int(warn.sum()):,}")
            m3.metric("hours truly Low",
                      f"{int((g['y_true_cls'] == 0).sum()):,}")
            if ok.any():
                err = (g.loc[ok, "pred_reg"] - g.loc[ok, "y_true_reg"])
                m4.metric("RMSE this month",
                          f"{np.sqrt((err ** 2).mean()):.3f} mg/L")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("Results on the 2025 test year")
    if REG is None:
        st.warning("Run `python -m src.evaluation.test_2025` first.")
    else:
        st.caption("Every number below comes from scoring 2025 once, with a "
                   "system frozen before the test. Nothing here was tuned "
                   "on 2025.")
        h = st.radio("Horizon", HORIZONS, index=2, horizontal=True,
                     format_func=lambda x: f"{x} hours")

        st.subheader("Predicting the DO value")
        r = REG[REG["horizon"] == h][["method", "r2", "rmse", "mae", "nse",
                                      "skill_vs_persistence", "danger_rmse"]]
        st.dataframe(r.rename(columns={
            "r2": "R²", "rmse": "RMSE", "mae": "MAE", "nse": "NSE",
            "skill_vs_persistence": "skill vs no-change",
            "danger_rmse": "RMSE in 3-7 mg/L"}).style.format(
                {c: "{:.4f}" for c in ["R²", "RMSE", "MAE", "NSE",
                                       "skill vs no-change",
                                       "RMSE in 3-7 mg/L"]}),
            use_container_width=True, hide_index=True)

        st.subheader("Classifying Low / Medium / Good")
        rule = st.radio("Decision rule", ["warn", "argmax"], horizontal=True,
                        format_func=lambda x: ("warning rule: P(Low) ≥ τ"
                                               if x == "warn"
                                               else "argmax (most likely class)"))
        c = CLS[(CLS["horizon"] == h) & (CLS["rule"] == rule)][
            ["method", "accuracy", "balanced_accuracy", "f1_macro",
             "roc_auc_ovr_macro", "recall_Low", "onset_recall",
             "false_alarm_rate", "transition_accuracy"]]
        st.dataframe(c.rename(columns={
            "accuracy": "accuracy", "balanced_accuracy": "balanced acc",
            "f1_macro": "macro F1", "roc_auc_ovr_macro": "ROC-AUC",
            "recall_Low": "Low recall", "onset_recall": "onset recall",
            "false_alarm_rate": "false alarms",
            "transition_accuracy": "transition acc"}).style.format(
                {c2: "{:.3f}" for c2 in ["accuracy", "balanced acc",
                                         "macro F1", "ROC-AUC", "Low recall",
                                         "onset recall", "transition acc"]}
                | {"false alarms": "{:.2%}"}),
            use_container_width=True, hide_index=True)
        st.caption("**Onset recall** is the share of hours where safe water "
                   "turns Low that the system caught - the number this "
                   "project exists for. Persistence scores 0 on it by "
                   "definition, because it never predicts a change.")

        st.subheader("Catching onsets without crying wolf")
        cc = CLS[(CLS["horizon"] == h) & (CLS["rule"] == "warn")]
        fig = go.Figure()
        for _, row in cc.iterrows():
            star = row["method"] == "stacking"
            fig.add_trace(go.Scatter(
                x=[row["false_alarm_rate"]], y=[row["onset_recall"]],
                mode="markers+text", text=[row["method"]],
                textposition="top center", name=row["method"],
                marker=dict(size=18 if star else 10,
                            symbol="star" if star else "circle",
                            color="#d62728" if star else "#1f77b4"),
                showlegend=False))
        fig.add_vline(x=FT["target_false_alarm_rate"], line_dash="dash",
                      line_color="#d62728",
                      annotation_text=f"budget {FT['target_false_alarm_rate']:.0%}")
        fig.update_layout(height=430, xaxis_title="false alarm rate",
                          yaxis_title="onset recall",
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Lead time")
            if LEAD is not None:
                w = LEAD["warned"]
                st.metric("events warned in advance",
                          f"{w.sum()} of {len(LEAD)} ({w.mean():.1%})")
                st.metric("median lead time",
                          f"{LEAD.loc[w, 'lead_hours'].median():.0f} hours")
                fig = go.Figure(go.Histogram(
                    x=LEAD.loc[w, "lead_hours"], nbinsx=24,
                    marker_color="#1f77b4"))
                fig.update_layout(height=260, xaxis_title="hours of warning "
                                  "before the event", yaxis_title="events",
                                  margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("What each data source is worth")
            if ABL is not None:
                a = ABL[ABL["horizon"] == h][["set", "n_features", "rmse",
                                              "f1_macro", "onset_recall"]]
                st.dataframe(a.rename(columns={
                    "set": "inputs", "n_features": "features",
                    "rmse": "RMSE", "f1_macro": "macro F1",
                    "onset_recall": "onset recall"}).style.format(
                        {"RMSE": "{:.4f}", "macro F1": "{:.3f}",
                         "onset recall": "{:.3f}"}),
                    use_container_width=True, hide_index=True)
                st.caption("**W** water only · **W+M** adds weather · "
                           "**W+M+A** adds air quality. LightGBM alone, so "
                           "the comparison is about the data, not the model.")

        st.subheader("Per station, 24 hours ahead")
        if PER_STATION is not None:
            p = PER_STATION.merge(
                STATIONS[["site_no", "state", "name"]],
                left_on="station", right_on="site_no", how="left")
            st.dataframe(p[["station", "state", "name", "rows", "rmse",
                            "f1_macro", "onset_recall", "false_alarm_rate",
                            "n_onsets"]].rename(columns={
                "rmse": "RMSE", "f1_macro": "macro F1",
                "onset_recall": "onset recall",
                "false_alarm_rate": "false alarms",
                "n_onsets": "onsets"}).style.format(
                    {"RMSE": "{:.3f}", "macro F1": "{:.3f}",
                     "onset recall": "{:.3f}", "false alarms": "{:.2%}"}),
                use_container_width=True, hide_index=True, height=320)

        st.subheader("Did the 2024 result hold up in 2025?")
        rep = project_path("reports", "final_results.md")
        if os.path.exists(rep):
            text = open(rep, encoding="utf-8").read()
            if "## 8. Did the 2024 result hold" in text:
                sec = text.split("## 8. Did the 2024 result hold")[1]
                sec = sec.split("## 9.")[0]
                st.markdown(sec[sec.index("|"):])


# ---------------------------------------------------------------------------
# Method
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("How it works")
    st.subheader("The models")
    st.markdown(f"""
Five models each make their own forecast, and a **stacking** meta-model
learns how to combine them:

| model | what it is |
|---|---|
| Random Forest | 300 trees on the 211 engineered features |
| XGBoost | gradient-boosted trees, handles missing values natively |
| LightGBM | gradient-boosted trees, station as a categorical |
| BiLSTM | 2-layer bidirectional LSTM over the last 72 hours |
| CNN-LSTM | convolutions then an LSTM, same 72-hour window |

The two deep models also receive the engineered features alongside the raw
sequence - step 5 found that was the change that mattered.
    """)

    st.subheader("The warning rule")
    st.markdown(f"""
Asking *which class is most likely* is the wrong question for a warning
system; at 24 hours the most likely class is almost never Low, even when
the risk is real. So the rule is:

> **Issue a warning when P(Low) ≥ τ.**

τ was chosen on out-of-fold 2024 data as the threshold catching the most
onsets while keeping false alarms at or below
**{FT['select_false_alarm_rate']:.1%}** - deliberately tighter than the
{FT['target_false_alarm_rate']:.0%} the system promises, because a
threshold tuned to the limit overshoots on unseen data. On 2025 it
delivered what it promised.

On the 2025 test this roughly **doubled** the share of onsets caught
compared with argmax, for about one extra false alarm per hundred safe
hours.
    """)

    st.subheader("Train, validate, test - and the freeze")
    st.markdown("""
| year | role |
|---|---|
| **2023** | training |
| **2024** | every choice: hyperparameters, ensemble, thresholds |
| **2025** | the test, scored **once** |

Before any 2025 target was read, the whole system was written into
`FROZEN.md` - hyperparameters, the number of boosting rounds and epochs,
the ensemble, the thresholds, and a SHA-256 for every model file - and
committed to git. A test only means something if it cannot be repeated
until it passes.
    """)

    st.subheader("Things that were tried and rejected")
    st.markdown("""
- **ATAE** (adaptive threshold-aware ensemble): weights each model by how
  well it has done recently, with errors near the 5 mg/L threshold counting
  for more. On validation it matched a plain average of the same five
  models, so it did not earn its four tuned parameters. *Not adopted.*
- **CAE** (context-aware ensemble): a meta-model that learns which member
  to trust given what the river is doing. It failed its pre-registered
  decision rule outright - worse onset recall and worse RMSE than stacking.
  *Not adopted.*

Both are still scored in the results for the record. Writing the decision
rule down before seeing the numbers is what makes "we rejected it" a
result rather than a preference.
    """)

    st.subheader("Design choices and how they were checked")
    st.caption("Read straight from `reports/final_results.md`, so every "
               "number here is the one in the report. These are additional "
               "analyses: they re-score predictions that already existed, "
               "and none of them was used to choose or change the frozen "
               "system.")
    design = report_section("## Design choices and how they were checked",
                            ["## Scope and future work"])
    if design:
        st.markdown(design)
    else:
        st.info("Run `python -m src.evaluation.step10_checks` and then "
                "`python -m src.evaluation.final_report` to generate this "
                "section.")

    plot = os.path.join(PLOTS, "error_correlation_h24.png")
    if os.path.exists(plot):
        st.image(plot, caption="Correlation of 24 h forecast errors between "
                               "the five members, 2025. Analysis only - not "
                               "used to change the system.",
                 use_container_width=False)

    st.subheader("Scope and future work")
    scope = report_section("## Scope and future work")
    if scope:
        st.markdown(scope)

    with st.expander("The exact frozen configuration"):
        p = project_path(FT["frozen_file"])
        if os.path.exists(p):
            st.markdown(open(p, encoding="utf-8").read()[:6000] + "\n\n*...*")
