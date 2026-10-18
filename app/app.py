"""
Dissolved-oxygen early warning - the app.

    streamlit run app/app.py

Everything shown here is read from disk: the saved models in models/, the
results in results/, and the reports in reports/. Nothing is trained,
tuned or re-scored, and no number is hard-coded - if a figure appears on a
slide, it came out of a file that the pipeline wrote.

The layout follows reports/ui_audit.md. Each tab has one audience and one
job: Overview answers a reviewer in five seconds, Predict is a live demo
on one station, Monitor is a month of warnings against reality, Results is
the full comparison for someone looking for the hole in it, and Method is
how it works and what was checked.
"""

import base64
import os
import re
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
FINAL = CFG["final_system"]
LOW_CLASS = CLASS_NAMES.index("Low")
HEADLINE_H = 24 if 24 in HORIZONS else HORIZONS[len(HORIZONS) // 2]
RESULTS = project_path(FT["results_dir"])
PLOTS = project_path(FT["plots_dir"])

st.set_page_config(page_title="DO Early Warning", page_icon="💧",
                   layout="wide")


# ---------------------------------------------------------------------------
# Colour
#
# Streamlit 1.33 gives the server no way to know whether the browser is in
# light or dark mode (st.get_option("theme.base") is None unless it is
# pinned in config.toml), so every colour here has to work on both. That
# rules out one thing outright: no colour reaches 4.5:1 against both white
# and #0e1117, so status colour is never carried by text. It is carried by
# chips - white text on a dark fill, which supplies its own background -
# and chart marks are held to the 3:1 that WCAG 1.4.11 asks of graphics.
#
# Measured, white text on the chip fill:
#   Low #5f1210 13.29:1   Medium #7a4a00 7.48:1   Good #15803d 5.02:1
# and their relative luminances are deliberately spread (0.029 / 0.090 /
# 0.159) so the three stay apart in greyscale, where the old
# amber-vs-green pair was identical to within 1%.
#
# Chart marks, minimum of (vs white, vs #0e1117):
#   ink #6b7280 3.91   blue #1f77b4 3.92   amber #d97706 3.19
#   warning red #d62728 3.42
# ---------------------------------------------------------------------------
STATUS = {"Low": ("#5f1210", "▲"),
          "Medium": ("#7a4a00", "●"),
          "Good": ("#15803d", "■")}
WARN_BG, OK_BG = STATUS["Low"][0], STATUS["Good"][0]
INK = "#6b7280"          # observed series, diagram strokes
BLUE = "#1f77b4"         # forecast series
AMBER = "#d97706"        # the Low threshold line
WARN_MARK = "#d62728"    # warning bands and ticks
DEPLOY_TINT = "rgba(31,119,180,0.18)"   # translucent, so it works on both

PRETTY = {"random_forest": "Random Forest", "xgboost": "XGBoost",
          "lightgbm": "LightGBM", "bilstm": "BiLSTM", "cnn_lstm": "CNN-LSTM",
          "stacking": "Stacking ensemble", "simple_average": "Simple average",
          "atae": "ATAE (not adopted)", "cae": "CAE (not adopted)",
          "persistence": "Persistence (no change)",
          "linear": "Linear regression"}


def pretty(method):
    """Readable model names. Names only - never a number."""
    return PRETTY.get(method, str(method).replace("_", " "))


def chip(text, bg, glyph=""):
    """
    A status pill: white text on its own dark fill, so it does not depend
    on the page background, and always carrying the word as well as the
    colour.
    """
    label = f"{glyph} {text}".strip()
    return (f"<span style='display:inline-block;background:{bg};color:#fff;"
            f"padding:2px 9px;border-radius:10px;font-size:13px;"
            f"font-weight:700'>{label}</span>")


def class_chip(class_name):
    bg, glyph = STATUS.get(class_name, (INK, ""))
    return chip(class_name, bg, glyph)


def verdict_badge(warn):
    bg = WARN_BG if warn else OK_BG
    word = "⚠ WARNING" if warn else "✓ NO WARNING"
    return (f"<div style='background:{bg};color:#fff;padding:9px 8px;"
            f"border-radius:7px;font-weight:800;text-align:center;"
            f"font-size:15px'>{word}</div>")


def decision_bar(p_low, tau):
    """
    The decision, drawn instead of left as arithmetic. The bar is P(Low);
    the tick is tau; past the tick is a warning. The tick is white with a
    dark outline so it shows against either fill or either theme, and the
    sentence underneath says the same thing in words - the bar is never
    the only carrier.
    """
    pct = max(0.0, min(1.0, float(p_low))) * 100
    over = tau is not None and float(p_low) >= float(tau)
    fill = WARN_BG if over else OK_BG
    tick = ""
    if tau is not None:
        tp = max(0.0, min(1.0, float(tau))) * 100
        tick = (f"<div style='position:absolute;left:{tp:.1f}%;top:-3px;"
                f"width:2px;height:20px;background:#fff;"
                f"outline:1px solid rgba(0,0,0,.75)'></div>")
    return ("<div style='position:relative;margin:8px 0 4px 0'>"
            "<div style='height:14px;border-radius:7px;"
            "background:rgba(107,114,128,.25);overflow:hidden'>"
            f"<div style='width:{pct:.1f}%;height:100%;background:{fill}'>"
            "</div></div>" + tick + "</div>")


def big_number(value, unit, sub=""):
    """
    The primary number, as markdown rather than st.metric, because
    st.metric truncates its value with an ellipsis in a narrow column and
    this is the one figure that must never be unreadable.
    """
    tail = (f"<div style='font-size:12px;opacity:.75;margin-top:-2px'>{sub}"
            "</div>") if sub else ""
    return (f"<div style='font-size:30px;font-weight:700;line-height:1.1'>"
            f"{value}<span style='font-size:14px;font-weight:600;"
            f"opacity:.8'> {unit}</span></div>{tail}")


def style_layout(fig, height=430, hovermode="x unified", **kw):
    """One layout for every figure, so the charts read as one system."""
    fig.update_layout(height=height, hovermode=hovermode,
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.10, x=0),
                      # Stack the hover toolbar down the right edge; across
                      # the top it lands on the end of the legend.
                      modebar=dict(orientation="v"),
                      **kw)
    return fig


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


@st.cache_data(show_spinner=False)
def frozen_stamp():
    """
    When the system was frozen and at which commit, read out of FROZEN.md
    so the honesty strip cannot drift from the file it describes.
    """
    p = project_path(FT["frozen_file"])
    if not os.path.exists(p):
        return None, None
    head = open(p, encoding="utf-8").read(600)
    m = re.search(r"Written \*\*([^*]+)\*\* at commit `([0-9a-f]{7,40})`", head)
    if not m:
        return None, None
    return m.group(1).strip()[:10], m.group(2)[:12]


@st.cache_data(show_spinner=False)
def design_choice_blocks():
    """
    The report's design-choices section, split on its numbered bold leads
    into (title, body) pairs. Step 10 pasted the whole section into one
    scroll; the content decision was right - the app must not keep its own
    copy of the numbers - but 5,500px of unbroken prose was not, so the
    same text goes into one expander per choice. Splitting it is the only
    thing done to it.
    """
    body = report_section("## Design choices and how they were checked",
                          ["## Scope and future work"])
    if not body:
        return []
    parts = re.split(r"(?m)^\*\*(\d+)\.\s+", body)
    out = []
    for i in range(1, len(parts) - 1, 2):
        num, chunk = parts[i], parts[i + 1]
        m = re.match(r"(.*?)\*\*", chunk, flags=re.S)
        title = m.group(1).strip().rstrip(".") if m else f"choice {num}"
        out.append((f"{num}. {title}", "**" + chunk.strip()))
    return out


def metric_row(reg, cls, method, h, rule="warn"):
    r = reg[(reg["method"] == method) & (reg["horizon"] == h)]
    c = cls[(cls["method"] == method) & (cls["horizon"] == h)
            & (cls["rule"] == rule)]
    return (r.iloc[0] if len(r) else None), (c.iloc[0] if len(c) else None)


def headline_claim(h=HEADLINE_H):
    """
    The project's actual claim, with its four numbers read out of the
    classification metrics. The app states the claim; the csv supplies it.
    """
    if CLS is None:
        return None
    w = CLS[(CLS["method"] == FINAL) & (CLS["horizon"] == h)
            & (CLS["rule"] == "warn")]
    a = CLS[(CLS["method"] == FINAL) & (CLS["horizon"] == h)
            & (CLS["rule"] == "argmax")]
    if not len(w) or not len(a):
        return None
    w, a = w.iloc[0], a.iloc[0]
    return {"warn_onset": float(w["onset_recall"]),
            "argmax_onset": float(a["onset_recall"]),
            "ratio": (float(w["onset_recall"]) / float(a["onset_recall"])
                      if a["onset_recall"] else None),
            "far": float(w["false_alarm_rate"]),
            "budget": float(FT["target_false_alarm_rate"])}


def deployed_table(df, fmt, label_col="method"):
    """
    Every method stays in the table - the baselines and the two rejected
    ensembles included, because leaving them out is how a comparison stops
    being one. The deployed row is marked twice over: a star in its name,
    which survives greyscale and a screen reader, and a translucent tint
    that works on either theme.
    """
    d = df.copy()
    star = (d[label_col] == FINAL).values
    d[label_col] = [("★ " if s else "") + pretty(m)
                    for m, s in zip(d[label_col], star)]
    marked = set(d.loc[star, label_col])
    d = d.rename(columns={label_col: "model"})

    def row_style(row):
        on = row["model"] in marked
        return [f"background-color: {DEPLOY_TINT};font-weight:700" if on
                else "" for _ in row]

    return d.style.apply(row_style, axis=1).format(fmt)


def pipeline_figure():
    """
    The pipeline, as a figure rather than the ASCII block it used to be.
    Fonts are left unset on purpose so Streamlit's plotly theme colours
    them for the active theme - no single hex clears 4.5:1 against both
    backgrounds, so the theme has to do it. The strokes use the one ink
    that clears 3:1 either way. The feature count comes from the ablation
    table rather than the sentence it used to be written into.
    """
    n_feat = int(ABL["n_features"].max()) if ABL is not None and len(ABL) else None
    steps = [
        ("Data sources", "USGS sensors<br>ERA5 weather<br>CAMS air quality"),
        ("Clean to hourly", "ranges, spikes<br>forward-fill &#8804; 3 h"),
        ("Features at t", (f"{n_feat} engineered" if n_feat else "engineered")
         + "<br>lags, rolling stats<br>rain sums, saturation"),
        (f"{len(MEMBERS)} models", "<br>".join(pretty(m) for m in MEMBERS)),
        (pretty(FINAL), "meta-model<br>fitted on 2024"),
        ("Warning rule", "P(Low) &#8805; &#964;"),
        ("Forecast + warning", " &#183; ".join(f"{h} h" for h in HORIZONS)),
    ]
    fig = go.Figure()
    n = len(steps)
    for i, (title, detail) in enumerate(steps):
        fig.add_shape(type="rect", x0=i + 0.05, x1=i + 0.95, y0=0.06, y1=0.94,
                      line=dict(color=INK, width=1.2),
                      fillcolor="rgba(107,114,128,0.10)")
        fig.add_annotation(
            x=i + 0.5, y=0.5, xref="x", yref="y", showarrow=False,
            align="center",
            text=(f"<b>{title}</b><br><span style='font-size:9.5px'>"
                  f"{detail}</span>"))
        if i < n - 1:
            fig.add_annotation(
                x=i + 1.05, y=0.5, ax=i + 0.95, ay=0.5,
                xref="x", yref="y", axref="x", ayref="y",
                text="", showarrow=True, arrowhead=3, arrowsize=1.1,
                arrowwidth=1.6, arrowcolor=INK)
    fig.update_xaxes(visible=False, range=[0, n], fixedrange=True)
    fig.update_yaxes(visible=False, range=[0, 1], fixedrange=True)
    fig.update_layout(height=185, hovermode=False,
                      margin=dict(l=4, r=4, t=4, b=4),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=12), showlegend=False)
    return fig


def image_with_alt(path, alt, width=620):
    """
    st.image leaves the <img> alt as the element's index, which is a WCAG
    1.1.1 failure for a chart that carries real information. Embedding the
    png lets the alt text actually describe it.
    """
    b64 = base64.b64encode(open(path, "rb").read()).decode("ascii")
    st.markdown(
        f"<img src='data:image/png;base64,{b64}' alt='{alt}' "
        f"width='{width}' style='max-width:100%;height:auto'>",
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# The deployed system (step 12). Read-only: the app never runs a live
# forecast, it shows the newest one that `python -m src.deploy.run` wrote.
# Every path and limit comes from config_deploy.yaml.
# ---------------------------------------------------------------------------
from src.deploy import common as DEPLOY                     # noqa: E402

LIVE_DIR = DEPLOY.RUN["results_dir"]
LIVE_BATCH = DEPLOY.RUN["batch_file"]
MONITOR_DAYS = DEPLOY.MON["window_days"]
MONITOR_FA = DEPLOY.MON["max_false_alarm_rate"]


@st.cache_data(show_spinner=False)
def load_live_forecast():
    """
    (frame, path) for the newest forecast file, or None if there has never
    been a run. Reusing the runner's own loader keeps the app and the csv
    schema from drifting apart.
    """
    from src.deploy import run as RUNNER
    df, path = RUNNER.load_latest()
    if df is None:
        return None
    df["warning"] = df["warning"].astype(str).str.lower().isin(
        ["true", "1", "yes"]) if df["warning"].dtype == object \
        else df["warning"].astype(bool)
    return df, path


@st.cache_data(show_spinner=False)
def load_monitor():
    """(monitor.csv, drift.csv), either of which may be None."""
    out = []
    for key in ("out_csv", "drift_csv"):
        p = project_path(DEPLOY.MON[key])
        out.append(pd.read_csv(p, dtype={"station": str})
                   if os.path.exists(p) else None)
    return out[0], out[1]


@st.cache_data(show_spinner=False)
def load_live_skipped():
    """
    The stations the last live fetch skipped, with the reason. Resolved with
    `project_path` rather than `dpath`, because `dpath` creates the folder it
    points at and opening a tab should not write anything.
    """
    p = project_path(DEPLOY.DATA["skipped"].format(mode="live"))
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, dtype={"station": str})


@st.cache_data(show_spinner=False)
def load_model_card():
    p = project_path(DEPLOY.DEP["models"]["model_card"])
    return open(p, encoding="utf-8").read() if os.path.exists(p) else None


MODEL_CARD = load_model_card()

REG = load_csv("metrics_regression_stage2.csv")
CLS = load_csv("metrics_classification_stage2.csv")
ABL = load_csv("ablation.csv")
LEAD = load_csv("lead_times.csv")
STATIONS = load_stations()
PER_STATION = load_csv("per_station_h24.csv")
STATION_LABEL = {f"{r.site_no} - {r['name'][:38]} ({r.state})": r.site_no
                 for _, r in STATIONS.iterrows()}


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("\U0001f4a7 Dissolved-oxygen early warning for US rivers")
st.caption("Forecasting low-oxygen conditions 6 to 48 hours ahead, from "
           "water-quality sensors, weather and air quality. "
           "Tested once on 2025, against a system frozen beforehand.")

tabs = st.tabs(["Overview", "Predict", "Monitor", "Live",
                "Results", "Method"])


# ---------------------------------------------------------------------------
# Overview - for the reviewer with five minutes. The claim and the evidence
# for it come first; the problem statement waits below the fold, which is
# the other way round from how this tab used to be.
# ---------------------------------------------------------------------------
with tabs[0]:
    claim = headline_claim()
    if claim and claim["ratio"]:
        st.markdown(
            f"#### Asking *“is P(Low) above a threshold?”* instead "
            f"of *“which class is most likely?”* catches "
            f"**{claim['ratio']:.1f}× more onsets** — "
            f"{claim['warn_onset']:.1%} against {claim['argmax_onset']:.1%} "
            f"at {HEADLINE_H} hours — while staying inside the "
            f"{claim['budget']:.0%} false-alarm budget at "
            f"**{claim['far']:.2%}**.")

    if REG is not None and CLS is not None:
        r, c = metric_row(REG, CLS, FINAL, HEADLINE_H)
        k = st.columns(5)
        k[0].metric(f"Onsets caught ({HEADLINE_H} h)",
                    f"{c['onset_recall']:.1%}",
                    help="Share of the hours when safe water turns Low that "
                         "the system warned about. The number this project "
                         "exists for.")
        k[1].metric("False alarms", f"{c['false_alarm_rate']:.2%}",
                    f"budget {FT['target_false_alarm_rate']:.0%}",
                    delta_color="off")
        k[2].metric(f"R² ({HEADLINE_H} h)", f"{r['r2']:.3f}")
        k[3].metric("RMSE", f"{r['rmse']:.3f} mg/L",
                    f"{r['skill_vs_persistence']:+.1%} vs no-change")
        if LEAD is not None:
            w = LEAD["warned"]
            k[4].metric("Median lead time",
                        f"{LEAD.loc[w, 'lead_hours'].median():.0f} h",
                        f"{w.mean():.0%} of {len(LEAD)} events warned",
                        delta_color="off")

        left, right = st.columns([3, 2])
        with left:
            if claim and claim["ratio"]:
                st.markdown("**The warning rule against argmax**  \n"
                            f"{pretty(FINAL)}, {HEADLINE_H} hours ahead, "
                            "onsets caught on 2025")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=["argmax<br>most likely class",
                       "warning rule<br>P(Low) ≥ τ"],
                    x=[claim["argmax_onset"], claim["warn_onset"]],
                    orientation="h", marker_color=[INK, OK_BG],
                    text=[f"{claim['argmax_onset']:.1%}",
                          f"{claim['warn_onset']:.1%}"],
                    textposition="outside", showlegend=False,
                    hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
                fig.update_xaxes(tickformat=".0%", range=[
                    0, max(claim["warn_onset"], claim["argmax_onset"]) * 1.3])
                fig.update_layout(height=190, hovermode="closest",
                                  margin=dict(l=4, r=4, t=4, b=4),
                                  yaxis_title=None, xaxis_title=None)
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
        with right:
            froze_when, froze_commit = frozen_stamp()
            froze = (f" Frozen {froze_when}, at commit `{froze_commit}`."
                     if froze_when and froze_commit else "")
            st.info("**How it was tested.** 2023 trains. 2024 decides every "
                    "hyperparameter, the ensemble and the warning "
                    "thresholds. 2025 was scored **once**, after the whole "
                    "system — settings, rounds, and a SHA-256 for every "
                    "model file — had been written into `FROZEN.md` and "
                    f"committed.{froze}")
            if LEAD is not None:
                w = LEAD["warned"]
                st.success(f"**Of the {len(LEAD)} real low-oxygen events in "
                           f"2025, {w.mean():.0%} were warned about in "
                           f"advance**, with a median lead time of "
                           f"{LEAD.loc[w, 'lead_hours'].median():.0f} hours.")
    else:
        st.warning("Run `python -m src.evaluation.test_2025` to populate "
                   "the results.")

    st.divider()
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("The problem")
        st.markdown(f"""
Fish and river life need dissolved oxygen (DO). Below **{LOW} mg/L** water
is stressful and can become lethal; the damage is done by the time anyone
measures it. A forecast a day ahead is the difference between reacting and
preparing - opening a sluice, pausing a discharge, or simply sending
someone to look.

This system forecasts DO at **{", ".join(str(x) for x in HORIZONS[:-1])}
and {HORIZONS[-1]} hours ahead** and raises a warning when the probability
of dropping below {LOW} mg/L is high enough to act on. Above {GOOD} mg/L
the water counts as Good.
        """)
    with c2:
        st.subheader("The data")
        st.markdown(f"""
- **{len(STATIONS)} USGS stations** across {STATIONS['state'].nunique()} US states, hourly, 2023-2025
- **Water sensors**: dissolved oxygen, temperature, conductivity, pH, turbidity, discharge
- **Weather** (ERA5 via Open-Meteo): temperature, humidity, rain, pressure, wind, radiation, soil moisture
- **Air quality** (CAMS via Open-Meteo): PM10, PM2.5, CO, NO₂, SO₂, ozone, aerosol, dust
        """)

    st.subheader("The pipeline")
    st.plotly_chart(pipeline_figure(), use_container_width=True,
                    config={"displayModeBar": False})
    st.caption(f"Data sources to warning. The {len(MEMBERS)} members each "
               f"forecast independently; the {pretty(FINAL).lower()} learns "
               f"how to combine them, and the warning rule turns the "
               f"probability of Low into a yes or a no.")


# ---------------------------------------------------------------------------
# Predict - the live demo. One row of controls, always visible; the model
# folder is an advanced setting rather than the first question anyone is
# asked.
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("Make a forecast")
    stages = PS.available_stages()
    if not stages:
        st.error("No trained models found. Run `python -m src.models.stage2`.")
    else:
        STAGE_NAME = {
            "stage2": "Tested models (trained 2023-24, scored on 2025)",
            "final": "Deployed models (2023-25; for 2026 and live data)"}

        mode = st.radio("What to forecast",
                        ["A real 2025 low-oxygen event",
                         "Any station and hour in 2025",
                         "Upload a window of readings"],
                        horizontal=True)

        # THE RULE (step 11, kept in step 12). A 2025 hour may only be
        # forecast with the tested models: both 2025 modes print what the
        # oxygen actually did next to the forecast, and models/final was
        # trained on that hour, so the comparison would not be a real one.
        # models/final is for 2026 and for live data - that is what the
        # Live tab and the upload mode use.
        on_2025 = not mode.startswith("Upload")
        allowed = (["stage2"] if on_2025 and "stage2" in stages
                   else list(stages))
        default = ("stage2" if on_2025 and "stage2" in stages
                   else ("final" if "final" in stages else stages[0]))
        with st.expander("Advanced — which models to run", expanded=False):
            stage = st.radio("Model set", allowed,
                             index=allowed.index(default),
                             format_func=lambda s: STAGE_NAME.get(s, s))
            st.caption(PS.STAGE_LABEL[stage])
            if on_2025 and "final" in stages:
                st.info("A 2025 hour is forecast with the **tested** models "
                        "only. The deployed models (`models/final`) were "
                        "trained on 2025, so a 2025 forecast from them is "
                        "not out of sample. They are used on 2026 and on "
                        "live data — see the **Live** tab — and are "
                        "available here for an uploaded window.")

        forecaster = get_forecaster(stage)
        result = None
        station = when = None
        go_btn = False

        if mode.startswith("A real"):
            if LEAD is None:
                st.warning("Run `python -m src.evaluation.test_2025` first.")
            else:
                ev = LEAD[LEAD["warned"]].copy()
                ev["onset"] = pd.to_datetime(ev["onset"])
                ev = ev.sort_values("lead_hours", ascending=False).head(30)
                names = STATIONS.set_index("site_no")["name"].to_dict()
                labels = {}
                for _, e in ev.iterrows():
                    nm = str(names.get(e["station"], ""))[:30]
                    labels[f"{e['station']} - {nm} - onset "
                           f"{e['onset']:%Y-%m-%d %H:%M} "
                           f"(warned {e['lead_hours']:.0f} h ahead)"] = (
                               e["station"], e["onset"])
                c1, c2 = st.columns([5, 1])
                with c1:
                    pick = st.selectbox(
                        "A real low-oxygen onset the system warned about",
                        list(labels),
                        help="Real onsets from results/test/lead_times.csv, "
                             "longest warning first. The forecast is made "
                             f"{HEADLINE_H} hours before the onset, so the "
                             f"{HEADLINE_H} h card below is the one aimed "
                             "at it.")
                with c2:
                    st.write("")
                    go_btn = st.button("Forecast", type="primary",
                                       use_container_width=True)
                station, onset = labels[pick]
                when = onset - pd.Timedelta(hours=HEADLINE_H)
                st.caption(f"Forecast made at **{when:%Y-%m-%d %H:%M} UTC**, "
                           f"{HEADLINE_H} hours before a real onset at "
                           f"{onset:%Y-%m-%d %H:%M}.")

        elif mode.startswith("Any station"):
            c1, c2, c3, c4 = st.columns([4, 2, 2, 1])
            with c1:
                pick = st.selectbox("Station", list(STATION_LABEL))
                station = STATION_LABEL[pick]
            with c2:
                day = st.date_input(
                    "Date in 2025", value=pd.Timestamp("2025-07-15").date(),
                    min_value=pd.Timestamp("2025-01-01").date(),
                    max_value=pd.Timestamp("2025-12-31").date())
            with c3:
                hour = st.slider("Hour (UTC)", 0, 23, 12)
            with c4:
                st.write("")
                go_btn = st.button("Forecast", type="primary",
                                   use_container_width=True)
            when = pd.Timestamp(day) + pd.Timedelta(hours=hour)

        if mode.startswith("Upload"):
            c1, c2 = st.columns([2, 3])
            with c1:
                up = st.file_uploader("CSV file", type=["csv"])
                sample = st.selectbox("...or use a sample",
                                      ["-", "before a real low-DO event",
                                       "a healthy window"])
                st_pick = st.selectbox("Station identity (for its embedding)",
                                       ["auto"] + STATIONS["site_no"].tolist())
                go_btn = st.button("Forecast", type="primary",
                                   key="upload_btn")
            with c2:
                st.caption("At least 72 hourly rows; 480 h gives the best "
                           "results because some features look back 7 days. "
                           "Columns must match the template. For a window "
                           "that is genuinely new, switch the model set to "
                           "the all-data models under **Advanced**.")
                with st.expander("Required columns"):
                    st.code(", ".join(PS.template_columns()))
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
                else:
                    with st.status("Running the forecast...",
                                   expanded=False) as s:
                        s.update(label="Building features from the window "
                                       "and running the models...")
                        out, stn, t, do_now = PS.predict_from_hourly(
                            forecaster, df,
                            station=None if st_pick == "auto" else st_pick)
                        s.update(label="Done.", state="complete")
                    result = ({h: {k: (v[0] if isinstance(v, np.ndarray) else v)
                                   for k, v in d.items()
                                   if k not in ("member_reg", "member_p_low",
                                                 "member_proba")}
                               | {"member_reg": {m: float(a[0]) for m, a in
                                                 d["member_reg"].items()},
                                  "member_p_low": {m: float(a[0]) for m, a in
                                                   d["member_p_low"].items()}}
                               for h, d in out.items()}, stn, t, do_now, {})
            elif go_btn:
                st.info("Choose a file or a sample first.")

        elif go_btn and station is not None:
            try:
                with st.status("Running the forecast...", expanded=False) as s:
                    s.update(label=f"Loading the {len(MEMBERS)} members and "
                                   f"the ensemble (a cold start takes a few "
                                   f"seconds)...")
                    res, row = forecaster.predict_station(station, when)
                    s.update(label="Done.", state="complete")
                result = (res, station, when, float(row["do_mgl"].iloc[0]),
                          {h: float(row[f"y_reg_{h}"].iloc[0])
                           if not pd.isna(row[f"y_reg_{h}"].iloc[0]) else None
                           for h in HORIZONS})
            except KeyError:
                st.error(f"No reading for {station} at {when}. That hour may "
                         f"be missing from the sensor record - try another.")

        st.divider()
        if result is None:
            st.markdown(
                "**What you will get.** One card per horizon: the verdict, "
                "the forecast DO value, and the decision itself — "
                "P(Low) against the warning threshold τ — drawn as "
                "a bar, so the comparison that *is* the decision is not left "
                "as arithmetic. Press **Forecast** above.")
        else:
            res, station, when, do_now, truth = result
            st.subheader(f"Station {station} — forecast made at "
                         f"{pd.Timestamp(when):%Y-%m-%d %H:%M} UTC")
            now_class = CLASS_NAMES[int(PS.classify(do_now))]
            st.markdown(
                f"Dissolved oxygen right now: **{do_now:.2f} mg/L** "
                f"{class_chip(now_class)} &nbsp;&nbsp; combined by "
                f"**{pretty(res[HORIZONS[0]]['method'])}**",
                unsafe_allow_html=True)

            cols = st.columns(len(HORIZONS))
            for col, h in zip(cols, HORIZONS):
                d = res[h]
                warn = bool(d["warn"])
                p = np.asarray(d["proba"], dtype=float).ravel()
                p_low = float(p[LOW_CLASS])
                tau = d["tau"]
                with col:
                    st.markdown(f"**{h} hours ahead**")
                    st.markdown(verdict_badge(warn), unsafe_allow_html=True)
                    st.markdown(
                        big_number(f"{float(d['do']):.2f}", "mg/L",
                                   f"{float(d['do']) - do_now:+.2f} vs now"),
                        unsafe_allow_html=True)
                    st.markdown(decision_bar(p_low, tau),
                                unsafe_allow_html=True)
                    if tau is not None:
                        rel = ("at or above" if p_low >= float(tau)
                               else "below")
                        st.markdown(
                            f"<span style='font-size:12.5px'>"
                            f"<b>{p_low:.1%}</b> chance of Low — {rel} "
                            f"the <b>{float(tau):.0%}</b> warning threshold"
                            f"</span>", unsafe_allow_html=True)
                    else:
                        st.caption(f"{p_low:.1%} chance of Low · rule: "
                                   f"most likely class")
                    st.caption(" · ".join(
                        f"{n} {p[i]:.0%}" for i, n in enumerate(CLASS_NAMES)))
                    if truth.get(h) is not None:
                        actual = CLASS_NAMES[int(PS.classify(truth[h]))]
                        st.caption(f"actual was **{truth[h]:.2f} mg/L** "
                                   f"({actual})")
                    else:
                        st.caption("actual: not in the record")

            with st.expander(f"What each of the {len(MEMBERS)} models said"):
                order = list(res[HORIZONS[0]]["member_reg"])
                tbl = pd.DataFrame({
                    "model": [pretty(m) for m in order],
                    **{f"{h} h DO": [res[h]["member_reg"][m] for m in order]
                       for h in HORIZONS},
                    **{f"{h} h P(Low)": [res[h]["member_p_low"][m]
                                         for m in order]
                       for h in HORIZONS}})
                st.dataframe(
                    tbl.style.format({c: "{:.2f}" for c in tbl.columns
                                      if "DO" in c}
                                     | {c: "{:.1%}" for c in tbl.columns
                                        if "P(Low)" in c}),
                    use_container_width=True, hide_index=True)
                st.caption(f"The {pretty(FINAL).lower()} is fitted on 2024, "
                           f"not a simple average, so its answer need not "
                           f"sit between the members'.")


# ---------------------------------------------------------------------------
# Monitor - warnings against reality, one station, one month
# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("Monitor a station through 2025")
    dh = load_pred(FINAL, HEADLINE_H)
    if dh is None:
        st.warning("Run `python -m src.evaluation.test_2025` first.")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            pick = st.selectbox("Station", list(STATION_LABEL),
                                key="mon_station")
        station = STATION_LABEL[pick]
        months = sorted(dh[dh["station"] == station]["time"]
                        .dt.to_period("M").astype(str).unique())
        with c2:
            month = st.selectbox("Month of 2025", months,
                                 index=min(6, len(months) - 1))
        g = dh[(dh["station"] == station)
               & (dh["time"].dt.to_period("M").astype(str) == month)]
        g = g.sort_values("time")

        if not len(g):
            st.info("No rows for that station and month.")
        else:
            # pred_warn is the class the warning rule picked, so a warning
            # is the Low class, not a boolean flag.
            warned = g["pred_warn"] == LOW_CLASS
            fig = go.Figure()
            if warned.any():
                blocks, start = [], None
                times = g["time"].tolist()
                for i, w in enumerate(warned.tolist()):
                    if w and start is None:
                        start = times[i]
                    if (not w or i == len(times) - 1) and start is not None:
                        blocks.append((start, times[i]))
                        start = None
                for a, b in blocks:
                    fig.add_vrect(x0=a, x1=b, fillcolor=WARN_MARK,
                                  opacity=0.25, line_width=0, layer="below")
            fig.add_hline(y=LOW, line_dash="dash", line_color=AMBER,
                          annotation_text=f"Low threshold ({LOW} mg/L)")
            fig.add_trace(go.Scatter(
                x=g["time"], y=g["y_true_reg"], name="DO as it turned out",
                line=dict(color=INK, width=2.0)))
            fig.add_trace(go.Scatter(
                x=g["time"], y=g["pred_reg"],
                name=f"forecast, made {HEADLINE_H} h earlier",
                line=dict(color=BLUE, width=1.6, dash="4px,3px")))
            # The bands say "warning" with colour alone; the triangles say
            # it with a shape and a legend entry too (WCAG 1.4.1).
            if warned.any():
                base = float(np.nanmin([g["y_true_reg"].min(),
                                        g["pred_reg"].min(), LOW])) - 0.3
                fig.add_trace(go.Scatter(
                    x=g.loc[warned, "time"],
                    y=np.full(int(warned.sum()), base), mode="markers",
                    name="warning issued",
                    marker=dict(color=WARN_MARK, symbol="triangle-up",
                                size=6)))
            style_layout(fig, yaxis_title="DO (mg/L)",
                         xaxis_title="time (UTC)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"The grey line is what the oxygen did; the dashed "
                       f"blue line is what the system said {HEADLINE_H} "
                       f"hours earlier. The shaded stripes, and the "
                       f"triangles along the bottom, are the hours it "
                       f"warned about.")

            ok = g["y_true_reg"].notna()
            m = st.columns(4)
            m[0].metric("hours shown", f"{len(g):,}")
            m[1].metric("hours warned", f"{int(warned.sum()):,}")
            m[2].metric("hours truly Low",
                        f"{int((g['y_true_cls'] == LOW_CLASS).sum()):,}")
            if ok.any():
                err = g.loc[ok, "pred_reg"] - g.loc[ok, "y_true_reg"]
                m[3].metric("RMSE this month",
                            f"{np.sqrt((err ** 2).mean()):.3f} mg/L")

            if PER_STATION is not None:
                row = PER_STATION[PER_STATION["station"] == station]
                if len(row):
                    row = row.iloc[0]
                    st.subheader("This station across the whole of 2025")
                    s = st.columns(4)
                    s[0].metric("onsets caught", f"{row['onset_recall']:.1%}",
                                f"{int(row['n_onsets'])} onsets",
                                delta_color="off")
                    s[1].metric("false alarms",
                                f"{row['false_alarm_rate']:.2%}")
                    s[2].metric("macro F1", f"{row['f1_macro']:.3f}")
                    s[3].metric("RMSE", f"{row['rmse']:.3f} mg/L")
                    st.caption(f"{pretty(FINAL)} at {HEADLINE_H} hours, this "
                               f"station only, all {int(row['rows']):,} "
                               f"scored hours of 2025. Stations differ a "
                               f"lot - the table below is all of them.")
                with st.expander("Every station, side by side"):
                    p = PER_STATION.merge(
                        STATIONS[["site_no", "state", "name"]],
                        left_on="station", right_on="site_no", how="left")
                    st.dataframe(
                        p[["station", "state", "name", "rows", "rmse",
                           "f1_macro", "onset_recall", "false_alarm_rate",
                           "n_onsets"]].rename(columns={
                               "rmse": "RMSE", "f1_macro": "macro F1",
                               "onset_recall": "onset recall",
                               "false_alarm_rate": "false alarms",
                               "n_onsets": "onsets"}).style.format(
                                   {"RMSE": "{:.3f}", "macro F1": "{:.3f}",
                                    "onset recall": "{:.3f}",
                                    "false alarms": "{:.2%}"}),
                        use_container_width=True, hide_index=True,
                        height=320)


# ---------------------------------------------------------------------------
# Live - the deployed system as it stands right now (step 12).
#
# Everything here is read from results/live/: the newest forecast file that
# `python -m src.deploy.run` wrote, and the monitoring csvs. Nothing is
# predicted in the app, so opening this tab cannot cost a GPU forecast, and
# nothing is hard-coded - if there has been no run, the tab says so and
# says how to make one.
#
# The models are models/final, which is the rule the Predict tab states:
# 2025 hours use the tested stage-2 models, and models/final is for 2026
# and for live data.
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("Live")
    live = load_live_forecast()
    mon, drift = load_monitor()

    if live is None:
        st.info(
            "**No live run yet.** The deployed system has not produced a "
            "forecast on this machine, so there is nothing to show — this "
            "tab deliberately shows an empty state rather than an old "
            "number dressed up as a current one.")
        st.markdown(f"""
**To make one:**

```bash
python -m src.deploy.run          # fetch the live window, forecast, log it
python -m src.deploy.monitor      # score the forecasts the sensors caught up with
```

or run `{LIVE_BATCH}` once, and schedule it hourly with Windows Task
Scheduler:

```bat
schtasks /create /tn "DO early warning" /tr "%CD%\\{LIVE_BATCH}" ^
         /sc hourly /mo 1 /st 00:05 /rl LIMITED
```

A run writes `{LIVE_DIR}/forecast_<UTC timestamp>.csv`, appends the same
rows to `{LIVE_DIR}/log.parquet`, and this tab reads the newest file.
        """)
        if MODEL_CARD:
            with st.expander("The deployed models (model card)"):
                st.markdown(re.sub(r"(?m)^#", "##", MODEL_CARD))
    else:
        df, live_path = live
        issued = pd.Timestamp(df["issued_at"].max())
        now = pd.Timestamp.utcnow().tz_localize(None)
        age_min = (now - issued).total_seconds() / 60.0
        at_head = df[df["horizon"] == HEADLINE_H]
        warned_stations = sorted(set(df.loc[df["warning"].astype(bool),
                                            "station"]))

        k = st.columns(5)
        k[0].metric("Last run",
                    f"{age_min / 60:.1f} h ago" if age_min >= 90
                    else f"{age_min:.0f} min ago",
                    f"{issued:%Y-%m-%d %H:%M} UTC", delta_color="off")
        k[1].metric("Stations forecast", f"{df['station'].nunique()}",
                    f"{len(HORIZONS)} horizons each", delta_color="off")
        k[2].metric("Under warning now", f"{len(warned_stations)}",
                    f"at any of {', '.join(str(h) for h in HORIZONS)} h",
                    delta_color="off")
        k[3].metric(f"Warned at {HEADLINE_H} h",
                    f"{int(at_head['warning'].astype(bool).sum())}"
                    f" of {len(at_head)}")
        k[4].metric("Oldest reading",
                    f"{df['data_age_hours'].max():.1f} h",
                    "sensor lag", delta_color="off")

        versions = sorted(set(df["model_version"].astype(str)))
        sources = df["weather_source"].value_counts()
        st.caption(
            f"From `{os.path.basename(live_path)}` · models "
            f"`{', '.join(versions)}` · weather "
            + ", ".join(f"{v} rows {k2}" for k2, v in sources.items())
            + (" · forecast-API hours are tagged, so a live forecast is "
               "never mistaken for a reanalysis one"
               if any("forecast" in s for s in sources.index) else ""))
        if age_min > 90:
            st.warning(f"The last run was {age_min / 60:.1f} hours ago. An "
                       f"hourly schedule should keep this under an hour — "
                       f"check the task and `{LIVE_DIR}/run_live.log`.")

        st.subheader("Every station, worst first")
        st.caption("Sorted by the highest chance of Low across the "
                   "horizons. The bar is P(Low); the tick on it is the "
                   "frozen warning threshold τ, so the comparison that *is* "
                   "the decision is drawn rather than left as arithmetic.")
        risk = (df.groupby("station")["p_low"].max()
                .sort_values(ascending=False))
        for station in risk.index:
            g = df[df["station"] == station].sort_values("horizon")
            first = g.iloc[0]
            warn_any = bool(g["warning"].astype(bool).any())
            name = str(STATIONS.set_index("site_no")["name"]
                       .get(station, ""))[:44]
            head = st.columns([3, 1, 1, 1, 1])
            head[0].markdown(f"**{station}** — {name}")
            head[1].markdown(verdict_badge(warn_any), unsafe_allow_html=True)
            now_cls = (CLASS_NAMES[int(PS.classify(first["do_now_mgl"]))]
                       if "do_now_mgl" in g.columns
                       and pd.notna(first["do_now_mgl"]) else None)
            if now_cls:
                head[2].markdown(
                    f"DO now **{first['do_now_mgl']:.2f}** mg/L<br>"
                    + class_chip(now_cls), unsafe_allow_html=True)
            else:
                head[2].markdown("DO now —")
            head[3].markdown(
                f"<span style='font-size:12.5px'>data age "
                f"<b>{first['data_age_hours']:.1f} h</b></span>",
                unsafe_allow_html=True)
            head[4].markdown(
                chip(str(first["weather_source"]).replace("_", " "),
                     BLUE if "era5" in str(first["weather_source"])
                     else AMBER), unsafe_allow_html=True)
            cols = st.columns(len(HORIZONS))
            for col, r in zip(cols, g.itertuples()):
                with col:
                    tau = None if pd.isna(r.tau) else float(r.tau)
                    st.markdown(
                        f"<span style='font-size:12.5px'><b>{r.horizon} h</b>"
                        f" → {pd.Timestamp(r.valid_at):%d %b %H:%M} UTC"
                        f"</span>", unsafe_allow_html=True)
                    st.markdown(big_number(f"{r.pred_do_mgl:.2f}", "mg/L"),
                                unsafe_allow_html=True)
                    st.markdown(decision_bar(r.p_low, tau),
                                unsafe_allow_html=True)
                    rel = ("at or above" if tau is not None
                           and r.p_low >= tau else "below")
                    st.markdown(
                        f"<span style='font-size:12px'><b>{r.p_low:.1%}</b> "
                        f"chance of Low — {rel} τ="
                        f"{'—' if tau is None else format(tau, '.0%')}"
                        f"</span>", unsafe_allow_html=True)
            st.divider()

        with st.expander("The forecast file, as it was written"):
            # shortpath, not os.path.relpath: on Windows relpath raises when
            # the forecast folder is on another drive, and a caption must
            # never be able to take the tab down.
            st.caption(f"`{DEPLOY.shortpath(live_path)}` — "
                       f"the schema every row carries, including the model "
                       f"version id and the weather source.")
            st.dataframe(df, use_container_width=True, hide_index=True)

        skipped = load_live_skipped()
        st.subheader("Stations skipped")
        if skipped is None or not len(skipped):
            st.success("No station was skipped on the last run.")
        else:
            st.warning(f"{len(skipped)} skipped. A station offline, an API "
                       f"down or a missing column never stops a run — it is "
                       f"logged and the other stations carry on.")
            st.dataframe(skipped, use_container_width=True, hide_index=True)

        st.subheader("Monitoring")
        if mon is None or not len(mon):
            st.info("Nothing scored yet. A forecast can only be scored once "
                    "the hour it was about has arrived and the sensor has "
                    "reported it — run `python -m src.deploy.monitor` after "
                    "a few hours of forecasts.")
        else:
            overall = mon[(mon["station"] == "ALL")
                          & (mon["window"] != "all_time")]
            if len(overall):
                st.caption(f"Rolling {MONITOR_DAYS}-day window, all "
                           f"stations, scored against the DO that actually "
                           f"turned up. A false-alarm rate above "
                           f"{MONITOR_FA:.0%} over the window is flagged as "
                           f"performance drift.")
                show = overall[["horizon", "n", "rmse", "f1_macro",
                                "onset_recall", "false_alarm_rate",
                                "n_onsets", "flag_performance_drift"]]
                st.dataframe(
                    show.rename(columns={
                        "n": "rows", "rmse": "RMSE",
                        "f1_macro": "macro F1",
                        "onset_recall": "onset recall",
                        "false_alarm_rate": "false alarms",
                        "n_onsets": "onsets",
                        "flag_performance_drift": "drift?"}).style.format(
                            {"RMSE": "{:.3f}", "macro F1": "{:.3f}",
                             "onset recall": "{:.3f}",
                             "false alarms": "{:.2%}"}, na_rep="—"),
                    use_container_width=True, hide_index=True)
                flagged = overall[overall["flag_performance_drift"]
                                  .astype(bool)]
                if len(flagged):
                    st.error(
                        "**Performance drift.** The false-alarm rate is "
                        "above the promise at "
                        + ", ".join(f"{int(h)} h" for h
                                    in flagged["horizon"])
                        + ". Retraining is gated: a candidate only replaces "
                          "`models/final` if it is at least as good on the "
                          "most recent months it did not train on "
                          "(`python -m src.deploy.retrain --dry-run`).")
                else:
                    st.success("No performance drift: the false-alarm rate "
                               "is inside the promise at every horizon "
                               "scored so far.")
            per = mon[(mon["station"] != "ALL")
                      & (mon["window"] != "all_time")]
            if len(per):
                with st.expander("Per station"):
                    st.dataframe(per, use_container_width=True,
                                 hide_index=True)
        if drift is not None and len(drift):
            st.markdown("**Data drift** — features outside the range they "
                        "had in training.")
            bad = drift[drift["flag_data_drift"].astype(bool)]
            if len(bad):
                st.error("Out of range: "
                         + ", ".join(f"{r.feature} "
                                     f"({r.share_out_of_range:.1%} of rows, "
                                     f"worst {r.worst_station})"
                                     for r in bad.itertuples()))
            else:
                st.success("Every watched feature is inside its training "
                           "range.")
            with st.expander("Every watched feature"):
                st.dataframe(drift, use_container_width=True,
                             hide_index=True)

        if MODEL_CARD:
            with st.expander("The deployed models (model card)"):
                st.markdown(re.sub(r"(?m)^#", "##", MODEL_CARD))


# ---------------------------------------------------------------------------
# Results - the full comparison, for someone looking for the hole in it.
# Every method stays in every table; the deployed one is marked, not
# promoted.
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("Results on the 2025 test year")
    if REG is None:
        st.warning("Run `python -m src.evaluation.test_2025` first.")
    else:
        st.caption("Every number below comes from scoring 2025 once, with a "
                   "system frozen before the test. Nothing here was tuned "
                   "on 2025.")
        h = st.radio("Horizon", HORIZONS, index=HORIZONS.index(HEADLINE_H),
                     horizontal=True, format_func=lambda x: f"{x} hours")

        st.info(f"**How to read these tables.** ★ marks "
                f"{pretty(FINAL).lower()}, the deployed system. Every "
                f"baseline and both rejected ensembles stay in the "
                f"comparison. ATAE and CAE were rejected in advance, on "
                f"2024, by a rule written down before any 2025 number "
                f"existed — so where ATAE edges ahead here, that is a "
                f"result the project reports rather than one it acts on. "
                f"See **Method → Things that were tried and rejected**.")

        st.subheader("Predicting the DO value")
        r = REG[REG["horizon"] == h][["method", "r2", "rmse", "mae", "nse",
                                      "skill_vs_persistence", "danger_rmse"]]
        r = r.rename(columns={
            "r2": "R²", "rmse": "RMSE", "mae": "MAE", "nse": "NSE",
            "skill_vs_persistence": "skill vs no-change",
            "danger_rmse": f"RMSE in {LOW:.0f}-{GOOD:.0f} mg/L"})
        st.dataframe(
            deployed_table(r, {c: "{:.4f}" for c in r.columns
                               if c != "method"}),
            use_container_width=True, hide_index=True)

        st.subheader("Classifying Low / Medium / Good")
        rule = st.radio(
            "Decision rule", ["warn", "argmax"], horizontal=True,
            format_func=lambda x: ("warning rule: P(Low) ≥ τ"
                                   if x == "warn"
                                   else "argmax (most likely class)"))
        c = CLS[(CLS["horizon"] == h) & (CLS["rule"] == rule)][
            ["method", "accuracy", "balanced_accuracy", "f1_macro",
             "roc_auc_ovr_macro", "recall_Low", "onset_recall",
             "false_alarm_rate", "transition_accuracy"]]
        c = c.rename(columns={
            "balanced_accuracy": "balanced acc", "f1_macro": "macro F1",
            "roc_auc_ovr_macro": "ROC-AUC", "recall_Low": "Low recall",
            "onset_recall": "onset recall",
            "false_alarm_rate": "false alarms",
            "transition_accuracy": "transition acc"})
        st.dataframe(
            deployed_table(c, {col: "{:.3f}" for col in c.columns
                               if col not in ("method", "false alarms")}
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
            star = row["method"] == FINAL
            fig.add_trace(go.Scatter(
                x=[row["false_alarm_rate"]], y=[row["onset_recall"]],
                mode="markers", name=pretty(row["method"]),
                marker=dict(size=19 if star else 10,
                            symbol="star" if star else "circle",
                            color=WARN_MARK if star else BLUE,
                            line=dict(width=1, color=INK)),
                hovertemplate=(f"<b>{pretty(row['method'])}</b><br>"
                               "false alarms %{x:.2%}<br>"
                               "onsets caught %{y:.1%}<extra></extra>")))
        dep = cc[cc["method"] == FINAL]
        if len(dep):
            dep = dep.iloc[0]
            # One label, on the one point that matters. The other nine live
            # in the legend and the hover, which is what stopped four of
            # them overprinting each other in the same corner.
            fig.add_annotation(x=dep["false_alarm_rate"],
                               y=dep["onset_recall"],
                               text=f"★ {pretty(FINAL)}",
                               showarrow=True, arrowhead=2, arrowcolor=INK,
                               ax=50, ay=-40, font=dict(size=12))
        fig.add_vline(
            x=FT["target_false_alarm_rate"], line_dash="dash",
            line_color=AMBER,
            annotation_text=f"budget {FT['target_false_alarm_rate']:.0%}")
        style_layout(fig, hovermode="closest",
                     xaxis_title="false alarm rate",
                     yaxis_title="onsets caught")
        fig.update_xaxes(tickformat=".1%")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Up and to the left is better: more onsets caught, fewer "
                   "false alarms. The dashed line is the budget the system "
                   "promised before the test. Hover a point for its numbers; "
                   "the legend names every method.")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Lead time")
            if LEAD is not None:
                w = LEAD["warned"]
                fig = go.Figure(go.Histogram(x=LEAD.loc[w, "lead_hours"],
                                             nbinsx=24, marker_color=BLUE))
                style_layout(fig, height=300, hovermode="closest",
                             xaxis_title="hours of warning before the event",
                             yaxis_title="events")
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"The {int(w.sum())} of {len(LEAD)} real 2025 "
                           f"events that were warned about, by how much "
                           f"warning they got. The other {int((~w).sum())} "
                           f"were missed.")
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
                st.caption(f"**W** water only · **W+M** adds weather "
                           f"· **W+M+A** adds air quality. "
                           f"{pretty('lightgbm')} alone, so the comparison "
                           f"is about the data, not the model.")
            st.caption("Per-station results are on the **Monitor** tab, "
                       "where they sit next to that station's own month.")

        st.subheader("Did the 2024 result hold up in 2025?")
        held = report_section("## 8. Did the 2024 result hold up in 2025?",
                              ["## 9."])
        if held:
            st.markdown(held[held.index("|"):] if "|" in held else held)


# ---------------------------------------------------------------------------
# Method - how it works and what was checked, navigable rather than a wall
# ---------------------------------------------------------------------------
with tabs[5]:
    st.header("How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("The models")
        st.markdown(f"""
{len(MEMBERS)} models each make their own forecast, and a
**{pretty(FINAL).lower()}** learns how to combine them:

| model | what it is |
|---|---|
| {pretty('random_forest')} | 300 trees on the engineered features |
| {pretty('xgboost')} | gradient-boosted trees, handles missing values natively |
| {pretty('lightgbm')} | gradient-boosted trees, station as a categorical |
| {pretty('bilstm')} | 2-layer bidirectional LSTM over the last 72 hours |
| {pretty('cnn_lstm')} | convolutions then an LSTM, same 72-hour window |

The two deep models also receive the engineered features alongside the raw
sequence - step 5 found that was the change that mattered.
        """)
    with c2:
        st.subheader("The warning rule")
        st.markdown(f"""
Asking *which class is most likely* is the wrong question for a warning
system; at {HEADLINE_H} hours the most likely class is almost never Low,
even when the risk is real. So the rule is:

> **Issue a warning when P(Low) ≥ τ.**

τ was chosen on out-of-fold 2024 data as the threshold catching the
most onsets while keeping false alarms at or below
**{FT['select_false_alarm_rate']:.1%}** - deliberately tighter than the
{FT['target_false_alarm_rate']:.0%} the system promises, because a
threshold tuned to the limit overshoots on unseen data. On 2025 it
delivered what it promised.
        """)
    with c3:
        st.subheader("Train, validate, test")
        st.markdown("""
| year | role |
|---|---|
| **2023** | training |
| **2024** | every choice: hyperparameters, ensemble, thresholds |
| **2025** | the test, scored **once** |
        """)
        froze_when, froze_commit = frozen_stamp()
        st.markdown(
            "Before any 2025 target was read, the whole system was written "
            "into `FROZEN.md` - hyperparameters, the number of boosting "
            "rounds and epochs, the ensemble, the thresholds, and a SHA-256 "
            "for every model file - and committed to git."
            + (f" That was {froze_when}, at commit `{froze_commit}`."
               if froze_commit else "")
            + " A test only means something if it cannot be repeated until "
              "it passes.")

    st.subheader("Things that were tried and rejected")
    st.markdown(f"""
- **ATAE** (adaptive threshold-aware ensemble): weights each model by how
  well it has done recently, with errors near the {LOW} mg/L threshold
  counting for more. On validation it matched a plain average of the same
  {len(MEMBERS)} models, so it did not earn its four tuned parameters.
  *Not adopted.*
- **CAE** (context-aware ensemble): a meta-model that learns which member
  to trust given what the river is doing. It failed its pre-registered
  decision rule outright - worse onset recall and worse RMSE than the
  {pretty(FINAL).lower()}. *Not adopted.*

Both are still scored on 2025 for the record, and both stay in the Results
tables. Writing the decision rule down before seeing the numbers is what
makes "we rejected it" a result rather than a preference.
    """)

    st.subheader("Design choices and how they were checked")
    st.caption("Read straight from `reports/final_results.md`, so every "
               "number here is the one in the report. These are additional "
               "analyses: they re-score predictions that already existed, "
               "and none of them was used to choose or change the frozen "
               "system. One expander per choice - open the ones you care "
               "about.")
    blocks = design_choice_blocks()
    if blocks:
        for title, body in blocks:
            with st.expander(title):
                st.markdown(body)
    else:
        st.info("Run `python -m src.evaluation.step10_checks` and then "
                "`python -m src.evaluation.final_report` to generate this "
                "section.")

    plot = os.path.join(PLOTS, f"error_correlation_h{HEADLINE_H}.png")
    if os.path.exists(plot):
        st.subheader("Do the members make the same mistakes?")
        image_with_alt(
            plot,
            f"Heat map of how the {HEADLINE_H} hour forecast errors of the "
            f"{len(MEMBERS)} ensemble members correlate on 2025. The three "
            f"tree models agree with each other most closely, the two deep "
            f"models agree with each other even more closely, and the two "
            f"groups agree with each other least - which is the diversity "
            f"the ensemble is paid for. The exact coefficients are in the "
            f"table under design choice 4 above.")
        st.caption(f"Correlation of {HEADLINE_H} h forecast errors between "
                   f"the {len(MEMBERS)} members, 2025. Analysis only - not "
                   f"used to change the system. The same coefficients are "
                   f"in the table under design choice 4.")

    st.subheader("Scope and future work")
    scope = report_section("## Scope and future work")
    if scope:
        st.markdown(scope)

    with st.expander("The exact frozen configuration"):
        p = project_path(FT["frozen_file"])
        if os.path.exists(p):
            # FROZEN.md opens with its own `# ` title, which would put a
            # second <h1> on the page (WCAG 1.3.1). Quoting it one level
            # down keeps the page's heading order honest; the text is
            # otherwise untouched.
            text = open(p, encoding="utf-8").read()[:6000]
            st.markdown(re.sub(r"(?m)^#", "##", text) + "\n\n*...*")
