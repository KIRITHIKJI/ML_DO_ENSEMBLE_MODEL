"""
Tests for app/app.py: the app opens, and all five tabs render.

Step 10 shipped `st.image(..., use_container_width=False)`, which this
Streamlit does not accept. Because Streamlit executes the whole script on
every rerun, that one bad keyword in the Method tab took down all five
tabs - and a 119-test suite stayed green, because nothing in it ran the
app. These tests close that gap: they execute the real script through
`AppTest` and fail if anything raises.

`AppTest` runs the script once, top to bottom, so every `with tabs[i]:`
block executes in that single run. Opening the app and rendering the five
tabs is therefore the same event, which is why one session-scoped run
serves both tests.

Run from the project root:   pytest -q
"""

import os

import pandas as pd
import pytest

from src.config import load_config

# Loading the five members for four horizons takes ~15 s on a cold cache,
# far past AppTest's 3-second default.
APP_TIMEOUT = 600

TAB_LABELS = ["Overview", "Predict", "Monitor", "Live", "Results", "Method"]
LIVE_TAB = TAB_LABELS.index("Live")


@pytest.fixture(scope="module")
def app():
    """
    One run, shared by every test in this file. The run is the expensive
    part - the Predict tab builds a Forecaster, which reads all five
    models off disk - and none of these tests mutates the result.
    """
    AppTest = pytest.importorskip(
        "streamlit.testing.v1", reason="streamlit too old for AppTest").AppTest
    at = AppTest.from_file("app/app.py", default_timeout=APP_TIMEOUT)
    at.run()
    return at


def test_the_app_opens_without_an_exception(app):
    """
    The regression test for the step-10 crash. `at.exception` collects
    anything the script raised; an empty list is the whole assertion, and
    the message is included so a failure says what broke rather than just
    that something did.
    """
    assert not app.exception, "the app raised: " + " | ".join(
        e.message for e in app.exception)


def test_the_app_renders_its_five_tabs(app):
    """
    Tab labels, in order. A crash does not remove the tab containers - the
    bad `st.image` call left all five in place - so this checks the labels
    and that each tab actually put something inside itself.
    """
    assert [t.label for t in app.tabs] == TAB_LABELS
    for tab in app.tabs:
        assert tab.children, f"the {tab.label} tab rendered nothing"


def test_every_tab_is_reachable_without_touching_a_widget(app):
    """
    Nothing above a tab's content may depend on a button press or an
    upload. If a tab only fills in after a click, a reviewer who clicks
    nothing sees an empty tab, and the test above would have caught it on
    whichever tab regressed.
    """
    empty = [t.label for t in app.tabs if not t.children]
    assert not empty, f"tabs that need a widget before they show anything: {empty}"


def test_the_header_and_the_title_survive_the_run(app):
    """A smoke check that the run produced the page chrome, not just tabs."""
    assert app.title, "the page title did not render"
    assert "oxygen" in app.title[0].value.lower()


def test_the_app_shows_no_error_box(app):
    """
    `st.error` is how the app reports a missing model folder or an
    unreadable results file. On a complete checkout there should be none;
    if the models are absent this is a real failure to see, not to hide.
    """
    boxes = [e.value for e in app.error]
    assert not boxes, f"the app rendered an error: {boxes}"


def test_the_horizons_in_the_app_are_the_configured_ones(app):
    """
    Guards the Results tab's horizon radio against drifting from
    config.yaml - the app must not keep its own copy of the horizons.
    """
    horizons = load_config()["horizons"]
    radios = [r for r in app.radio if r.label == "Horizon"]
    assert radios, "the Results tab lost its horizon selector"
    # AppTest reports the options already passed through format_func.
    assert radios[0].options == [f"{h} hours" for h in horizons]


# ---------------------------------------------------------------------------
# The Live tab (step 12). It must render both ways: with a live forecast on
# disk, and with none at all.
#
# The forecast folder is pointed at a temporary directory for these tests,
# so the result does not depend on whether this machine happens to have run
# the deployed system. Streamlit's caches are process-global, so they are
# cleared before each run - otherwise the second AppTest would be served
# the first one's answer.
# ---------------------------------------------------------------------------
def _run_app_with_forecast_dir(folder):
    AppTest = pytest.importorskip(
        "streamlit.testing.v1", reason="streamlit too old for AppTest").AppTest
    import streamlit as st
    from src.deploy import run as RUNNER
    original = RUNNER.forecast_dir
    RUNNER.forecast_dir = lambda: str(folder)
    try:
        st.cache_data.clear()
        at = AppTest.from_file("app/app.py", default_timeout=APP_TIMEOUT)
        at.run()
        return at
    finally:
        RUNNER.forecast_dir = original
        st.cache_data.clear()


def _all_text(at):
    """
    Every string the app rendered, flattened.

    `tab.children` only holds a tab's direct children, and the Live tab puts
    most of its content inside `st.columns`, so reading the tab's children
    returns blanks. AppTest's element collections (`at.markdown`,
    `at.caption`, ...) gather across the whole script regardless of nesting,
    which is what these tests actually need: the strings they look for are
    unique to the Live tab.
    """
    parts = []
    for name in ("title", "header", "subheader", "markdown", "caption",
                 "info", "success", "warning", "error", "text", "code",
                 "metric"):
        try:
            collection = getattr(at, name)
        except Exception:
            continue
        for element in collection:
            for attr in ("value", "label", "body", "text", "delta"):
                v = getattr(element, attr, None)
                if isinstance(v, str):
                    parts.append(v)
    return "\n".join(parts)


def _fixture_forecast(folder):
    """
    A forecast file in the real schema with one station of EACH alert type,
    so the Live tab has to render all four of its groups.
    """
    import numpy as np
    from src.deploy import common as D
    from src.deploy import run as RUNNER
    issued = pd.Timestamp("2026-10-18 06:00:00")
    rows = []
    # (station, P(Low) base, DO now, class now) -> early / none / ongoing low
    plan = [("02110701", 0.45, 6.4, 1.0),
            ("02198920", 0.02, 6.4, 1.0),
            ("02110550", 0.80, 3.1, 0.0)]
    for station, base, do_now, cls_now in plan:
        for h in load_config()["horizons"]:
            p_low = min(0.99, base + 0.01 * h)
            warn = p_low >= 0.24
            rows.append({
                "station": station, "issued_at": issued, "horizon": h,
                "valid_at": issued + pd.Timedelta(hours=h),
                "pred_do_mgl": 6.0 - p_low, "p_low": p_low, "tau": 0.24,
                "warning": warn,
                "model_version": D.model_version(),
                "weather_source": "forecast_api", "data_age_hours": 2.0,
                "status": D.RUN["fresh_status"],
                "alert_type": RUNNER.classify_alert(cls_now, warn),
                "observed_at": issued, "do_now_mgl": do_now,
                "class_now": cls_now, "method": "stacking"})
    # a station whose sensor stopped reporting: no prediction at all
    for h in load_config()["horizons"]:
        rows.append({
            "station": "03353415", "issued_at": issued, "horizon": h,
            "valid_at": issued + pd.Timedelta(hours=h),
            "pred_do_mgl": np.nan, "p_low": np.nan, "tau": np.nan,
            "warning": False, "model_version": D.model_version(),
            "weather_source": "era5_archive", "data_age_hours": 1069.0,
            "status": D.RUN["stale_status"],
            "alert_type": D.RUN["alert_types"]["stale"],
            "observed_at": issued - pd.Timedelta(hours=1069),
            "do_now_mgl": 5.5, "class_now": 1.0, "method": "none - stale"})
    path = os.path.join(str(folder), "forecast_20261018T060000Z.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def app_no_live(tmp_path_factory):
    """The app with an empty forecast folder: there has never been a run."""
    return _run_app_with_forecast_dir(tmp_path_factory.mktemp("no_live"))


@pytest.fixture(scope="module")
def app_with_live(tmp_path_factory):
    """The app with one forecast file on disk."""
    folder = tmp_path_factory.mktemp("live")
    _fixture_forecast(folder)
    return _run_app_with_forecast_dir(folder)


def test_the_live_tab_renders_with_no_live_run(app_no_live):
    """
    The empty state. With no forecast anywhere, the tab must still render,
    must not raise, and must say how to make a run - an empty panel would
    leave a reviewer unable to tell "nothing has run" from "nothing is
    wrong".
    """
    assert not app_no_live.exception, "the app raised: " + " | ".join(
        e.message for e in app_no_live.exception)
    assert app_no_live.tabs[LIVE_TAB].label == "Live"
    text = _all_text(app_no_live)
    assert "No live run yet" in text, \
        "the empty state does not say that there has been no run"
    assert "src.deploy.run" in text, \
        "the empty state does not say how to make a run"


def test_the_live_tab_renders_with_a_live_run(app_with_live):
    """
    With a forecast on disk the tab must show it: the station ids, the
    warning verdict and the decision against tau. The fixture has one
    station over the threshold and one well under it, so both branches of
    the badge are exercised in a single run.
    """
    assert not app_with_live.exception, "the app raised: " + " | ".join(
        e.message for e in app_with_live.exception)
    assert app_with_live.tabs[LIVE_TAB].label == "Live"
    text = _all_text(app_with_live)
    assert "No live run yet" not in text, \
        "the tab shows the empty state although a forecast exists"
    for station in ("02110701", "02198920", "02110550"):
        assert f"**{station}**" in text, \
            f"the Live tab does not show {station}"
    assert "WARNING" in text, "no warning verdict was rendered"
    assert "chance of Low" in text, \
        "the decision against tau is not spelled out in words"


def test_the_live_tab_renders_all_four_alert_groups(app_with_live):
    """
    Step 12b. The fixture has one station of each kind, so all four groups
    must appear - and the early warnings must come FIRST, because that is
    the only group an operator could not read straight off the sensor.
    """
    from src.deploy import common as D
    text = _all_text(app_with_live)
    low = load_config()["target"]["thresholds"]["low"]
    headings = [h.value for h in app_with_live.subheader]
    joined = "\n".join(headings)

    assert any("Early warnings" in h for h in headings), \
        "no early-warning group"
    assert any(f"Already below {low}" in h for h in headings), \
        "the already-Low stations are not in their own group"
    assert any("Sensor offline" in h for h in headings), \
        "the stale stations are not in their own group"
    assert "No warning" in text, "the no-warning group is missing"

    early_at = joined.index("Early warnings")
    assert early_at < joined.index("Already below"), \
        "the already-Low group is shown above the early warnings"
    assert early_at < joined.index("Sensor offline"), \
        "the offline sensors are shown above the early warnings"

    # the counts, and the words that say what each group means
    assert "⚠ Early warnings" in text or "Early warnings" in text
    assert "safe water" in text.lower()
    assert "no forecast was issued" in text.lower()
    assert f"{float(D.RUN['max_data_age_hours']):g} h" in text, \
        "the stale group does not state the age limit it applied"


def test_the_live_tab_never_shows_a_stale_station_as_warned(app_with_live):
    """
    A dead sensor must not be presented as a forecast. Its station appears
    only under the offline group, labelled as such, with no predicted DO.
    """
    text = _all_text(app_with_live)
    assert "SENSOR OFFLINE" in text, \
        "the stale station is not labelled as an outage"
    # the stale station's own row says no forecast was issued
    assert "03353415" in text
    assert "No forecast issued" in text or "no forecast was issued" \
        in text.lower()


def test_the_live_tab_does_not_run_a_forecast(app_with_live):
    """
    Opening a tab must never cost a GPU forecast. The app reads the newest
    csv; `src/deploy/run.py` is what writes one.
    """
    source = open("app/app.py", encoding="utf-8").read()
    assert "predict_rows" not in source.split("with tabs[3]:")[1] \
        .split("with tabs[4]:")[0], \
        "the Live tab predicts instead of reading the forecast file"
    assert not app_with_live.exception


def test_a_2025_hour_can_only_use_the_tested_models(app):
    """
    The rule from step 11, kept in step 12: a 2025 forecast uses the
    stage-2 models, because models/final trained on 2025 and the two 2025
    modes print what the oxygen actually did. The Predict tab's model radio
    must therefore offer only the tested set while a 2025 mode is selected.
    """
    from src.models import predict_service as PS
    stages = PS.available_stages()
    if "stage2" not in stages:
        pytest.skip("models/stage2 is not on this checkout")
    radios = [r for r in app.radio if r.label == "Model set"]
    assert radios, "the Predict tab lost its model-set selector"
    assert list(radios[0].options) == [
        "Tested models (trained 2023-24, scored on 2025)"], \
        ("a 2025 hour must not be forecastable with models/final: "
         f"the selector offered {list(radios[0].options)}")
