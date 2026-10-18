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

import pytest

from src.config import load_config

# Loading the five members for four horizons takes ~15 s on a cold cache,
# far past AppTest's 3-second default.
APP_TIMEOUT = 600

TAB_LABELS = ["Overview", "Predict", "Monitor", "Results", "Method"]


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
