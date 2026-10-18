# UI/UX audit of the app - step 11, phase 1

**Scope.** `app/app.py`, the five-tab Streamlit app, as it stands at commit
`91e7ebc`. This is a design document: it changes no model, no threshold and
no number. Every figure quoted here is read from `results/test/*.csv` or
`reports/final_results.md`, the same files the app reads.

**Method.** Three proto-personas and their journeys (assumption-based - see
the warning below), a heuristic critique against Nielsen's ten heuristics
and the UX laws, and a WCAG 2.2 AA accessibility pass with measured
contrast ratios. The app was driven in a real browser at 1366x768 and
900px, in light and dark theme; screenshots are in
`reports/ui_screenshots/`.

> **These are proto-personas, not research.** No real reviewer, operator or
> researcher was interviewed. They are written-down assumptions about the
> project's actual audience, and they are useful only as a shared target to
> design against. Where a claim here rests on an assumption rather than on
> something observed in the app, it is marked **(assumption)**.

---

## 0. One thing had to be fixed before the audit could start

The app crashed on load. `st.image(..., use_container_width=False)` in the
Method tab is not valid in Streamlit 1.33 (the argument was added later),
and because Streamlit runs the whole script on every rerun, the `TypeError`
took down **all five tabs**, not just the Method tab - the Overview showed
a red traceback where the heatmap should be.

That line was introduced by step 10 and committed. The step-10 test suite
passed because nothing in it renders the app. This is the single strongest
argument for the `AppTest` tests planned in phase 2: a 116-test suite that
is all green while the app will not open is not telling the truth about the
app.

Fixed in place (`width=620`) so the audit could proceed. It is the only
code change made in phase 1.

---

## 1. Proto-personas

### P1 - "Five minutes to judge it" (project reviewer / examiner)

| | |
|---|---|
| **Who** | Marks or reviews the project. Technically literate, no domain stake. |
| **Time budget** | 5 minutes, often less. Will look at one or two tabs. |
| **Wants** | To decide: is this real work, honestly evaluated? |
| **Fears** | Being shown a cherry-picked number. Test-set leakage. |
| **Success** | Can state what the system does, how well, and why they believe it. |

**Top tasks**
1. Find the headline result and the year it was measured on.
2. Check the evaluation was not tuned on the test year.
3. See one concrete example of the system working.
4. Find the weaknesses without having to dig for them.

### P2 - "Is there a warning at my station?" (river operator)

| | |
|---|---|
| **Who** | Runs a treatment works, a hatchery or a monitoring programme. Domain expert, not an ML person. |
| **Time budget** | Seconds, repeatedly, possibly on a phone or a narrow window. |
| **Wants** | A yes/no about a specific river, with enough lead time to act. |
| **Fears** | Acting on a false alarm. Missing a real one. |
| **Success** | Knows whether to act, by when, and how confident to be. |

**Top tasks**
1. Pick their station and get a current verdict.
2. Understand how much warning they have.
3. Judge whether this warning is trustworthy (how often is it wrong?).
4. See the recent history of warnings at that station.

### P3 - "Show me the evidence" (researcher / reviewer-in-depth)

| | |
|---|---|
| **Who** | Another modeller, a supervisor, or a reviewer going deep. |
| **Time budget** | 30+ minutes. Will read the report. |
| **Wants** | To find the hole in the evaluation. |
| **Fears** | Unstated choices; numbers that cannot be traced to a file. |
| **Success** | Can reproduce the claim and knows exactly what was and wasn't tested. |

**Top tasks**
1. Compare every method at every horizon, both decision rules.
2. Find the design choices and the checks behind them.
3. Trace a number back to the file it came from.
4. Understand the scope limits and what is future work.

---

## 2. Journeys through the app as it is

Syntax: actions are verbs, thoughts are questions, feelings are adjectives,
pain points are gerunds, opportunities are change verbs.
**MoT** = moment of truth.

### P1, the five-minute reviewer

| Tab | Does | Thinks | Feels | Pain point |
|---|---|---|---|---|
| **Overview** (lands here) | Reads two paragraphs about fish and oxygen | "Fine, but is it any good?" | Patient, then restless | **Reading 180 words of problem statement before seeing a single result** |
| **Overview**, scrolled | Finds the KPI row | "0.951, 0.886 - good at what?" | Reassured but unanchored | **Scrolling past the fold to reach the only thing they came for** |
| **Results** | Sees a 10x7 table of 4-decimal numbers | "Which row is the system? Why is `atae` better?" | Suspicious | **Hunting for the deployed row in an unranked table** |
| Leaves | | "Looks competent. Not sure what it does." | Neutral | |

**MoT: the first screen.** It currently spends its whole budget on the
problem and none on the answer. **(assumption** that reviewers scroll at
all - many will not.**)**

**The `atae` trap is real and specific.** On the Results tab at 24 h, `atae`
shows RMSE 0.6670 against stacking's 0.6718, one row above the deployed
system, with nothing on screen explaining that ATAE was rejected in advance
by a pre-registered rule. A sceptical reviewer reads that as the project
shipping the second-best model and not mentioning it. The report explains
this; the Results tab does not.

### P2, the operator

| Tab | Does | Thinks | Feels | Pain point |
|---|---|---|---|---|
| **Overview** | Skims past research metrics | "None of this is my river." | Impatient | **Wading through model vocabulary to reach a station** |
| **Predict** | Hunts the station dropdown among 5 stacked controls | "Why am I choosing a model folder?" | Confused | **Being asked `models/final` vs `models/stage2` before anything else** |
| **Predict** | Presses Forecast, waits | "Did it work?" | Uncertain | **Waiting through a cold start with only a one-line spinner** |
| **Predict** | Reads `Low: 11.4%` and `threshold τ = 0.24` | "Is 11.4% near the threshold or not?" | **Doing the arithmetic the system should do** |
| **Monitor** | Finds faint pink bands | "Which bits are warnings?" | Squinting | **Reading 13%-opacity bands as the primary signal** |

**MoT: the forecast verdict.** The badge says NO WARNING, but the numbers
underneath invite a second-guess the user cannot resolve - P(Low) and tau
are shown as two unrelated captions four lines apart. The comparison that
*is* the decision is left as homework. This is a Tesler's Law violation:
the complexity has been pushed onto the person.

### P3, the researcher

| Tab | Does | Thinks | Feels | Pain point |
|---|---|---|---|---|
| **Results** | Switches horizon and rule | "Good, both rules are here." | Satisfied | Raw `snake_case` method names |
| **Results** | Reads the onset/false-alarm scatter | "Where is stacking?" | Frustrated | **Deciphering four overlapping labels at the one point that matters** |
| **Method** | Scrolls | "...how long is this?" | Overwhelmed | **Scrolling 5,500px of unbroken report prose with no way to navigate** |
| **Method** | Hits the 15-column pipeline table | "This doesn't fit." | Resigned | **Scrolling a 15-column table sideways inside a page that also scrolls** |

**MoT: the Method tab.** It contains exactly what P3 wants and presents it
as an undifferentiated wall. Step 10 doubled this tab's length by pasting
two report sections in verbatim. The content decision was right - the app
must not keep its own copy of the numbers - but the presentation was not.

---

## 3. Heuristic critique

Severity: 4 = catastrophe, 3 = major, 2 = minor, 1 = cosmetic.

| # | Finding | Heuristic / law | Sev |
|---|---|---|---|
| C1 | `st.image(use_container_width=...)` crashes every tab on Streamlit 1.33 | H1 Visibility, H9 Errors | **4** |
| C2 | Predict: predicted DO truncates to `6.86 …` at 900px - the primary number is unreadable | H1, responsive | **4** |
| C3 | Monitor: observed-DO line hard-coded `black`, invisible in dark theme | H4 Consistency | **4** |
| C4 | Predict: P(Low) and tau shown as separate captions; user must compare them mentally | Tesler's Law, H6 Recognition | **3** |
| C5 | Results: scatter labels collide - `stacking`/`lightgbm`/`xgboost` illegible at the key point | H8, Von Restorff | **3** |
| C6 | Overview: headline results below the fold, behind 180 words of prose | Visual hierarchy, satisficing | **3** |
| C7 | Results: deployed system is an unmarked row; `atae` outranks it with no explanation | H2 Match, Von Restorff | **3** |
| C8 | Method: 5,500px of report prose, no navigation or progressive disclosure | H8 Minimalist, Miller's Law | **3** |
| C9 | Predict: model-folder radio is the first control a user meets | H8, design for the probable | **3** |
| C10 | Monitor: warning bands at 0.13 opacity - the main signal is the faintest mark | Selective attention | **3** |
| C11 | Predict: right-hand 2/3 of the screen is an empty info box until a forecast runs | Blank-slate pattern | **2** |
| C12 | Raw `snake_case` method names throughout (`cnn_lstm`, `simple_average`) | H2 Match real world | **2** |
| C13 | No loading skeleton on first forecast; deep models take tens of seconds cold | Doherty threshold | **2** |
| C14 | Tables: 7 columns at 4 decimals, no emphasis on the metric that matters | H8, scanning | **2** |
| C15 | Four plotly figures, four different ad-hoc palettes and no shared template | H4 Consistency | **2** |
| C16 | Monitor legend "observed DO (at t+24 h)" is ambiguous | H2 Match | **1** |
| C17 | 12 h column shows no "actual was" line while 6/24/48 h do | H4 Consistency | **1** |

---

## 4. Accessibility audit (WCAG 2.2 AA)

Contrast computed from the hex values in `app/app.py`, not estimated.

| Pair | Ratio | Verdict |
|---|---|---|
| white on WARNING red `#d62728` | **5.02:1** | passes AA |
| white on NO-WARNING green `#2ca02c` | **3.40:1** | **fails 1.4.3 AA** (badge text is 14px bold, below the 18.66px large-text exemption) |
| Medium `#ff7f0e` on white | **2.53:1** | **fails 1.4.3 AA** - also the 5 mg/L threshold annotation |
| Good `#2ca02c` on white | 3.40:1 | fails AA for normal text |
| Low `#d62728` on white | 5.02:1 | passes |
| plotly blue `#1f77b4` on white | 4.82:1 | passes |

| ID | Finding | WCAG | Level | Sev |
|---|---|---|---|---|
| A1 | NO-WARNING badge 3.40:1 | 1.4.3 Contrast (Minimum) | AA | **High** |
| A2 | Threshold annotation + Medium class at 2.53:1 | 1.4.3 | AA | **High** |
| A3 | Heatmap image carries `alt="0"` - Streamlit's default index, not a description | 1.1.1 Non-text Content | A | **High** |
| A4 | Monitor warning bands convey warning by colour alone (the caption explains them, but nothing in the chart distinguishes them non-visually) | 1.4.1 Use of Colour | A | **Moderate** |
| A5 | Status trio relies on red/amber/green - the classic deuteranopia triple. Luminance between Medium and Good differs by only **1.34:1**, so they are near-identical in greyscale | 1.4.1 | A | **Moderate** |
| A6 | Two `<h1>` elements on the page | 1.3.1 Info and Relationships | A | **Low** |
| A7 | No `<main>` landmark (Streamlit framework-level) | 1.3.1 | A | **Low** |

**Already done right, and worth keeping:** the warning badges pair colour
with the words WARNING / NO WARNING and a glyph, so the badge itself is not
colour-only. The redesign must preserve that pairing everywhere, not just
on the badge.

---

## 5. Strategy: what each tab is for

**The 5-second message of the first screen** must be the project's actual
claim, and the brief names it:

> **Asking "is P(Low) above a threshold?" instead of "which class is most
> likely?" catches 1.7x more onsets - 49.6% against 28.5% - while staying
> inside the 2% false-alarm budget at 1.50%.**

Those four numbers all come from
`results/test/metrics_classification_stage2.csv`; none is hard-coded.

| Tab | Is FOR | Is NOT for | Primary persona |
|---|---|---|---|
| **Overview** | The claim, the evidence for it in five numbers, and the honesty frame | Explaining dissolved-oxygen chemistry | P1 |
| **Predict** | One station, one verdict, with the P(Low)-vs-tau decision shown | Model-folder administration | P2 |
| **Monitor** | Warnings against reality over a month at one station | A second results table | P2 |
| **Results** | The full comparison, both rules, every horizon | The first thing anyone sees | P3 |
| **Method** | How it works and what was checked, navigable | A verbatim report dump | P1 skim / P3 deep |

### Cut, merge, move

| Action | What | Why |
|---|---|---|
| **Move** | Headline KPIs above the problem statement | P1 spends the whole budget before reaching them |
| **Move** | Model-folder choice into an "Advanced" expander, default `final` | Not a first-run decision for P2 |
| **Cut** | ASCII pipeline diagram from the first screen | Costs a third of the viewport, serves nobody's top task |
| **Cut** | Duplicate lead-time block (Overview and Results both show it) | Say it once, on Overview |
| **Merge** | Per-station table into Monitor | It is station context, and Results is already long |
| **Add** | A one-line "why not ATAE" note next to the Results table | Closes the credibility gap; the text exists in the report |
| **Add** | P(Low) vs tau bar on every Predict horizon | The decision the user is currently making by hand |
| **Keep** | Report-driven Method content | Right call; needs expanders, not a rewrite |

---

## 6. Proposed layout per tab

### Overview
```
[ headline claim, one sentence, large ]
[ KPI row: onsets caught - false alarms - R2 - RMSE - lead time ]
[ rule-vs-argmax comparison, the 1.7x, as a small paired bar  ]
[ honesty strip: train 2023 / choose 2024 / test 2025 once, frozen at <commit> ]
---------------- below the fold ----------------
[ the problem, 2 short paragraphs ] [ the data, 3 bullets ]
```

### Predict
```
[ station + when ........................ ] [ Forecast ]   <- one row, always visible
[ Advanced: which models (expander, collapsed) ]
[ now: DO x.xx mg/L, class ]
[ 6h ][ 12h ][ 24h ][ 48h ]   each a card:
         VERDICT badge (colour + word + glyph)
         predicted DO, large, never truncated
         P(Low) ----|----------- tau        <- the decision, as a bar
         plain-language line: "12% chance of Low, below the 24% threshold"
[ what each model said (expander, collapsed) ]
```

### Monitor
```
[ station ] [ month ]
[ chart: observed vs forecast, 5 mg/L line always marked,
          warning bands at readable opacity + hatch, theme-aware colours ]
[ hours shown | hours warned | hours truly Low | RMSE this month ]
[ this station's 2025 scorecard  <- moved from Results ]
```

### Results
```
[ horizon ] [ decision rule ]        <- unchanged, they work
[ the deployed system highlighted in every table, human-readable names ]
[ onset-vs-false-alarm chart with non-colliding labels + budget line ]
[ note: ATAE/CAE were rejected in advance - one line, links to Method ]
[ lead time | data-source ablation ]
```

### Method
```
[ how it works: models, the rule, the freeze - 3 compact cards ]
[ Design choices and how they were checked  -> 6 expanders, one per choice,
  content read from the report verbatim, collapsed by default ]
[ error-correlation heatmap, with a real alt description ]
[ Scope and future work ]
[ the frozen configuration (expander) ]
```

---

## 7. What phase 2 must not break

- Every number still read from `results/` or `reports/`; no metric literal
  in `app/app.py`.
- `src/models/predict_service.py` untouched.
- Both Predict input modes, the Monitor selectors, the Results horizon and
  rule toggles, and the Method report sections all keep working.
- `FROZEN.md` hashes still verify.

---

# Phase 2 - the redesign

Scope: `app/app.py` only. No model, threshold, metric or report changed.
`FROZEN.md`, `config.yaml` and `src/` are untouched, and the suite that
verifies the frozen hashes still passes.

## 8. What was done, finding by finding

| # | Finding | What changed |
|---|---|---|
| C1 | `st.image` crashed every tab | Fixed and committed on its own, with `tests/test_app.py` - an `AppTest` run of the real script - so a crash cannot pass the suite again. The `st.image` call is gone entirely; the heatmap is now an `<img>` with a written alt (A3). |
| C2 | Predicted DO truncated at 900px | The primary number is markdown at 30px, not `st.metric`, so it cannot ellipsize. The result area is also full width now rather than two thirds. Measured at 900px: all four values render unclipped. |
| C3 | Observed line hard-coded `black` | Theme-neutral ink `#6b7280`, verified against the dark theme in the browser. |
| C4 | P(Low) and tau left as mental arithmetic | A bar per horizon: fill is P(Low), the tick is tau, past the tick is a warning - plus a sentence saying the same thing in words. |
| C5 | Scatter labels collided | One annotation, on the deployed point only. Every other method is named in the legend and the hover. |
| C6 | Headline results below the fold | Overview now opens with the claim sentence and the five KPIs; the problem statement and the data moved below the divider. At 1366x768 the claim, all five KPIs, the rule-vs-argmax bar and the honesty strip are on the first screen. |
| C7 | Deployed row unmarked, `atae` above it | Every method stays in every table. The deployed row carries a star in its name and a translucent tint, and a note above the tables says ATAE and CAE were rejected in advance by a pre-registered rule. Nothing is hidden or dropped. |
| C8 | Method tab a 5,500px wall | The report's design-choices section is split on its numbered leads into six expanders, still verbatim. |
| C9 | Model folder the first control | Moved into a collapsed **Advanced** expander. |
| C10 | Warning bands at 0.13 opacity | 0.25, plus a row of triangles with its own legend entry (A4). |
| C11 | Empty right-hand panel before a forecast | Replaced with a short "what you will get", and the layout no longer reserves two thirds of the screen for nothing. |
| C12 | Raw `snake_case` names | One `pretty()` map, used in tables, charts, hovers and prose. Names only - it never supplies a number. |
| C13 | No feedback on a cold start | `st.status` with a staged label. |
| C14 | Tables undifferentiated | Human-readable headers, the deployed row marked. |
| C15 | Four ad-hoc palettes | One `style_layout()` and one palette for every figure. |
| C16 | Ambiguous Monitor legend | "DO as it turned out" against "forecast, made 24 h earlier". |
| C17 | 12 h column missing "actual was" | Every horizon now prints an actual, or says it is not in the record. |

**Also added, not in the phase-1 list:** a *"A real 2025 low-oxygen
event"* mode on Predict, built from `results/test/lead_times.csv` - it
picks a real onset the system warned about and makes the forecast
`24 h` before it, so a demo lands on something that matters instead of an
arbitrary hour.

### The model-set default was changed

Phase 1 left this open. The default is now **stage2**, the tested models,
not `final`.

Both 2025 modes print what the oxygen actually did next to the forecast.
Only the tested models have never seen that hour, so only with them is
that comparison a real one; the all-data models were trained on 2025, and
showing their "forecast" beside the 2025 truth invites exactly the
leakage question P1 is already looking for. The all-data models are one
click away under **Advanced**, where choosing them raises a warning
saying what they can and cannot be used for, and the upload mode - the
one case that is genuinely new data - points at them.

### Colour

No colour reaches 4.5:1 against both `#ffffff` and `#0e1117`, and
Streamlit 1.33 gives the server no way to know which one it is facing
(`st.get_option("theme.base")` is `None`). So status is never carried by
coloured text. It is carried by chips - white text on a fill that comes
with them - and chart marks are held to the 3:1 of WCAG 1.4.11.

| role | colour | white text on it | in charts, min(vs light, vs dark) |
|---|---|---|---|
| Low / WARNING | `#5f1210` | **13.29:1** | - |
| Medium | `#7a4a00` | **7.48:1** | - |
| Good / NO WARNING | `#15803d` | **5.02:1** | - |
| observed series, diagram | `#6b7280` | - | **3.91:1** |
| forecast series | `#1f77b4` | - | **3.92:1** |
| Low threshold line | `#d97706` | - | **3.19:1** |
| warning bands and ticks | `#d62728` | - | **3.42:1** |

The three status fills also have deliberately spread luminance (0.029 /
0.090 / 0.159), so they stay apart in greyscale - where the old amber and
green were identical to within 1%.

---

# Phase 3 - verification

The app was driven in a real browser again, light and dark, at 1366x768
and 900px. Screenshots are `reports/ui_screenshots/p3_*.png`.

## 9. Measured after the change

Contrast read off the live DOM with `getComputedStyle`, not from the
source:

| element | size | ratio | 1.4.3 AA |
|---|---|---|---|
| `⚠ WARNING` badge | 15px 800 | **13.29:1** | passes |
| `✓ NO WARNING` badge | 15px 800 | **5.02:1** | passes (was 3.40, failed) |
| `● Medium` chip | 13px 700 | **7.48:1** | passes (was 2.53, failed) |

| check | result |
|---|---|
| Script raises on load | none - `AppTest` run, 6 tests |
| `<h1>` elements | **1** (was 2; `FROZEN.md`'s own title is quoted a level down) |
| `<img>` without a real alt | **0** (was 1, `alt="0"`) |
| Predicted DO clipped at 900px | **0 of 4** (was truncating) |
| Decision bars rendered | 4 of 4 horizons |
| Horizontal page scroll at 900px | none |
| Browser console errors | none |
| Full test suite | **125 passed** (119 before, +6) |
| `FROZEN.md` hashes | verify (`tests/test_final.py`) |
| Metric literals in `app/app.py` | none |
| `src/models/predict_service.py` | untouched |

Two defects were found during phase 3 itself and fixed: the shared
`style_layout()` took `hovermode` twice and crashed the app - caught by
the new `AppTest` test on its first run, which is the point of it - and
the plotly hover toolbar sat on top of the scatter legend, so the toolbar
is now stacked down the right edge.

## 10. What is still open

- **A5, partly.** The three status fills are now 1.78 and 1.49 apart in
  greyscale rather than identical, but 4.5:1 with white text caps how
  light the Good fill can be, so the gap cannot widen further without
  failing 1.4.3. Every status is therefore also written as a word and a
  glyph, which is what 1.4.1 actually asks for.
- **A7.** No `<main>` landmark. This is Streamlit's own DOM; the app
  cannot set it.
- **The personas are still proto-personas.** Nothing in phase 2 or 3
  tested the design with a real reviewer, operator or researcher. The
  layout is an argued guess at what P1 needs in five minutes, not a
  measured one.
