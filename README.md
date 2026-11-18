# F1 Pit-Stop Strategy Simulator

Live site: https://junyanguo02.github.io/f1a-strategy-sim/

**Do F1 teams stop too often under green flag?** A Bayesian tyre-degradation model, a single- and multi-car Monte Carlo race simulator, and a teammate natural experiment, built on public lap-by-lap data from the 2023–2026 seasons.

> Status: research preview, November 2026 (package version 0.1.0). Author: Junyan Guo. Code: MIT. Data: see [Data and licences](#data-and-licences).

## Key findings

| | Result |
|---|---|
| Teammate natural experiment (82 pairs, dry races) | The car that made **one extra stop under green flag finished on average 16.4 s behind its teammate** (95% race-clustered bootstrap CI 8.8–25.0 s; within-race permutation p ≈ 0.0005). |
| After removing "forced" extra stops (tyre wear, slow stops) and adjusting for grid and first-stint pace | Still **+10.3 s** (cluster SE 3.6, p = 0.006). |
| Extra stop under Safety Car / VSC (26 pairs) | **−11.6 s** on average — the "cheap stop" effect points the other way (point estimate only: 95% CI −28.4 to +3.3 s). |
| Simulator back-test vs teams' actual choices | Pre-race mode agrees with the actual number of stops **47%** of the time; when it disagrees, it recommends **fewer stops in 84%** of cases (exactly one fewer in 64%). Robust to pit-loss data cleaning: 79.9% (pre-race) and 93.1% (post-race) after removing wet-weather changes, timing anomalies and penalties. |
| Explanations tested inside the model and not supported | Tyre "cliff", circuit-specific Safety Car rates, traffic/overtaking/undercut, deliberate tyre management on long stints, stint-to-stint degradation differences. |

These are descriptive results. These tests are run inside the model and do not prove causes in real races. Stop counts are chosen by teams, not randomised; team orders, track-position defence and risk preferences are not observable in public data.

## What is inside

- **Data pipeline** — OpenF1 laps, stints, pit stops and race-control messages; Jolpica grids and results; reconciliation and clean-lap rules (81 dry races, 77,652 clean laps, 3,957 stints).
- **Tyre model v3.1** — hierarchical Bayesian lap-time model (NumPyro), fitted per season: fuel effect, compound offset and linear + quadratic degradation per race, traffic penalty, per-stint random intercept, Student-t errors.
- **Race inputs** — pit loss relative to same-lap cars (green 22.8 s, VSC 20.6 s, SC 19.3 s; under SC by time since deployment), Safety Car / VSC hazard and duration, overtaking probability by pace difference, start effect by compound, SC restart spacing.
- **Simulator** (`src/f1a/sim/`)
  - single car: total-time distribution per strategy, posterior parameter uncertainty, grid search over pit laps;
  - multi car: full field with traffic, overtaking, pit-lane rejoin (undercut emerges naturally) and SC compression;
  - common random numbers, so strategies are compared by paired differences;
  - **pre-race mode**: only information available before the race (previous-season tyre posterior, last three races' driver pace, pit-loss and SC history up to the race date).
- **Interactive app** (`app/`) — pick a race, edit strategies, compare paired time differences, switch Safety Car scenarios and pre-race / post-race mode.

## Install

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[app]'        # simulator + app
python -m pip install -e '.[app,model]'  # also the NumPyro tyre model
python -m pytest                         # tests that need the data are skipped until it is built
```

## Reproduce

Raw data and fitted posteriors are not distributed. Rebuild them with [`scripts/download_data.md`](scripts/download_data.md):

1. Download and reconcile OpenF1 / Jolpica data, build the model tables.
2. Fit the tyre model per season: `models/tyre_v3/fit_tyre_v3_1.py` (about 30–45 min per season on a laptop CPU, 4 chains).
3. Back-tests: `scripts/backtest.py` (single car) and `scripts/backtest_field.py` (multi car); add `--prerace` for pre-race mode.
4. App: `streamlit run app/strategy_app.py`.

Every analysis script runs with one command and a fixed seed. Scripts with a two-run manifest were run twice and produced byte-identical outputs on the same inputs and environment.

## Repository layout

| Path | Contents |
|---|---|
| `src/f1a/` | data pipeline (`data.py`, `openf1.py`, `clean.py`, `pit.py`, `pitloss.py`, `fcy.py`) |
| `src/f1a/sim/` | simulator: parameters, race configuration, strategies, single-car engine, multi-car field, optimiser, back-test, pre-race mode |
| `models/tyre_v3/` | tyre model v3.1 and its notes |
| `analysis/` | scripts that build the race inputs and the strategy datasets |
| `analysis/pitloss_audit/` | pit-stop classifications, cleaning rules and pool statistics |
| `analysis/pitloss_sensitivity/` | pre-race and post-race back-test results across three pit-loss pool definitions |
| `scripts/` | data build, back-tests, demo |
| `app/` | Streamlit app |
| `tests/` | unit and integration tests |

## Limitations

- Pit-loss samples were audited (analysis/pitloss_audit/); the default simulator uses the original pools for reproducibility, and `pit_loss_filter="clean"` reproduces the robustness check (analysis/pitloss_sensitivity/).
- Pre-race mode uses last season's tyre parameters; tyre behaviour can change between seasons (e.g. the posterior mean soft-tyre degradation slope at Barcelona is about 0.10 s/lap in 2024 vs 0.06 s/lap in 2023), and pre-race mode cannot see such changes.
- Overtaking, start and SC-restart constants are calibrated on the full sample (sensitivity: 15 of 520 recommendations change under ±0.5 on the overtaking intercept or 0.5 / 1.2 s restart spacing).
- The model tables cover dry races only. The optimiser searches one- and two-stop plans; actual strategies with three or more stops (71 of 520 back-tested car-races) are still scored against it.
- Multi-car opponents in pre-race mode all follow the simulator's own recommended strategy.

## Data and licences

The website JSON contains derived summaries of public OpenF1 and Jolpica data for non-commercial research demonstration.

- [OpenF1](https://openf1.org/) — historical data for non-commercial use; check their terms before any commercial use.
- [Jolpica F1](https://github.com/jolpica/jolpica-f1) — calendars, grids and results.
- This repository's code is MIT-licensed; data rights remain with their providers. This project is unofficial and not associated with Formula 1 or any team.

## Citation

See [`CITATION.cff`](CITATION.cff).
