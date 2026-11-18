# Do F1 teams pit too often under green flag?

A race-strategy simulator and a teammate natural experiment, 2023–2026 · Junyan Guo · November 2026

## Summary

Two independent lines of evidence point the same way: in dry races from 2023 to 2026, an extra pit stop under green flag cost the driver about 10–16 seconds on average.

- **The simulator.** I fitted a Bayesian tyre-and-fuel model to 81 dry races and used it to simulate thousands of races per strategy. When its recommended number of stops differs from what the team actually did, it recommends *fewer* stops in 84% of cases using only pre-race information (94% with post-race information). I tested five plausible flaws in the model that could make it under-stop; none of them explains the gap.
- **Teammates.** When two drivers of the same team finish the same race with different numbers of stops, the one who stopped once more under green flag finished on average **16.4 s** behind (95% CI 8.8–25.0 s, 82 pairs). After removing forced stops and adjusting for grid position and first-stint pace, the gap is still **10.3 s**.

This does not prove teams were wrong: stop counts are chosen, not randomised, and track position, rivals and risk appetite are not in public data. But it does say that in most races an extra green-flag stop is hard to pay back with fresh-tyre pace. Two case studies at the end — Spain and Australia 2024 — show one race where the simulator agrees with the teams and one where it does not.

## The question

A pit stop under green flag costs roughly 23 seconds against cars that stay out. What it buys is a fresh set of tyres. Whether one more stop is worth it depends on whether those tyres can win the 23 seconds back over the remaining laps.

Teams make this call every race, and it is the call strategists are most often second-guessed on. The question here is narrow: **looking only at lap times and tyre physics, do teams make too many stops under green flag?**

I answer it two ways. The first is a strategy simulator: where does its advice differ from what teams actually did? The second does not use the model at all: when teammates split on stop count in the same race, who got to the flag first?

## Data and model

The data come from OpenF1 (lap times, tyre stints, pit stops, race-control messages) and Jolpica (grids and results) for 90 Grands Prix from 2023 to 2026. After removing 8 wet races and one race with incomplete data, the model table has 81 races, 77,652 clean laps and 3,957 tyre stints.

**Lap-time model.** A hierarchical Bayesian model (NumPyro, fitted separately for each season):

```
t_l = alpha[r,d] + k[r,c] + gamma[r] * rem_l + beta[r,c] * (a_l - 10) + kappa[c] * (a_l - 10)^2 + delta_gap + u_s + eps_l
```

Here r is the race, d the driver, c the compound, a the tyre age, rem the laps remaining (a fuel proxy), u a random intercept per stint, and the error is Student-t. All four seasons converged. Typical estimates: fuel about 0.06 s per lap; tyre degradation (median across races) 0.044 s/lap on the hard, 0.059 on the medium and 0.066 on the soft; running within one second of the car ahead costs about 0.3 s per lap.

**Pit loss** is measured relative to cars on the same lap: 22.8 s under green flag, 20.6 s under VSC and 19.3 s under Safety Car (medians). Under a Safety Car the timing matters: stopping between 60 seconds and one lap after deployment is cheapest (median 14.5 s), while stopping just before it is called costs about as much as a green-flag stop (21.9 s). One detail I liked: the time spent in the pit lane is about 23 s in all three conditions — everything a Safety Car saves comes from the rest of the field being slowed down.

**Safety Cars.** Across the 90 races there were 117 neutralisations (60 SC, 57 VSC). The probability of a Safety Car on lap 1 is about 13%; after that it is about 0.6–1.3% per lap.

**A physics check on the fuel term.** I rebuilt a point-mass lap-time simulator on the real centrelines of five 2024 circuits (extracted from OpenF1 position data) and calibrated it to measured speed traces. Its mass sensitivity is 0.017–0.030 s per kg per lap, which with 105–110 kg of fuel gives a fuel effect of about 0.034–0.049 s per lap across calibration variants (0.035–0.040 for the main fit). The statistical term above is larger for the same races (0.048–0.070 s/lap), but ranks the circuits in the same order when grip may differ by circuit (Spearman 0.9–1.0). A rough estimate of track evolution — laps getting faster as rubber goes down, at equal tyre age and compound — accounts for most of the gap on average (about 0.018 of 0.021 s/lap). So the model's "fuel" term is better read as fuel plus track evolution. Because it applies equally to every strategy in a paired comparison, it does not drive the stop-count results. The lap simulator is a point mass with effective parameters (its fitted drag area is implausibly high), so it is a cross-check, not a vehicle model.

## The simulator

The simulator assembles these pieces into a full race and runs each strategy thousands of times, returning a distribution of total race time or finishing position.

- **Single car.** One car on an empty track; strategies are compared on total time. Parameters are drawn from the posterior, so model uncertainty flows into the result.
- **Multi car.** The whole field runs together. Each car has its own pace and grid slot; following costs time, overtaking probability depends on the pace difference, and a car that cannot pass is held up. After a stop, cars rejoin by cumulative time, so the undercut emerges naturally. Safety Car restarts compress the field to about 0.8 s gaps. Overtaking, start and restart parameters are calibrated on real races.
- **Fair comparison.** All candidate strategies share the same random numbers — the same Safety Car timeline, the same noise — so differences are measured as paired differences rather than as the gap between two wide distributions.
- **Pre-race mode.** Only information available before the race: last season's tyre parameters, the driver's pace over the last three races, and pit-loss and Safety Car history up to race day. Opponents follow the simulator's own recommended plan. Back-test numbers quoted below use pre-race mode unless marked post-race.

## Evidence 1: the simulator keeps stopping less

For every top-ten car in every race, I let the simulator choose the number of stops (other cars' strategies fixed) and compared it with what that car actually did.

| Mode | Car-races | Same number of stops | When different: simulator stops less |
| --- | --- | --- | --- |
| Pre-race (2024–2026) | 520 | 47% | 84% (exactly one fewer: 64%) |
| Post-race (2023–2026) | 727 | 62% | 94% |

The usual reading is that the model is missing something. I tested five flaws that could make it under-value a second stop. Each was checked inside the model and none explains the gap (these are tests within the model, not proof about causes in real races):

| Hypothesis | If true, we should see | What we see |
| --- | --- | --- |
| Tyre "cliff" | Faster drop-off late in long stints | Late-stint slopes are lower, not higher (3,448 stints) |
| Circuit-specific Safety Car rates | Recommendations change once rates are circuit-specific | 2024–2025 agreement unchanged |
| Traffic, overtaking, undercut | Multi-car version much closer to reality | Only +2.75 percentage points |
| Tyre management on long stints | Cars planning long stints are slower from the start | At equal tyre age, long stints are faster |
| Deciding as you go | Cars with faster degradation stop more often | Stint-to-stint variation is large (SD 0.08 s/lap) but fast-degrading cars do not stop more |

**Robustness to pit-loss data.** An audit found that the empirical pit-loss pool contained records that are not normal pit-stop costs — wet-weather tyre changes, timing anomalies, penalties — which pushed the mean green-flag loss from 23.5 s up to 25.2 s and, in principle, nudge the simulator towards fewer stops. Re-running the back-tests with the cleaned pool, the share of disagreements where the simulator stops less moves from 84.0% to 79.9% (pre-race) and from 93.5% to 93.1% (post-race). The conclusion stands.

With all five explanations failing, another possibility comes into view: the simulator is right, and teams really do stop too often under green flag. Testing that needs evidence that does not come from the same model.

## Evidence 2: a teammate natural experiment

Two teammates driving the same car in the same race are the closest thing to a control group that public data offers. I took every pair where both cars finished and their stop counts differed by one, and compared their finishing times; the 367 teammate pairs with the same stop count serve as a reference.

| Comparison | Pairs | Extra-stop car vs teammate | 95% interval |
| --- | --- | --- | --- |
| One extra stop under green flag | 82 | **+16.4 s** | 8.8 to 25.0 s (race-clustered bootstrap) |
| Unforced stops only, adjusted for grid and first-stint pace | 426 in regression | **+10.3 s** | SE 3.6 s, p = 0.006 |
| Trimmed for outliers | 82 | +14.0 s | — |
| One extra stop under SC / VSC | 26 | −11.6 s | −28.4 to +3.3 s |

The car that stopped once more under green flag finished on average 16.4 s and 2.3 places behind its teammate. When I shuffled which teammate was the "extra-stop" car within each race, none of 2,000 permutations produced a gap this large (p ≈ 0.0005). Under a Safety Car the point estimate points the other way — a "cheap" stop — but with 26 pairs its interval still crosses zero, so treat it as a hint about direction only.

**Were the extra stops forced?** The car that stopped more might simply have had worse tyres, damage or traffic. With thresholds fixed in advance, the 82 pairs split into four groups: faster tyre degradation (17), abnormal stationary time (6), held up or covering a rival (34), and no visible reason (25). Only 28% were forced, and the voluntary "no visible reason" stops were the most expensive of all (19.5 s on average). Dropping the forced group and adjusting for grid position and first-stint pace still leaves +10.3 s.

The direction holds in every subsample: by season the extra stop cost 8.0, 25.0, 12.9 and 14.0 s; 12.6 s for the top five teams and 19.2 s for the rest; 21.7 s on street circuits and 15.0 s on permanent tracks.

Back to the simulator: for these teammate pairs it picks the faster driver correctly 68% of the time. The two lines of evidence support each other.

## Two races

One race where the simulator agrees with the teams and one where it does not shows when it can be trusted and when to be careful. "X s slower" below always means a paired difference for the same car with only the strategy changed (5,000 simulations). Both races can be replayed lap by lap on the home page of this site.

### Spain 2024: the simulator agrees

The whole top ten stopped twice. Verstappen won on soft–medium–soft (stops on laps 17 and 44); Mercedes and Ferrari each ran one car on soft–medium–hard.

- **Post-race mode** also picks two stops. A soft–hard one-stop is 11.0 s slower than Verstappen's actual plan, and soft–medium–hard is 7.9 s slower. The multi-car version recommends soft–medium–soft for six of the top ten, with stop laps within one to three laps of reality (Verstappen: recommended laps 16 and 43, actual 17 and 44).
- **Pre-race mode** recommends one stop instead (two one-stop plans are statistically tied). It uses the 2023 Barcelona tyre parameters, while in 2024 the soft degraded faster here (posterior mean about 0.10 s/lap versus 0.06 in 2023). This is pre-race mode's biggest weakness: it cannot see tyre behaviour change between seasons.
- **Teammates.** Pérez made one stop more than Verstappen (three) and finished 59.6 s behind him; Tsunoda three-stopped and finished 19.6 s behind Ricciardo. In this race two stops were right, and a third was expensive.

**Takeaway:** at a high-degradation track the simulator does recommend two stops. It is not unconditionally biased towards stopping less.

### Australia 2024: the simulator disagrees

The top ten again all stopped twice, eight of them on medium–hard–hard. Sainz won with stops on laps 15 and 40; Leclerc made his first stop as early as lap 8. There was a VSC on laps 17–18, but none of the top ten made their first stop under it.

- **Every version of the model prefers one stop.** A medium–hard one-stop is about 11 s faster than Sainz's actual strategy and about 14 s faster than Leclerc's; even without Safety Cars it is 9–12 s faster. Pre- and post-race modes, single- and multi-car versions all recommend one stop for every car in the top ten.
- **The reason is in the tyre parameters.** At this circuit the model estimates the medium as barely faster than the hard (about 0.01 s/lap) but degrading faster (0.080 versus 0.050 s/lap), so it prefers a short medium stint and a long hard one.
- **Why did everyone stop twice?** Public data cannot say. The long hard stint may have had problems invisible in clean laps (graining, wear risk), or the front-runners may have been covering each other: when rivals two-stop, a lone one-stopper is exposed to the undercut. Only one teammate pair split on stops here (Ocon three, Gasly two); Ocon was 18 s slower, but one pair is not evidence.

**Takeaway:** this is the archetypal under-stopping race. The simulator has a clear reason for its call, but it cannot see what the teams could. I treat races like this as open questions rather than as proof that teams got it wrong.

## What this shows, and what it does not

**What it shows.** In dry races from 2023 to 2026, an extra green-flag stop usually does not earn back its cost. Two methods with nothing in common support this, and it survives removing forced stops, trimming outliers, cleaning the pit-loss data, and splitting by season and by team.

**What it does not show:**

- **That teams are wrong.** Stop counts are not randomly assigned. I removed the forced reasons I could measure, but team orders, defending track position and risk aversion are invisible. A decision that costs 10 s on average can still be right in a particular situation.
- **That the simulator is ready for live decisions.** In pre-race mode it matches teams' stop counts only about 47% of the time, and its compound choices match even less often: last season's compound differences do not always carry over.
- **That it holds everywhere.** The teammate sample is 82 pairs. The model tables cover dry races only, and the optimiser searches one- and two-stop plans (the 71 car-races that actually used three or more stops are still scored against it).
- **That every parameter was known before the race.** Overtaking, start and Safety Car restart constants are calibrated on the full sample; a sensitivity check shows they change 15 of 520 recommendations.

A stricter test would need team radio and internal data, which public data cannot provide.

## Code, site and reproduction

All code, tests and analyses are open: [github.com/junyanguo02/f1a-strategy-sim](https://github.com/junyanguo02/f1a-strategy-sim) (release v0.1.0). This site has lap-by-lap replays for 2024 and a strategy lab where you can drag pit stops and re-simulate in the browser.

Raw data are not redistributed; scripts in the repository rebuild them from OpenF1 and Jolpica (OpenF1 data are for non-commercial use).

1. Download and build the data: see `scripts/download_data.md`.
2. Fit the tyre model: `models/tyre_v3/fit_tyre_v3_1.py` (about 30–45 minutes per season on a laptop CPU, 4 chains).
3. Back-tests: `scripts/backtest.py` and `scripts/backtest_field.py`; add `--prerace` for pre-race mode and, for the multi-car back-test, `--pit-loss-filter clean` for the robustness check.
4. Local app: `streamlit run app/strategy_app.py`.

The teammate and robustness analyses live in `analysis/teammate_outcomes/`, `analysis/extra_stop/` and `analysis/extra_stop_robustness/`. Each runs with one command; scripts with a two-run record produced byte-identical outputs on the same inputs and environment.

*This project is unofficial and not associated with Formula 1 or any team.*
