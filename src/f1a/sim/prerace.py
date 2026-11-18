"""Pre-race features. Results/actual strategies are joined only in the scoring layer."""
from dataclasses import dataclass, replace
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .backtest import candidate_sequences
from .field import Car, FieldParams
from .optimize import optimize_stops
from .params import COMPOUNDS, TyreParams, race_index_map
from .race import RaceConfig
from .strategy import Strategy


class Unavailable(ValueError):
    """A prescribed input is unavailable; exclude the race and record the reason."""


def calendar(root):
    records = []
    for path in sorted((root / "data/raw/jolpica").glob("*_races_limit=100.json")):
        for r in json.loads(path.read_text())["MRData"]["RaceTable"]["Races"]:
            records.append((int(r["season"]), int(r["round"]), pd.Timestamp(r["date"]), r["Circuit"]["circuitId"]))
    result = pd.DataFrame(records, columns=["year", "round", "date", "circuit_id"])
    if result.duplicated(["year", "round"]).any():
        raise ValueError("duplicate calendar key")
    return result


def before(frame, schedule, cutoff):
    """Whitelist race keys by date, preserving source order (including posterior indices)."""
    keys = ["year", "round"]
    known = schedule.loc[schedule.date < cutoff, keys]
    return frame.merge(known, on=keys, how="inner", validate="many_to_one", sort=False)


def prior_tyre(samples, previous_laps, circuit, n_laps, n_draws, seed):
    seasons = previous_laps.year.unique()
    races = race_index_map(previous_laps, seasons)
    if len(seasons) != 1 or samples["gamma"].shape[1] != len(races):
        raise Unavailable("prior_index_mismatch")
    sites = previous_laps.loc[previous_laps.circuit_id.eq(circuit), "race_id"].unique()
    if len(sites) > 1:
        raise Unavailable("ambiguous_prior_circuit")
    count = len(samples["nu"])
    take = np.arange(count)
    if n_draws < count:
        take = np.sort(np.random.default_rng(seed).choice(count, n_draws, replace=False))
    if len(sites):
        i = races.index(sites[0])
        gamma, beta, k0 = (samples[name][take, i] for name in ("gamma", "beta", "k0"))
    else:
        gamma = samples["mu_gamma"][take, 0]
        beta = samples["mu_beta"][take, 0]
        k0 = np.column_stack([np.zeros(len(take)), samples["mu_k0"][take, 0]])
    return TyreParams(k0, beta, samples["kappa"][take], gamma, np.median(samples["sigma"][take], axis=1),
                      samples["nu"][take], samples["sd_stint"][take], n_laps / 2,
                      {"year":int(seasons[0]), "source_race":sites[0] if len(sites) else "season_mean"})


def driver_paces(clean, past, season, first_round):
    keys = ["year", "round"]
    events = past[past.year.eq(season - int(first_round))].sort_values("date")
    if first_round:
        events = events.tail(3)
    eligible = clean.merge(events[keys + ["date"]], on=keys, validate="many_to_one")
    eligible["offset"] = eligible.lap_time_s - eligible.groupby(keys).lap_time_s.transform("median")
    offsets = eligible.groupby(["driver", "date"]).offset.median().dropna().reset_index().sort_values("date")
    return offsets.groupby("driver").tail(3).groupby("driver").offset.median().to_dict()


def cars_from_prerace(grid, paces, fastest):
    """Read only driver identity and starting slot; finishing order must not seed draws."""
    slots = sorted((int(row["grid"]), row["Driver"]["code"]) for row in grid)
    if any(n < 1 for n, _ in slots) or len({n for n, _ in slots}) != len(slots):
        raise Unavailable("unresolved_starting_grid")
    cars = [Car(name, paces.get(name, .5), slot, fastest) for slot, name in slots]
    return cars, [c.name for c in cars if c.name not in paces]


@dataclass
class Inputs:
    race: RaceConfig
    tyre: TyreParams
    paces: dict
    circuit: str
    cutoff: pd.Timestamp
    historical_end: pd.Timestamp
    field_params: FieldParams


def build_inputs(root, season, rnd, n_draws, seed, by_circuit=False):
    root = Path(root)
    posterior = root / f"models/tyre_v3/outputs31_{season-1}/posterior_core.npz"
    if not posterior.is_file():
        raise Unavailable("no_prior_season")
    schedule = calendar(root)
    target = schedule[schedule.year.eq(season) & schedule["round"].eq(rnd)].iloc[0]
    past = schedule[schedule.date < target.date]
    if (schedule.loc[schedule.year.eq(season-1), "date"] >= target.date).any():
        raise Unavailable("prior_season_not_complete")
    processed = root / "data/processed"
    def history(frame):
        return before(frame, schedule, target.date)
    laps = history(pd.read_parquet(processed / "laps_pit_filled.parquet"))
    clean = laps.loc[laps.is_clean.fillna(False)].merge(past[["year","round","circuit_id"]],
                                                       on=["year","round"], validate="many_to_one")
    base = clean[clean.year.eq(season-1) & clean.circuit_id.eq(target.circuit_id)].lap_time_s.median()
    if pd.isna(base):
        raise Unavailable("no_base_lap")
    distances = pd.read_csv(root / "analysis/backtest/prerace/scheduled_laps.csv")
    distance = distances[distances.year.eq(season) & distances["round"].eq(rnd)
                         & distances.circuit_id.eq(target.circuit_id)]
    if len(distance) != 1:
        raise Unavailable("no_scheduled_laps")
    n_laps = int(distance.total_laps.iloc[0])
    previous = history(pd.read_parquet(processed / "model_laps.parquet")).query("year == @season - 1")
    with np.load(posterior) as samples:
        tyre = TyreParams.from_prerace(samples, previous, target.circuit_id, n_laps, n_draws, seed)
        fp = FieldParams(d_gap_lt1=float(samples["d_gap_lt1"].mean()), d_gap_1to2=float(samples["d_gap_1to2"].mean()))
    race = RaceConfig.from_prerace(root, schedule, target.date, f"{season}-{rnd}", target.circuit_id,
                                   n_laps, base, laps, by_circuit)
    first = not past.year.eq(season).any()
    return Inputs(race, tyre, driver_paces(clean, past, season, first), target.circuit_id,
                  target.date, past.date.max(), fp)


def optimization(inputs, n_sims, seed, step):
    # Dry model vocabulary. Never look at target entrants' tyre choices for eligibility.
    return optimize_stops(inputs.race, inputs.tyre, candidate_sequences(COMPOUNDS), n_sims=n_sims, seed=seed, step=step)


def fastest_plan(opt):
    best = opt.dropna(subset=["mean"]).iloc[0]
    names = {t[0]:t for t in COMPOUNDS}
    tyres = [names[c] for c in best.sequence.split("-")]
    return Strategy(best.sequence, tyres[0], list(zip(best.stops, tyres[1:])))


def sensitivity_params(fp):
    """One change at a time, all other frozen T023 parameters unchanged."""
    return {"pass_minus": replace(fp, pass_a=fp.pass_a - .5),
            "pass_plus": replace(fp, pass_a=fp.pass_a + .5),
            "restart_050": replace(fp, sc_restart_gap=.5),
            "restart_120": replace(fp, sc_restart_gap=1.2)}


def predict_field(inputs, grid, opt, drivers, n_sims, seed, sensitivity=False):
    from .field import evaluate_ego, field_candidates
    fastest = fastest_plan(opt)
    cars, new = cars_from_prerace(grid, inputs.paces, fastest)
    if len(cars) < 8:
        raise Unavailable("insufficient_field")
    candidates = field_candidates(opt.copy(), inputs.race.total_laps)
    rows = []
    for ego, car in enumerate(cars):
        if car.name not in drivers:
            continue
        result = evaluate_ego(inputs.race, inputs.tyre, cars, ego, candidates, inputs.field_params,
                              n_sims=n_sims, seed=seed)
        winner = result.iloc[0]
        plan = next(s for s in candidates if s.name == winner.strategy)
        variants = {}
        if sensitivity:
            for name, fp in sensitivity_params(inputs.field_params).items():
                evaluated = evaluate_ego(inputs.race, inputs.tyre, cars, ego, candidates, fp,
                                         n_sims=n_sims, seed=seed)
                variant = next(s for s in candidates if s.name == evaluated.iloc[0].strategy)
                variants["stops_" + name] = len(variant.stops)
        rows.append(dict(**variants, driver=car.name, grid=car.grid, pace=car.pace, new_driver=car.name in new,
                         fallback_drivers=",".join(new), field_size=len(cars), opponent_plan=fastest.label(),
                         model_best=plan.label(), model_stops=len(plan.stops), best_exp_pos=winner.exp_pos))
    return pd.DataFrame(rows)


def run_season(root, args, field=False):
    from .backtest import score_single
    root = Path(root)
    actual_dir = root / "analysis/actual_strategies"
    scores = pd.read_parquet(actual_dir / "strategies.parquet")
    races = pd.read_csv(actual_dir / "outputs/race_summary.csv")
    # These filters reproduce the retrospective scoring cohort, never feature inputs.
    races = races[races.year.eq(args.season) & races.dry_for_analysis & ~races.has_red_flag & races.usable]
    rows = []
    for r in races.itertuples():
        rid = f"{r.year}-{r.round}"
        if field and args.races and rid not in args.races:
            continue
        try:
            inputs = build_inputs(root, r.year, r.round, args.draws, args.seed, getattr(args, "fcy_by_circuit", False))
            grid = None
            if field:
                path = root / f"data/raw/jolpica/{r.year}_{r.round}_results_limit=100.json"
                if not path.is_file():
                    raise Unavailable("missing_starting_grid")
                grid = json.loads(path.read_text())["MRData"]["RaceTable"]["Races"][0]["Results"]
            opt = optimization(inputs, 1000 if field else args.sims, args.seed, 2 if field else args.step)
            observed = scores[scores.year.eq(r.year) & scores["round"].eq(r.round) & scores.finisher]
            if field:
                actual = observed[observed.finish_position.between(1,args.top)].set_index("driver")
                forecast = predict_field(inputs, grid, opt, set(actual.index), args.sims, args.seed, args.sensitivity)
                present = set(forecast.driver) if not forecast.empty else set()
                for driver in actual.index.difference(list(present)):
                    rows.append(dict(race_id=rid,driver=driver,skipped="missing_driver_grid"))
                for row in forecast.to_dict("records"):
                    car = actual.loc[row["driver"]]
                    if not isinstance(car.strategy,str) or not set(car.strategy)<=set("HMS") or pd.isna(car.n_pit_stops):
                        rows.append(dict(race_id=rid,driver=row["driver"],skipped="missing_dry_strategy_label"))
                        continue
                    n = int(car.n_pit_stops)
                    seq = row["model_best"].split(" @")[0].replace("-", "")
                    rows.append(dict(**row,race_id=rid,finish=car.finish_position,actual=car.strategy,actual_stops=n,
                                     stops_match=row["model_stops"]==n,combo_match=sorted(seq)==sorted(car.strategy),
                                     cutoff=str(inputs.cutoff.date()),latest_history=str(inputs.historical_end.date())))
            else:
                observed = observed[~observed.wet_tyre_used & observed.strategy.str.fullmatch("[HMS]+", na=False)]
                if len(observed) < 5:
                    raise Unavailable("insufficient_scoring_finishers")
                row = score_single(opt, observed, inputs.race, inputs.circuit, COMPOUNDS, len(opt))
                rows.append(dict(**row,cutoff=str(inputs.cutoff.date()),latest_history=str(inputs.historical_end.date())))
            print(rid, "完成", flush=True)
        except Unavailable as exc:
            rows.append(dict(race_id=rid,skipped=str(exc)))
            print(rid, "跳过", str(exc), flush=True)
    out = pd.DataFrame(rows)
    path = Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(path,index=False)
    valid = out.dropna(subset=["stops_match"]) if "stops_match" in out else out.iloc[:0]
    print(f"赛季 {args.season}，有效 {len(valid)}，跳过记录 {len(out)-len(valid)}")
    if len(valid):
        model, truth = ("model_stops","actual_stops") if field else ("model_n_stops","top10_modal_stops")
        print(f"进站次数一致 {valid.stops_match.mean():.1%}，配方组合一致 {valid.combo_match.mean():.1%}，"
              f"少停 {(valid[model]<valid[truth]).sum()}，多停 {(valid[model]>valid[truth]).sum()}")
