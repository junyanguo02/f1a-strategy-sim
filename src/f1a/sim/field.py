"""多车蒙特卡洛模拟器（v0.2 草稿，Claude 编写）。

在单车引擎的圈时模型之上加入：每车配速偏移、发车位、前车交通惩罚、超车概率与被堵、进站后按累计时间
回到赛道（自然产生 undercut / overcut）、SC 结束时车距压缩、第 1 圈按起步配方的时间偏移（方向 D）。
被评估的“本车”换不同策略，其他车策略固定；所有策略共用同一组随机数。

参数来源见 FieldParams 的说明（超车、起步、SC 重启已按 T023 校准）。
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field as dc_field, replace
from pathlib import Path
import numpy as np
import pandas as pd
from .params import TyreParams, COMPOUNDS, AGE_REF
from .race import RaceConfig, N_PROGRESS_BINS
from .strategy import Strategy
from .engine import common_draws, _quantile_pick

_CI = {c: i for i, c in enumerate(COMPOUNDS)}


@dataclass
class FieldParams:
    """默认值来源（2026-11-18 校准）：
    - 交通惩罚：v3.1 后验均值（回测时按赛季后验覆盖）。
    - 超车：T023 “上一圈 Δ” 表（距前车 <1 秒的车对、下一圈超车比例）加权 logistic 拟合，Δ 为后车干净圈时优势。
      该表是全赛道合并；街道赛中等 Δ 下更难超车，暂未区分。
    - 起步：T023 起步表，后段发车位软胎比中性胎多提升约 0.5 位、硬胎第 1 圈明显更慢，取软 −0.5 / 中 0 / 硬 +1.5 秒（粗略）。
    - SC 重启间距：T023 中位数 0.81 秒。
    - 发车间距、被堵最小间隔：经验值，未校准。"""
    d_gap_lt1: float = 0.30
    d_gap_1to2: float = 0.13
    pass_a: float = -2.68
    pass_b: float = 2.05
    min_gap: float = 0.25
    grid_gap: float = 0.25
    start_offset: dict = dc_field(default_factory=lambda: {"SOFT": -0.5, "MEDIUM": 0.0, "HARD": 1.5})
    sc_restart_gap: float = 0.81
    notes: list = dc_field(default_factory=lambda: ["超车/起步/SC 重启按 T023 校准；发车间距、被堵间隔为经验值"])


@dataclass
class Car:
    name: str
    pace: float          # 相对基础圈时的配速偏移（秒，负为快）
    grid: int            # 发车位（1 起）
    strategy: Strategy


def _plan(strategy: Strategy, N: int):
    stop_at = dict(strategy.stops)
    comp, age, stint = strategy.start_compound, strategy.start_age, 0
    C = np.empty(N, int); A = np.empty(N, float); J = np.empty(N, int); PIT = np.zeros(N, bool)
    for l in range(1, N + 1):
        C[l - 1], A[l - 1], J[l - 1] = _CI[comp], age, stint
        if l in stop_at:
            PIT[l - 1] = True; comp = stop_at[l]; age = 1; stint += 1
        else:
            age += 1
    return C, A, J, PIT


def field_draws(race: RaceConfig, tyre: TyreParams, n_cars: int, n_sims: int, seed: int, max_stints: int = 6) -> dict:
    """共同随机数：FCY 时间线与后验抽样全场共用；每车独立的噪声、stint 截距、进站损失分位、超车随机数。"""
    base = common_draws(race, tyre, n_sims, seed, max_stints=1)
    rng = np.random.default_rng(seed + 7919)
    S, N = n_sims, race.total_laps
    nu = tyre.nu[base["d"]][:, None, None]
    z = rng.standard_normal((S, n_cars, N))
    g = rng.chisquare(np.broadcast_to(nu, (S, n_cars, N)))
    return dict(seed=seed, n_sims=S, d=base["d"], status=base["status"], phase=base["phase"],
                t_noise=z / np.sqrt(g / nu), u_stint=rng.standard_normal((S, n_cars, max_stints)),
                pit_u=rng.random((S, n_cars, N)), pass_u=rng.random((S, n_cars, N)))


def simulate_field(race: RaceConfig, tyre: TyreParams, cars: list, fp: FieldParams, n_sims: int = 300,
                   seed: int = 0, draws: dict | None = None) -> dict:
    N, K = race.total_laps, len(cars)
    for c in cars:
        c.strategy.validate(N)
    max_st = max(len(c.strategy.stops) for c in cars) + 1
    D = draws if draws is not None else field_draws(race, tyre, K, n_sims, seed, max_stints=max(6, max_st))
    S, d = D["n_sims"], D["d"]
    plans = [_plan(c.strategy, N) for c in cars]
    pace = np.array([c.pace for c in cars])
    st, ph = D["status"], D["phase"]
    # 起点：按发车位排开
    T = np.tile((np.array([c.grid for c in cars], float) - 1) * fp.grid_gap, (S, 1))
    tot_pit = np.zeros((S, K))
    gamma = tyre.gamma[d]; sd_u = tyre.sd_stint[d]; sig = tyre.sigma[d]
    rows = np.arange(S)
    for l in range(1, N + 1):
        li = l - 1
        rem_c = (N - l) - tyre.rem_mean
        free = np.empty((S, K)); pit = np.zeros((S, K)); pits_now = np.zeros(K, bool)
        for k, (C, A, J, PIT) in enumerate(plans):
            c, a = C[li], A[li] - AGE_REF
            free[:, k] = (race.base_lap_s + pace[k] + tyre.k0[d, c] + gamma * rem_c + tyre.beta[d, c] * a
                          + tyre.kappa[d, c] * a ** 2 + sd_u * D["u_stint"][:, k, J[li]] + sig * D["t_noise"][:, k, li])
            if l == 1:
                free[:, k] += fp.start_offset.get(cars[k].strategy.start_compound, 0.0)
            if PIT[li]:
                pits_now[k] = True
                u = D["pit_u"][:, k, li]; loss = _quantile_pick(race.pit_normal, u)
                s_l, p_l = st[:, li], np.minimum(ph[:, li], 3)
                for code, name in ((1, "VSC"), (2, "SC")):
                    for phv in (1, 2, 3):
                        m = (s_l == code) & (p_l == phv)
                        if m.any():
                            loss[m] = _quantile_pick(race.pit_fcy[(name, phv)], u[m])
                pit[:, k] = loss
        tot_pit += pit
        green = st[:, li] == 0
        fcy_mult = np.where(st[:, li] == 2, race.lap_mult["SC"], race.lap_mult["VSC"])
        # FCY 圈：所有车同一圈时（不超车），只加各自进站损失
        Tn = np.where(green[:, None], np.nan, T + (race.base_lap_s * fcy_mult)[:, None] + pit)
        # 绿旗圈：按圈初位置从前往后处理交通与超车
        if green.any():
            order = np.argsort(T, axis=1)
            Tg = np.empty((S, K))
            for pos in range(K):
                ci = order[:, pos]
                t_start = T[rows, ci]
                f = free[rows, ci]
                prop = t_start + f + pit[rows, ci]
                if pos > 0:
                    ai = order[:, pos - 1]
                    gap = t_start - T[rows, ai]
                    pen = np.where(gap < 1.0, fp.d_gap_lt1, np.where(gap < 2.0, fp.d_gap_1to2, 0.0))
                    prop = prop + pen
                    ahead_new = Tg[rows, ai]
                    either_pits = pits_now[ci] ^ pits_now[ai]   # 只有一方进站时不互相阻挡；两车同圈进站仍按先后排队
                    closing = (prop < ahead_new + fp.min_gap) & ~either_pits
                    delta = free[rows, ai] - f
                    log_odds = fp.pass_a + fp.pass_b * delta
                    p_pass = 1.0 / (1.0 + np.exp(np.minimum(-log_odds, 709.0)))
                    passed = D["pass_u"][rows, ci, li] < p_pass
                    blocked = closing & ~passed
                    prop = np.where(blocked, ahead_new + fp.min_gap, prop)
                Tg[rows, ci] = prop
            Tn = np.where(green[:, None], Tg, Tn)
        # SC 段最后一圈：压缩车距
        nxt_sc = st[:, li + 1] == 2 if li < N - 1 else np.zeros(S, bool)
        sc_end = (st[:, li] == 2) & ~nxt_sc
        if np.any(sc_end):
            o = np.argsort(Tn, axis=1)
            ordered = np.take_along_axis(Tn, o, axis=1)
            gaps = np.minimum(np.diff(ordered, axis=1), fp.sc_restart_gap)
            compressed = ordered.copy()
            compressed[:, 1:] = ordered[:, :1] + np.cumsum(gaps, axis=1)
            np.put_along_axis(Tn, o, np.where(sc_end[:, None], compressed, ordered), axis=1)
        T = Tn
    pos = np.argsort(np.argsort(T, axis=1), axis=1) + 1
    return dict(total=T, position=pos, pit_loss=tot_pit)


def evaluate_ego(race, tyre, cars: list, ego: int, strategies: list, fp: FieldParams, n_sims: int = 300,
                 seed: int = 0) -> pd.DataFrame:
    """本车（cars[ego]）换不同策略，其他车不变；共用随机数。返回期望名次、名次分布与总用时差。"""
    max_st = max([len(s.stops) for s in strategies] + [len(c.strategy.stops) for c in cars]) + 1
    D = field_draws(race, tyre, len(cars), n_sims, seed, max_stints=max(6, max_st))
    res = []
    for s in strategies:
        cs = [replace(c) for c in cars]; cs[ego] = replace(cs[ego], strategy=s)
        r = simulate_field(race, tyre, cs, fp, draws=D)
        res.append((s, r["position"][:, ego], r["total"][:, ego]))
    P = np.vstack([p for _, p, _ in res]); T = np.vstack([t for _, _, t in res])
    best = int(np.argmin(P.mean(1) + 1e-6 * T.mean(1)))
    rows = []
    for i, (s, p, t) in enumerate(res):
        rows.append(dict(strategy=s.name, label=s.label(), exp_pos=p.mean(), p_win_pos=(p <= P[best]).mean(),
                         pos_p10=np.quantile(p, .1), pos_p90=np.quantile(p, .9), mean_time=t.mean(),
                         d_pos_vs_best=(p - P[best]).mean(), d_time_vs_best=(t - T[best]).mean()))
    return pd.DataFrame(rows).sort_values(["exp_pos", "mean_time"]).reset_index(drop=True)


def cars_from_project(root, race_id: str, strategies: pd.DataFrame | None = None) -> list:
    """按实际比赛装配全场：配速偏移 = 该车干净圈中位数 − 全场干净圈中位数（事后值）；发车位来自 Jolpica；
    策略为该车实际策略（只取干胎、无湿胎、完赛或至少跑完 90% 的车）。"""
    root = Path(root)
    yr, rd = map(int, race_id.split("-"))
    ml = pd.read_parquet(root / "data" / "processed" / "model_laps.parquet")
    r = ml[ml.race_id == race_id]
    med = r.lap_time_s.median()
    pace = (r.groupby("driver").lap_time_s.median() - med).to_dict()
    j = json.load(open(root / "data" / "raw" / "jolpica" / f"{yr}_{rd}_results_limit=100.json"))
    res = j["MRData"]["RaceTable"]["Races"][0]["Results"]
    grid = {x["Driver"]["code"]: (int(x["grid"]) if int(x["grid"]) > 0 else 20) for x in res}
    if strategies is None:
        strategies = pd.read_parquet(root / "analysis" / "actual_strategies" / "strategies.parquet")
    s = strategies[(strategies.year == yr) & (strategies["round"] == rd)]
    N = int(r.total_laps.iloc[0])
    cars = []
    for _, x in s.iterrows():
        if x.driver not in pace or not isinstance(x.strategy, str) or not set(x.strategy) <= set("HMS") or len(x.strategy) < 1:
            continue
        if (x.finish_laps if pd.notna(x.finish_laps) else 0) < 0.9 * N:
            continue
        laps = [int(v) for v in str(x.pit_laps).split(";") if v not in ("", "nan", "None")]
        comps = [{"H": "HARD", "M": "MEDIUM", "S": "SOFT"}[ch] for ch in x.strategy]
        if len(comps) != len(laps) + 1 or len(set(comps)) < 2 or any(l >= N for l in laps):
            continue
        cars.append(Car(x.driver, float(pace[x.driver]), grid.get(x.driver, 20), Strategy(x.strategy, comps[0], list(zip(laps, comps[1:])))))
    # 发车位按入选车辆重新排名，避免空位
    for i, c in enumerate(sorted(cars, key=lambda c: c.grid)):
        c.grid = i + 1
    return cars


def backtest_field_race(root, race_id: str, season: int, fp: FieldParams | None = None, n_sims: int = 300,
                        n_draws: int = 200, seed: int = 0, top: int = 10, strategies: pd.DataFrame | None = None) -> pd.DataFrame:
    """对一场比赛前 top 名完赛车逐一做“本车”：其他车固定实际策略，在一停 / 两停候选里选期望名次最好的，
    与该车实际进站次数比较。候选来自单车优化（每个进站次数取最好的两个配方顺序），再在进站圈附近 ±3、±6 圈微调。"""
    from .optimize import optimize_stops
    from .params import estimated_compounds
    root = Path(root)
    ml = pd.read_parquet(root / "data" / "processed" / "model_laps.parquet")
    race = RaceConfig.from_project(root, race_id)
    npz = root / "models" / "tyre_v3" / f"outputs31_{season}" / "posterior_core.npz"
    tyre = TyreParams.from_posterior(npz, race_id, [season], root / "data" / "processed" / "model_laps.parquet", n_draws=n_draws, seed=seed)
    if fp is None:
        z = np.load(npz); fp = FieldParams(d_gap_lt1=float(z["d_gap_lt1"].mean()), d_gap_1to2=float(z["d_gap_1to2"].mean()))
    cars = cars_from_project(root, race_id, strategies)
    if len(cars) < 8:
        return pd.DataFrame([dict(race_id=race_id, skipped="可用车辆不足 8 辆")])
    est = estimated_compounds(ml, race_id)
    used = {ch for c in cars for ch in c.strategy.name}
    allowed = [c for c in COMPOUNDS if c[0] in used and c in est]
    if len(allowed) < 2:
        return pd.DataFrame([dict(race_id=race_id, skipped=f"可用配方不足：{allowed}")])
    import itertools
    seqs = [q for k in (1, 2) for q in itertools.product(allowed, repeat=k + 1) if len(set(q)) >= 2]
    opt = optimize_stops(race, tyre, seqs, n_sims=600, seed=seed, step=2)
    cand = field_candidates(opt, race.total_laps)
    fin = (strategies if strategies is not None else pd.read_parquet(root / "analysis" / "actual_strategies" / "strategies.parquet"))
    yr, rd = map(int, race_id.split("-"))
    fin = fin[(fin.year == yr) & (fin["round"] == rd) & fin.finisher & fin.finish_position.between(1, top)].set_index("driver").finish_position
    egos = sorted([i for i, c in enumerate(cars) if pd.notna(fin.get(c.name))], key=lambda i: fin[cars[i].name])
    if not egos:
        return pd.DataFrame([dict(race_id=race_id, skipped="前列完赛车均缺少可用策略或配速")])
    single_best_n = int(opt.dropna(subset=["mean"]).iloc[0].n)
    rows = []
    for i in egos:
        ev = evaluate_ego(race, tyre, cars, i, cand + [cars[i].strategy], fp, n_sims=n_sims, seed=seed)
        best = ev.loc[ev.strategy != cars[i].strategy.name].iloc[0]
        model_n = int(best.label.count("@") and len(best.label.split("@")[1].split(",")))
        actual = cars[i].strategy
        act_row = ev[ev.strategy == actual.name]
        rows.append(dict(race_id=race_id, driver=cars[i].name, finish=int(fin[cars[i].name]),
                         actual=actual.label(), actual_stops=len(actual.stops), model_best=best.label, model_stops=model_n,
                         single_car_stops=single_best_n, stops_match=model_n == len(actual.stops),
                         single_match=single_best_n == len(actual.stops),
                         best_exp_pos=best.exp_pos, actual_exp_pos=float(act_row.exp_pos.iloc[0]) if len(act_row) else np.nan))
    return pd.DataFrame(rows)


def field_candidates(opt, N):
    opt["n"] = opt.sequence.str.count("-")
    cand = []
    for n in (1, 2):
        for _, o in opt[opt.n == n].dropna(subset=["mean"]).head(2).iterrows():
            comps = [{"H": "HARD", "M": "MEDIUM", "S": "SOFT"}[ch] for ch in o.sequence.split("-")]
            base = list(o.stops)
            shifts = [(a,) for a in (-6, -3, 0, 3, 6)] if n == 1 else [(a, b) for a in (-6, -3, 0, 3, 6) for b in (-3, 0, 3)]
            for sh in shifts:
                laps = [base[i] + sh[i] for i in range(n)]
                if laps[0] < 5 or laps[-1] > N - 5 or any(b - a < 6 for a, b in zip(laps, laps[1:])):
                    continue
                cand.append(Strategy(f"{n}停 {o.sequence} @{','.join(map(str, laps))}", comps[0], list(zip(laps, comps[1:]))))
    return cand
