"""单车蒙特卡洛引擎：共同随机数下逐圈生成圈时并汇总策略比较。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .params import TyreParams, COMPOUNDS, AGE_REF
from .race import RaceConfig, FCY_TYPES, N_PROGRESS_BINS
from .strategy import Strategy

_CI = {c: i for i, c in enumerate(COMPOUNDS)}


def common_draws(race: RaceConfig, tyre: TyreParams, n_sims: int, seed: int, max_stints: int = 6) -> dict:
    """一次性生成所有随机量，供多个策略共用（共同随机数）。同 seed 结果逐位一致。"""
    rng = np.random.default_rng(seed)
    N, S = race.total_laps, n_sims
    cycles, remainder = divmod(S, tyre.n_draws)
    d = np.tile(np.arange(tyre.n_draws), cycles)
    if remainder:
        extra = rng.choice(tyre.n_draws, remainder, replace=False)
        d = np.concatenate((d, extra))
    rng.shuffle(d)
    # FCY 时间线
    status = np.zeros((S, N), dtype=np.int8)               # 0 绿旗 1 VSC 2 SC
    phase = np.zeros((S, N), dtype=np.int16)               # FCY 段内第几圈（1 起）
    remaining = np.zeros(S, dtype=int); cur = np.zeros(S, dtype=np.int8); ph = np.zeros(S, dtype=int)
    edges = np.linspace(0, 1, N_PROGRESS_BINS + 1)
    u_start = rng.random((S, N)); u_type = rng.random((S, N))
    dur_u = rng.random((S, N))
    for l in range(1, N + 1):
        idle = remaining == 0
        if l == 1:
            p_sc, p_vsc = race.lap1_prob["SC"], race.lap1_prob["VSC"]
        else:
            b = min(int(np.digitize(l / N, edges[1:-1])), N_PROGRESS_BINS - 1)
            p_sc, p_vsc = race.hazard["SC"][b], race.hazard["VSC"][b]
        start = idle & (u_start[:, l - 1] < p_sc + p_vsc)
        is_sc = u_type[:, l - 1] < p_sc / max(p_sc + p_vsc, 1e-12)
        for t_code, t_name, mask in ((2, "SC", start & is_sc), (1, "VSC", start & ~is_sc)):
            if mask.any():
                dur = np.sort(race.durations[t_name])
                k = np.minimum((dur_u[mask, l - 1] * len(dur)).astype(int), len(dur) - 1)
                remaining[mask] = dur[k]; cur[mask] = t_code; ph[mask] = 0
        active = remaining > 0
        ph[active] += 1
        status[active, l - 1] = cur[active]; phase[active, l - 1] = ph[active]
        remaining[active] -= 1
    # 圈速噪声：t(nu) = Z / sqrt(G/nu)，nu 随后验抽样变化
    nu = tyre.nu[d][:, None]
    z = rng.standard_normal((S, N))
    g = rng.chisquare(np.repeat(nu, N, axis=1))
    t_noise = z / np.sqrt(g / nu)
    return dict(seed=seed, n_sims=S, d=d, status=status, phase=phase, t_noise=t_noise,
                u_stint=rng.standard_normal((S, max_stints)), pit_u=rng.random((S, N)))


def _quantile_pick(samples: np.ndarray, u: np.ndarray) -> np.ndarray:
    s = np.sort(np.asarray(samples, float))
    if len(s) == 0:
        raise ValueError("进站损失样本为空")
    return s[np.minimum((u * len(s)).astype(int), len(s) - 1)]


def simulate(race: RaceConfig, tyre: TyreParams, strategy: Strategy, n_sims: int = 1000, seed: int = 0,
             draws: dict | None = None, return_laps: bool = False) -> dict:
    N = race.total_laps
    strategy.validate(N)
    D = draws if draws is not None else common_draws(race, tyre, n_sims, seed,
                                                      max_stints=max(6, len(strategy.stops) + 1))
    S, d = D["n_sims"], D["d"]
    if len(strategy.stops) + 1 > D["u_stint"].shape[1]:
        raise ValueError("段数超过 common_draws 的 max_stints")
    # 每圈的配方、胎龄、段序号、是否进站圈（确定性）
    stop_at = dict(strategy.stops)
    comp, age, stint = strategy.start_compound, strategy.start_age, 0
    C = np.empty(N, int); A = np.empty(N, float); J = np.empty(N, int); PIT = np.zeros(N, bool)
    for l in range(1, N + 1):
        C[l - 1], A[l - 1], J[l - 1] = _CI[comp], age, stint
        if l in stop_at:
            PIT[l - 1] = True; comp = stop_at[l]; age = 1; stint += 1
        else:
            age += 1
    rem_c = (N - np.arange(1, N + 1)) - tyre.rem_mean
    a = A - AGE_REF
    k0, beta, kappa = tyre.k0[d][:, C], tyre.beta[d][:, C], tyre.kappa[d][:, C]
    green = (race.base_lap_s + k0 + tyre.gamma[d][:, None] * rem_c + beta * a + kappa * a ** 2
             + tyre.sd_stint[d][:, None] * D["u_stint"][:, J] + tyre.sigma[d][:, None] * D["t_noise"])
    st = D["status"]
    lap = np.where(st == 2, race.base_lap_s * race.lap_mult["SC"],
                   np.where(st == 1, race.base_lap_s * race.lap_mult["VSC"], green))
    pit_loss = np.zeros((S, N))
    for l in np.flatnonzero(PIT):
        u = D["pit_u"][:, l]; s_l = st[:, l]; p_l = np.minimum(D["phase"][:, l], 3)
        loss = _quantile_pick(race.pit_normal, u)
        for code, name in ((1, "VSC"), (2, "SC")):
            for phv in (1, 2, 3):
                m = (s_l == code) & (p_l == phv)
                if m.any():
                    loss[m] = _quantile_pick(race.pit_fcy[(name, phv)], u[m])
        pit_loss[:, l] = loss
    total = (lap + pit_loss).sum(axis=1)
    out = dict(strategy=strategy.name, label=strategy.label(), total=total,
               n_sc=((st == 2) & (D["phase"] == 1)).sum(1),
               n_vsc=((st == 1) & (D["phase"] == 1)).sum(1),
               pit_loss=pit_loss[:, PIT].sum(1))
    if return_laps:
        out["laps"] = lap + pit_loss
    return out


def evaluate(race: RaceConfig, tyre: TyreParams, strategies: list, n_sims: int = 5000, seed: int = 0) -> pd.DataFrame:
    """共同随机数下比较多个策略。返回每个策略的总用时分布摘要、相对最优策略的配对差、以及成为最快策略的比例。"""
    max_st = max(len(s.stops) for s in strategies) + 1
    D = common_draws(race, tyre, n_sims, seed, max_stints=max(6, max_st))
    res = [simulate(race, tyre, s, draws=D) for s in strategies]
    T = np.vstack([r["total"] for r in res])
    best = int(np.argmin(T.mean(1)))
    fastest = np.bincount(T.argmin(0), minlength=len(res)) / n_sims
    rows = []
    for i, r in enumerate(res):
        t = r["total"]; diff = t - T[best]
        q95 = np.quantile(t, 0.95)
        rows.append(dict(strategy=r["strategy"], label=r["label"], mean=t.mean(), sd=t.std(ddof=1),
                         p05=np.quantile(t, .05), p50=np.median(t), p95=q95, cvar95=t[t >= q95].mean(),
                         delta_mean=diff.mean(), delta_lo=np.quantile(diff, .025), delta_hi=np.quantile(diff, .975),
                         p_fastest=fastest[i]))
    df = pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)
    df.attrs.update(seed=seed, n_sims=n_sims, race_id=race.race_id, draws=tyre.n_draws,
                    share_sims_with_sc=float((D["status"] == 2).any(1).mean()),
                    share_sims_with_vsc=float((D["status"] == 1).any(1).mean()))
    return df
