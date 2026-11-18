"""在给定配方顺序下网格搜索进站圈（共同随机数，比较公平）。"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from .engine import common_draws, simulate
from .strategy import Strategy


def optimize_stops(race, tyre, sequences, n_sims: int = 2000, seed: int = 0, step: int = 1,
                   min_stint: int = 8, objective: str = "mean", max_stint: int | None = None) -> pd.DataFrame:
    """sequences：配方顺序列表，例如 [("MEDIUM","HARD"), ("SOFT","MEDIUM","SOFT")]。
    对每个顺序枚举所有满足最短段长 min_stint（以及可选的最长段长 max_stint，例如轮胎供应商规定的单套圈数上限）的进站圈组合（步长 step），按 objective（mean 或 cvar95）选最优。
    返回每个顺序的最优进站圈及其指标，所有候选共用同一组随机数。"""
    if objective not in ("mean", "cvar95"):
        raise ValueError("objective 必须是 mean 或 cvar95")
    N = race.total_laps
    D = common_draws(race, tyre, n_sims, seed, max_stints=max(len(s) for s in sequences) + 1)
    rows = []
    for seq in sequences:
        k = len(seq) - 1
        grid = range(min_stint, N - min_stint + 1, step)
        best = None
        for laps in itertools.combinations(grid, k):
            if any(b - a < min_stint for a, b in zip(laps, laps[1:])):
                continue
            if max_stint is not None:
                bounds = (0,) + laps + (N,)
                if any(b - a > max_stint for a, b in zip(bounds, bounds[1:])):
                    continue
            s = Strategy("-".join(seq), seq[0], list(zip(laps, seq[1:])))
            t = simulate(race, tyre, s, draws=D)["total"]
            q = np.quantile(t, .95)
            score = t.mean() if objective == "mean" else t[t >= q].mean()
            if best is None or score < best[0]:
                best = (score, laps, t.mean(), t[t >= q].mean())
        if best is None:
            rows.append(dict(sequence="-".join(c[0] for c in seq), stops=None, mean=np.nan, cvar95=np.nan, objective=objective)); continue
        rows.append(dict(sequence="-".join(c[0] for c in seq), stops=best[1], mean=best[2], cvar95=best[3], objective=objective))
    return pd.DataFrame(rows).sort_values("mean" if objective == "mean" else "cvar95").reset_index(drop=True)
