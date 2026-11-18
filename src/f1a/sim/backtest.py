"""事前回测：模拟器在赛前会推荐什么策略，与前十名车实际采用的策略对照。

“事前”指 SC/VSC 用全赛道基线随机生成，不使用该场实际发生的安全车。所以有安全车改变策略的比赛，
模拟器推荐与实际不一致是预期之内的；结果表里给出该场 SC/VSC 次数，便于分开看。
"""
from __future__ import annotations
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from .params import TyreParams, COMPOUNDS, estimated_compounds
from .race import RaceConfig
from .optimize import optimize_stops

_ABBR = {"H": "HARD", "M": "MEDIUM", "S": "SOFT"}


def candidate_sequences(allowed, max_stops: int = 2):
    seqs = []
    for k in range(1, max_stops + 1):
        for seq in itertools.product(allowed, repeat=k + 1):
            if len(set(seq)) >= 2:
                seqs.append(tuple(seq))
    return seqs


def backtest_race(root, race_id: str, season: int, strategies: pd.DataFrame, n_sims: int = 1000,
                  n_draws: int = 200, step: int = 2, seed: int = 0, max_stops: int = 2,
                  fcy_by_circuit: bool = False) -> dict:
    root = Path(root)
    yr, rd = map(int, race_id.split("-"))
    s = strategies[(strategies.year == yr) & (strategies["round"] == rd) & strategies.finisher.astype(bool)
                   & ~strategies.wet_tyre_used.astype(bool)].copy()
    s = s[s.strategy.str.fullmatch(r"[HMS]+")]
    if len(s) < 5:
        return dict(race_id=race_id, skipped="干地完赛车不足 5 辆")
    ml = pd.read_parquet(root / "data" / "processed" / "model_laps.parquet")
    est = estimated_compounds(ml, race_id)
    # 候选配方：至少 2 辆干地完赛车用过，且该站拟合有独立估计（避免样本太少的配方沿用赛季均值而被高估）
    allowed = [c for c in COMPOUNDS if s.strategy.str.contains(c[0], regex=False).sum() >= 2 and c in est]
    if len(allowed) < 2:
        return dict(race_id=race_id, skipped=f"有独立估计的可用配方不足两种：{allowed}")
    race = RaceConfig.from_project(root, race_id, fcy_by_circuit=fcy_by_circuit)
    npz = root / "models" / "tyre_v3" / f"outputs31_{season}" / "posterior_core.npz"
    tyre = TyreParams.from_posterior(npz, race_id, [season], root / "data" / "processed" / "model_laps.parquet",
                                     n_draws=n_draws, seed=seed)
    seqs = candidate_sequences(allowed, max_stops)
    opt = optimize_stops(race, tyre, seqs, n_sims=n_sims, seed=seed, step=step)
    circuit_id = ml.loc[ml.race_id == race_id, "circuit_id"].iloc[0]
    return score_single(opt, s, race, circuit_id, allowed, len(seqs))


def score_single(opt, s, race, circuit_id, allowed, sequence_count):
    race_id = race.race_id
    opt = opt.copy()
    opt["sequence"] = opt.sequence.str.replace("-", "", regex=False)
    opt["n_stops"] = opt.sequence.str.len() - 1
    opt["multiset"] = opt.sequence.map(lambda x: "".join(sorted(x)))
    best = opt.iloc[0]
    top10 = s.sort_values("finish_position").head(10)
    dom = top10.strategy.value_counts()
    dom_seq, dom_share = dom.index[0], dom.iloc[0] / len(top10)
    modal_stops = int(top10.n_pit_stops.mode().iloc[0])
    dom_ms = "".join(sorted(dom_seq))
    in_opt = opt[opt.sequence == dom_seq]
    # 模型里配方顺序不影响总用时（圈时各项与该段在比赛中的位置无关，只有进站圈对称地变化），
    # 所以主要对照“配方组合”（不计顺序）：例如 M-H 与 H-M 视为同一组合
    ms_best = opt.groupby("multiset")["mean"].min().sort_values()
    ms_rank = int(list(ms_best.index).index(dom_ms)) + 1 if dom_ms in ms_best.index else None
    ms_gap = float(ms_best.get(dom_ms, np.nan) - ms_best.iloc[0])
    dom_rank = int(in_opt.index[0]) + 1 if len(in_opt) else None
    dom_gap = float(in_opt["mean"].iloc[0] - best["mean"]) if len(in_opt) else np.nan
    best_by_stops = opt.groupby("n_stops")["mean"].min()
    # 实际前十名中与模拟器最优顺序相同的车，第一次进站圈中位数
    same = top10[top10.strategy.map(lambda x: "".join(sorted(x))) == best.multiset]
    first = [int(str(pl).split(";")[0]) for pl in same.pit_laps if str(pl) not in ("", "nan", "None")]
    first_pit_actual = float(np.median(first)) if first else np.nan
    return dict(race_id=race_id, circuit=circuit_id, total_laps=race.total_laps, allowed="".join(c[0] for c in allowed),
                model_best=best.sequence, model_best_stops=tuple(int(x) for x in best.stops),
                model_n_stops=int(best.n_stops),
                one_vs_two_stop_gap=float(best_by_stops.get(1, np.nan) - best_by_stops.get(2, np.nan)),
                top10_dominant=dom_seq, top10_dominant_share=float(dom_share), top10_modal_stops=modal_stops,
                stops_match=int(best.n_stops) == modal_stops, sequence_match=best.sequence == dom_seq,
                combo_match=best.multiset == dom_ms, dominant_combo_rank=ms_rank, dominant_combo_gap_s=ms_gap,
                dominant_rank_in_model=dom_rank, dominant_gap_s=dom_gap,
                n_top10_on_model_best=int(len(same)), first_pit_model=int(best.stops[0]),
                first_pit_actual_same_combo=first_pit_actual, n_sequences=sequence_count)
