"""比赛设置：总圈数、基础圈时、进站损失样本、SC/VSC 出现率与持续圈数、FCY 圈时倍数。"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

FCY_TYPES = ("VSC", "SC")          # 状态编码：0 绿旗，1 VSC，2 SC
N_PROGRESS_BINS = 5
# T018 的 dt 分组 → 模拟器里“进站发生在 FCY 段的第几圈”
PHASE_GROUPS = {1: ("宣布后0-60秒", "60秒-1圈"), 2: ("1-2圈",), 3: ("2圈以后",)}


# The batch entry point selects one filter for nested pre- and post-race builds.
_batch_filter = ContextVar("batch_pit_loss_filter", default=(None, (10., 60.)))


@contextmanager
def pit_loss_batch(mode, bounds=(10., 60.)):
    """Apply a filter across nested race constructors in one CLI run."""
    token = _batch_filter.set((mode, bounds))
    try:
        yield
    finally:
        _batch_filter.reset(token)


def pit_tables(root, mode=None, bounds=(10., 60.), *, history=None):
    """Preserve source row order in both the normal and the T018 phase pools."""
    root = Path(root)
    if mode is None:
        mode, bounds = _batch_filter.get()
    if mode not in (None, "range", "clean"):
        raise ValueError(f"unknown pit_loss_filter: {mode}")
    path = root / "analysis/fcy_inputs/pit_loss_relative.parquet"
    if mode == "clean":
        path = root / "analysis/pitloss_audit/outputs/pit_loss_relative_clean.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"T042 clean pit-loss table missing: {path}")
    pl = pd.read_parquet(path)
    det = pd.read_csv(root / "analysis/sc_pit_timing/outputs/pit_match_detail.csv")
    if history is not None:
        pl, det = history(pl), history(det)
    if mode == "range":
        lo, hi = bounds
        if not np.isfinite([lo, hi]).all() or lo > hi:
            raise ValueError("pit_loss_bounds must be finite and ordered")
        return (pl.loc[pl.relative_loss_s.between(lo, hi)],
                det.loc[det.relative_loss_s.between(lo, hi)])
    if mode == "clean":
        keys = ["year", "round", "driver", "lap"]
        keep = pl["clean_keep"]
        if not pd.api.types.is_bool_dtype(keep) or keep.isna().any():
            raise ValueError("T042 clean_keep must be boolean and non-null")
        flags = pl.set_index(keys, verify_integrity=True).clean_keep
        selected = flags.reindex(pd.MultiIndex.from_frame(det[keys]))
        if selected.isna().any():
            raise ValueError("T018 pit without matching T042 clean_keep decision")
        return pl.loc[keep], det.loc[selected.to_numpy(dtype=bool)]
    return pl, det


@dataclass
class RaceConfig:
    race_id: str
    total_laps: int
    base_lap_s: float
    pit_normal: np.ndarray                      # 正常条件相对进站损失样本（秒）
    pit_fcy: dict                               # {(类型, 阶段1/2/3): 样本数组}
    lap1_prob: dict                             # {类型: 第 1 圈发生概率}
    hazard: dict                                # {类型: 长度 N_PROGRESS_BINS 的每圈出现率}，第 2 圈起
    durations: dict                             # {类型: 持续圈数经验样本}
    lap_mult: dict = field(default_factory=lambda: {"VSC": 1.4, "SC": 1.6})
    notes: list = field(default_factory=list)

    def fcy_rate_scale(self, factor: float) -> "RaceConfig":
        """把 SC/VSC 出现率整体乘以 factor（做敏感性分析用）。"""
        import copy
        c = copy.deepcopy(self)
        c.lap1_prob = {k: min(1.0, v * factor) for k, v in c.lap1_prob.items()}
        c.hazard = {k: np.minimum(1.0, v * factor) for k, v in c.hazard.items()}
        c.notes.append(f"FCY 出现率 ×{factor}")
        return c

    @classmethod
    def from_project(cls, root, race_id: str, min_circuit_pits: int = 10, fcy_by_circuit: bool = False,
                     prior_laps: float = 300.0, *, pit_loss_filter=None, pit_loss_bounds=(10., 60.)) -> "RaceConfig":
        """Use the original pools, inclusive range bounds, or the T042 clean_keep flag."""
        root = Path(root)
        P = root / "data" / "processed"
        ml = pd.read_parquet(P / "model_laps.parquet")
        r = ml[ml.race_id == race_id]
        if r.empty:
            raise KeyError(race_id)
        circuit = r.circuit_id.iloc[0]
        pl, det = pit_tables(root, pit_loss_filter, pit_loss_bounds)
        ep = pd.read_parquet(P / "fcy_episodes.parquet")
        rs = pd.read_csv(P / "fcy_race_summary.csv")
        L = pd.read_parquet(P / "laps_pit_filled.parquet")
        return cls.from_samples(race_id, circuit, int(r.total_laps.iloc[0]), float(r.lap_time_s.median()),
                                pl, det, ep, rs, L, min_circuit_pits, fcy_by_circuit, prior_laps)

    @classmethod
    def from_samples(cls, race_id, circuit, total_laps, base_lap_s, pl, det, ep, rs, L,
                     min_circuit_pits=10, fcy_by_circuit=False, prior_laps=300.0):
        """Use precisely the sample population supplied by the caller."""
        notes = []
        pl = pl[pl.relative_usable & (pl.condition == "正常")]
        own = pl[pl.circuit_id == circuit].relative_loss_s.to_numpy(float)
        if len(own) >= min_circuit_pits:
            pit_normal = own; notes.append(f"正常进站损失：本赛道 {len(own)} 个样本")
        else:
            pit_normal = pl.relative_loss_s.to_numpy(float); notes.append(f"正常进站损失：本赛道仅 {len(own)} 个，改用全体 {len(pit_normal)} 个")
        # FCY 下进站损失（T018，全赛道合并）
        det = det[det.dt_group.notna() & det.relative_loss_s.notna()]
        pit_fcy = {}
        for t in FCY_TYPES:
            for ph, groups in PHASE_GROUPS.items():
                v = det[(det.condition == t) & det.dt_group.isin(groups)].relative_loss_s.to_numpy(float)
                pit_fcy[(t, ph)] = v
        notes.append("FCY 下进站损失：T018 按距宣布时间分组，全赛道合并")
        # 出现率与持续圈数（全赛道基线）
        n_races = len(rs)
        lap1_prob, hazard, durations = {}, {}, {}
        edges = np.linspace(0, 1, N_PROGRESS_BINS + 1)
        # 每个进度段的“风险圈数”：各站第 2 圈到最后一圈按进度归段
        exposure = np.zeros(N_PROGRESS_BINS)
        for tl in rs.total_laps:
            laps = np.arange(2, int(tl) + 1)
            b = np.clip(np.digitize(laps / tl, edges[1:-1]), 0, N_PROGRESS_BINS - 1)
            exposure += np.bincount(b, minlength=N_PROGRESS_BINS)
        for t in FCY_TYPES:
            e = ep[ep.type == t]
            lap1_prob[t] = float((e.start_lap == 1).sum() / n_races)
            later = e[e.start_lap > 1]
            b = np.clip(np.digitize(later.start_lap / later.total_laps, edges[1:-1]), 0, N_PROGRESS_BINS - 1)
            hazard[t] = np.bincount(b, minlength=N_PROGRESS_BINS) / exposure
            durations[t] = e.n_laps.to_numpy(int)
        notes.append(f"SC/VSC 出现率与持续圈数：{n_races} 场全赛道基线（含 {int(rs.is_wet.sum())} 场湿地）")
        if fcy_by_circuit:
            # 按赛道收缩估计：赛道的“第 2 圈起每圈出现率”= (该赛道出现次数 + prior_laps×基线率) / (该赛道风险圈数 + prior_laps)，
            # 再按基线的进度形状分配到 5 段；第 1 圈概率同样按 (次数 + 2×基线) / (场数 + 2) 收缩。
            # 注意：含本场在内（事前回测时有轻微信息泄漏，样本小，影响有限，报告中注明）。
            rc = rs[rs.circuit_id == circuit]
            laps_c = float((rc.total_laps - 1).sum())
            for t in FCY_TYPES:
                e = ep[(ep.type == t) & (ep.circuit_id == circuit)]
                base_rate = float((hazard[t] * exposure).sum() / exposure.sum())
                n_c = float((e.start_lap > 1).sum())
                rate_c = (n_c + prior_laps * base_rate) / (laps_c + prior_laps)
                hazard[t] = hazard[t] * (rate_c / base_rate if base_rate > 0 else 1.0)
                lap1_prob[t] = float(((e.start_lap == 1).sum() + 2 * lap1_prob[t]) / (len(rc) + 2))
            notes.append(f"按赛道收缩的 SC/VSC 出现率（{circuit}，{len(rc)} 场，先验 {prior_laps:.0f} 圈）")
        # FCY 圈时倍数：从数据估计（非进出站圈）
        med = L[L.is_clean].groupby(["year", "round"]).lap_time_s.median().rename("med")
        L = L.join(med, on=["year", "round"])
        normal_lap = ~L.pit_in_filled.fillna(False) & ~L.pit_out.fillna(False) & (L.lap > 1)
        lap_mult = {}
        for t, col in (("SC", "safety_car"), ("VSC", "virtual_safety_car")):
            x = L[normal_lap & L[col].fillna(False)]
            ratio = (x.lap_time_s / x.med).dropna()
            ratio = ratio[(ratio > 1.0) & (ratio < 3.0)]
            lap_mult[t] = float(ratio.median()) if len(ratio) > 50 else {"SC": 1.6, "VSC": 1.4}[t]
        notes.append(f"FCY 圈时倍数（数据中位数，整段平均含部署/结束过渡圈）：SC {lap_mult['SC']:.3f}，VSC {lap_mult['VSC']:.3f}")
        return cls(race_id=race_id, total_laps=total_laps, base_lap_s=base_lap_s,
                   pit_normal=pit_normal, pit_fcy=pit_fcy, lap1_prob=lap1_prob, hazard=hazard,
                   durations=durations, lap_mult=lap_mult, notes=notes)

    @classmethod
    def from_prerace(cls, root, schedule, cutoff, race_id, circuit, n_laps, base, historical_laps, by_circuit=False, *,
                     pit_loss_filter=None, pit_loss_bounds=(10., 60.)):
        """Build pit/FCY distributions only from completed races before cutoff."""
        from .prerace import Unavailable, before
        processed = root / "data/processed"

        def history(frame):
            return before(frame, schedule, cutoff)

        rs = history(pd.read_csv(processed / "fcy_race_summary.csv"))
        ep = history(pd.read_parquet(processed / "fcy_episodes.parquet"))
        if rs.empty or not {"SC","VSC"}.issubset(set(ep.type)):
            raise Unavailable("no_fcy_history")
        pl, det = pit_tables(root, pit_loss_filter, pit_loss_bounds, history=history)
        race = cls.from_samples(race_id, circuit, n_laps, float(base),
                                pl, det,
                                ep, rs, history(historical_laps), fcy_by_circuit=by_circuit)
        if not len(race.pit_normal) or any(not len(v) for v in race.pit_fcy.values()):
            raise Unavailable("no_pit_history")
        return race
