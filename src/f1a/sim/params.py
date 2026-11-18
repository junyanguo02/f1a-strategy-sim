"""轮胎与圈速参数：从 tyre v3.1 后验读取，或手工构造。

圈时（绿旗、相对本站基础圈时）：
    k0[c] + gamma*rem_c + beta[c]*(age-10) + kappa[c]*(age-10)^2 + u_stint + sigma*t(nu)
rem_c = 剩余圈数 - 本站剩余圈数均值（与拟合时一致：用 model_laps 里该站的均值）。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

COMPOUNDS = ("HARD", "MEDIUM", "SOFT")
AGE_REF = 10.0


def race_index_map(model_laps: pd.DataFrame, years) -> list[str]:
    """重建拟合时的 race_i → race_id 映射。

    fit_tyre_v3_1.prepare() 先按 years 过滤、再左连接交通特征（保持行序），然后 pd.factorize(race_id)，
    即按首次出现顺序编号。这里用同样规则，不 import 拟合脚本（避免 JAX 依赖）。
    """
    df = model_laps[model_laps["year"].isin(list(years))].reset_index(drop=True)
    codes, uniques = pd.factorize(df["race_id"])
    return [str(u) for u in uniques]


MIN_LAPS, MIN_STINTS = 40, 3   # 与 fit_tyre_v3_1.py 相同：低于此门槛的“站×配方”格子在拟合中用赛季均值代替


def estimated_compounds(model_laps: pd.DataFrame, race_id: str) -> list[str]:
    """该站在拟合里有独立估计（非赛季均值代替）的配方。只用这些配方做策略推荐更可靠：
    样本太少的配方，模型给的是赛季平均表现，可能远好于它在该站的真实表现（例如 2024 拉斯维加斯的软胎）。"""
    r = model_laps[model_laps.race_id == race_id]
    g = r.groupby("compound").agg(n=("lap", "size"), s=("stint_uid", "nunique"))
    ok = g[(g.n >= MIN_LAPS) & (g.s >= MIN_STINTS)].index
    return [c for c in COMPOUNDS if c in ok]


@dataclass
class TyreParams:
    """形状约定：D 为后验抽样数（手工构造时 D=1）。"""
    k0: np.ndarray        # (D, 3)，相对 HARD
    beta: np.ndarray      # (D, 3)，秒/圈
    kappa: np.ndarray     # (D, 3)
    gamma: np.ndarray     # (D,)
    sigma: np.ndarray     # (D,)  圈速噪声尺度（该车手；未指定车手时取车手中位数）
    nu: np.ndarray        # (D,)
    sd_stint: np.ndarray  # (D,)
    rem_mean: float = 0.0  # 拟合时该站剩余圈数均值，用于中心化
    meta: dict = field(default_factory=dict)

    @property
    def n_draws(self) -> int:
        return int(self.gamma.shape[0])

    @classmethod
    def from_dict(cls, k0, beta, kappa, gamma, sigma, nu=30.0, sd_stint=0.0, rem_mean=0.0):
        a = lambda v, n: np.asarray(v, dtype=float).reshape(1, n) if n else np.asarray([v], dtype=float)
        return cls(k0=a(k0, 3), beta=a(beta, 3), kappa=a(kappa, 3), gamma=a(gamma, 0), sigma=a(sigma, 0),
                   nu=a(nu, 0), sd_stint=a(sd_stint, 0), rem_mean=float(rem_mean), meta={"source": "manual"})

    @classmethod
    def from_posterior(cls, npz_path, race_id: str, years, model_laps_path, n_draws: int | None = None,
                       driver_sigma_index: int | None = None, seed: int = 0):
        z = np.load(npz_path)
        ml = pd.read_parquet(model_laps_path)
        races = race_index_map(ml, years)
        if len(races) != z["k0"].shape[1]:
            raise ValueError(f"站数不一致：数据 {len(races)}，后验 {z['k0'].shape[1]}")
        if race_id not in races:
            raise KeyError(f"{race_id} 不在 {list(years)} 的拟合数据里")
        r = races.index(race_id)
        D = z["gamma"].shape[0]
        idx = np.arange(D)
        if n_draws is not None and n_draws < D:
            idx = np.sort(np.random.default_rng(seed).choice(D, n_draws, replace=False))
        sig = z["sigma"][idx]
        sigma = sig[:, driver_sigma_index] if driver_sigma_index is not None else np.median(sig, axis=1)
        rem_mean = float(ml.loc[ml.race_id == race_id, "laps_remaining"].mean())
        return cls(k0=z["k0"][idx, r, :], beta=z["beta"][idx, r, :], kappa=z["kappa"][idx, :],
                   gamma=z["gamma"][idx, r], sigma=sigma, nu=z["nu"][idx], sd_stint=z["sd_stint"][idx],
                   rem_mean=rem_mean, meta={"source": str(npz_path), "race_id": race_id, "race_index": r,
                                            "years": list(years), "draws": len(idx)})

    @classmethod
    def from_prerace(cls, samples, previous_laps, circuit, n_laps, n_draws, seed):
        """Read race-level or season-level draws from the specified prior posterior."""
        from .prerace import prior_tyre
        return prior_tyre(samples, previous_laps, circuit, n_laps, n_draws, seed)
