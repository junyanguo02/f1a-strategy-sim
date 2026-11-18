"""T032 单元测试：聚类 SE、自助、置换检验、分位区间。

在项目根目录执行：

    .venv/bin/pytest -q -p no:cacheprovider analysis/extra_stop_robustness/test_extra_stop_robustness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_extra_stop_robustness as mod  # noqa: E402


def test_ols_cluster():
    rng = np.random.default_rng(6)
    x = rng.normal(0, 1, 200)
    y = 1.2 * x + rng.normal(0, 0.2, 200)
    X = np.column_stack([np.ones(200), x])
    g = np.repeat(np.arange(20), 10)
    fit = mod.ols_cluster(y, X, g)
    assert fit["n_clusters"] == 20
    assert abs(fit["beta"][1] - 1.2) < 0.1
    assert fit["se"][1] > 0


def test_percentile_ci():
    draws = np.arange(1, 101, dtype=float)
    lo, hi = mod.percentile_ci(draws)
    assert abs(lo - 3.475) < 0.1
    assert abs(hi - 97.525) < 0.1


def test_cluster_bootstrap_deterministic():
    df = pd.DataFrame({"race_key": ["a"] * 3 + ["b"] * 3,
                       "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    d1 = mod.cluster_bootstrap(df, "y", n_boot=50, seed=7)
    d2 = mod.cluster_bootstrap(df, "y", n_boot=50, seed=7)
    assert np.allclose(d1, d2)
    assert 3.0 < d1.mean() < 4.0


def test_permutation_positive_effect():
    rng = np.random.default_rng(8)
    y = rng.normal(10, 5, 60)  # 明显正效应
    df = pd.DataFrame({"race_key": np.repeat(np.arange(20), 3), "y": y})
    out = mod.permutation_test(df, "y", n_perm=300, seed=9)
    assert out["p_one_sided"] < 0.05
    assert out["obs"] == pytest.approx(y.mean())


def test_team_tier_mapping_values():
    assert mod.TEAM_MAP["Red Bull Racing"] == "Red Bull"
    assert mod.TEAM_MAP["RB"] == "RB F1 Team"
    assert mod.TEAM_MAP["Kick Sauber"] == "Sauber"
