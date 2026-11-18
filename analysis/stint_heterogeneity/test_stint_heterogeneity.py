"""T026 单元测试：OLS 斜率、聚类 SE、方差分解、R²、格式函数。

在项目根目录执行：

    .venv/bin/pytest -q -p no:cacheprovider analysis/stint_heterogeneity/test_stint_heterogeneity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_stint_heterogeneity as mod  # noqa: E402


def test_decompose_known_components():
    rng = np.random.default_rng(3)
    # 两个格子，每格 200 段；真实段效应 sd=0.05，估计误差 sd=0.02
    cells, slopes, ses = [], [], []
    for c, mu in [("a", 0.04), ("b", 0.10)]:
        for _ in range(200):
            true = mu + rng.normal(0, 0.05)
            err = rng.normal(0, 0.02)
            slopes.append(true + err)
            ses.append(0.02)
            cells.append(c)
    d = mod.decompose(np.array(slopes), np.array(ses), np.array(cells))
    assert abs(d["sd_true"] - np.sqrt(0.05 ** 2 + 0.06 ** 2 / 2)) < 0.01
    assert d["sd_between_true"] > 0.035
    assert d["sd_within_true"] > 0.04
    assert 0.019 <= d["sd_err"] <= 0.021
    assert 0.0 <= d["err_share"] < 0.5


def test_decompose_all_error():
    slopes = np.array([0.05, 0.06, 0.07, 0.08])
    ses = np.full(4, 0.10)  # 估计误差远大于观测方差
    cells = np.array(["a", "a", "b", "b"])
    d = mod.decompose(slopes, ses, cells)
    assert d["v_true"] == 0.0
    assert d["sd_true"] == 0.0


def test_ols_cluster():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 200)
    y = 2.0 * x + rng.normal(0, 0.1, 200)
    X = np.column_stack([np.ones(200), x])
    g = np.repeat(np.arange(20), 10)
    res = mod.ols_cluster(y, X, g)
    assert res["n_clusters"] == 20
    assert abs(res["beta"][1] - 2.0) < 0.05
    assert res["se"][1] > 0


def test_ols_slope_recovery():
    age = np.arange(1, 21, dtype=float)
    y = 0.08 * age + 90.0
    slope, se = mod.ols_slope(age, y)
    assert abs(slope - 0.08) < 1e-10
    assert se < 1e-9


def test_r2_ols():
    x = np.linspace(0, 10, 100)
    y = 3 * x + 1
    assert mod.r2_ols(y, x[:, None]) == pytest.approx(1.0)


def test_fmt_p():
    assert mod.fmt_p(0.0001) == "<0.001"
    assert mod.fmt_p(0.25) == "0.250"
    assert mod.fmt_p(np.nan) == "—"
