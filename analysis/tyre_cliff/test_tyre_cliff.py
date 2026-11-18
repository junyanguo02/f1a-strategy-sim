"""T021 单元测试：设计矩阵、分段/二次拟合、拐点交叉验证、样本筛选、分站格子。

在项目根目录执行：

    .venv/bin/pytest -q -p no:cacheprovider analysis/tyre_cliff/test_tyre_cliff.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_tyre_cliff as mod  # noqa: E402


def test_fmt_p():
    assert mod.fmt_p(0.0004) == "<0.001"
    assert mod.fmt_p(0.0123) == "0.012"
    assert mod.fmt_p(np.nan) == "—"


def test_design_piecewise_continuous():
    age = np.array([1.0, 5.0, 12.0, 20.0, 30.0])
    X = mod.design(age, 15.0)
    assert X.shape == (5, 3)
    assert np.allclose(X[:, 0], 1.0)
    assert np.allclose(X[:, 1], age)
    assert np.allclose(X[:, 2], [0, 0, 0, 5, 15])


def test_fit_piecewise_recovers_known_slopes():
    rng = np.random.default_rng(7)
    stints, rows = [], []
    for st in range(120):
        n = rng.integers(18, 31)
        for age in range(1, int(n) + 1):
            # 前段 0.05 s/lap，k=20 之后 0.15 s/lap
            y = 0.05 * age + 0.10 * max(0, age - 20)
            y += rng.normal(0, 0.02)
            rows.append({"compound": "HARD", "tyre_life": age, "y": y,
                         "stint_uid": f"s{st}"})
    df = pd.DataFrame(rows)
    out = mod.fit_piecewise(df, "HARD", 20)
    assert abs(out["early_slope"] - 0.05) < 0.005
    assert abs(out["late_slope"] - 0.15) < 0.01
    assert abs(out["diff_late_minus_early"] - 0.10) < 0.01
    assert out["diff_p_one_sided"] < 0.001


def test_quadratic_kappa_sign():
    rng = np.random.default_rng(8)
    rows = []
    for st in range(60):
        for age in range(1, 30):
            y = 0.05 * age + 0.002 * age ** 2 + rng.normal(0, 0.02)
            rows.append({"compound": "SOFT", "tyre_life": age, "y": y,
                         "stint_uid": f"s{st}"})
    df = pd.DataFrame(rows)
    kappa = mod.fit_quadratic(df, "SOFT")["kappa_quadratic"]
    assert abs(kappa - 0.002) < 2e-4


def test_knot_cv_finds_known_break():
    rng = np.random.default_rng(9)
    rows = []
    for st in range(80):
        for age in range(1, 41):
            y = 0.04 * age + 0.20 * max(0, age - 25) + rng.normal(0, 0.05)
            rows.append({"compound": "MEDIUM", "tyre_life": age, "y": y,
                         "stint_uid": f"s{st}"})
    df = pd.DataFrame(rows)
    k, cv = mod.knot_cv(df, "MEDIUM")
    assert abs(k - 25) <= 1, f"knot={k}"
    assert len(cv) == len(mod.KNOT_GRID)


def test_analysis_sample_filters():
    df = pd.DataFrame([
        {"base": 1.0, "start_age": 2, "y": 0.5, "compound": "HARD", "stint_uid": "s1"},
        {"base": 1.0, "start_age": 2, "y": 9.0, "compound": "HARD", "stint_uid": "s2"},    # 异常
        {"base": 1.0, "start_age": 8, "y": 0.5, "compound": "HARD", "stint_uid": "s3"},    # 起始太晚
        {"base": np.nan, "start_age": 2, "y": 0.5, "compound": "HARD", "stint_uid": "s4"},  # 无基准
        {"base": 1.0, "start_age": 2, "y": 0.5, "compound": "WET", "stint_uid": "s5"},      # 非配方
    ])
    sample, excl = mod.analysis_sample(df)
    assert len(sample) == 1
    assert int(excl.loc[excl["项目"].str.contains("起始胎龄"), "圈数"].iloc[0]) == 1
    assert int(excl.loc[excl["项目"].str.contains("计时异常"), "圈数"].iloc[0]) == 1


def test_circuit_cells_thresholds():
    rng = np.random.default_rng(10)
    rows = []
    for circuit, n_st in [(("a", "HARD"), 15), (("b", "HARD"), 3)]:
        for st in range(n_st):
            for age in range(1, 36):
                y = 0.04 * age + 0.08 * max(0, age - 20) + rng.normal(0, 0.05)
                rows.append({"circuit_id": circuit[0], "compound": circuit[1],
                             "tyre_life": age, "y": y,
                             "stint_uid": f"{circuit[0]}{st}"})
    df = pd.DataFrame(rows)
    fits = {"HARD": {"knot": 20}}
    cells = mod.circuit_cells(df, fits)
    assert len(cells) == 1
    assert cells.iloc[0]["circuit_id"] == "a"
    assert abs(cells.iloc[0]["diff_late_minus_early"] - 0.08) < 0.02
