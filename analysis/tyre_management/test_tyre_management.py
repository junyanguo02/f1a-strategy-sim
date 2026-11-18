"""T024 单元测试：去均值、聚类 CI、设计矩阵、格子拟合、格式函数。

在项目根目录执行：

    .venv/bin/pytest -q -p no:cacheprovider analysis/tyre_management/test_tyre_management.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_tyre_management as mod  # noqa: E402


def test_within_demean_matches_groupby():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                       "g": ["x", "x", "x", "y", "y", "y"]})
    out = mod.within_demean(df[["a"]], df["g"])
    assert np.allclose(out[:, 0], [-1, 0, 1, -1, 0, 1])


def test_cluster_mean_ci():
    z = pd.Series([0.1, 0.2, 0.3, 0.4])
    g = pd.Series(["a", "a", "b", "b"])
    st = mod.cluster_mean_ci(z, g)
    assert st["n_laps"] == 4 and st["n_stints"] == 2
    assert st["mean"] == pytest.approx(0.25)
    assert st["ci_lo"] < st["mean"] < st["ci_hi"]


def test_build_design_columns():
    df = pd.DataFrame({"length": [10, 20, 30, 40],
                       "tyre_life": [5, 6, 7, 8],
                       "compound": ["HARD", "MEDIUM", "SOFT", "HARD"],
                       "circuit_id": ["a", "a", "b", "b"]})
    X = mod.build_design(df)
    assert "len10" in X.columns and "age" in X.columns
    assert X["len10"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert X.shape[0] == 4


def test_fit_cell_recovers_known_slope():
    rng = np.random.default_rng(11)
    rows = []
    for st in range(40):
        length = rng.integers(10, 45)
        for age in range(1, length + 1):
            y = 0.05 * (length / 10) + 0.01 * age + rng.normal(0, 0.05)
            rows.append({"length": length, "tyre_life": age, "y": y,
                         "stint_uid": f"s{st}"})
    df = pd.DataFrame(rows)
    r = mod.fit_cell(df)
    assert abs(r["coef_len10"] - 0.05) < 0.02


def test_fmt_p():
    assert mod.fmt_p(0.0004) == "<0.001"
    assert mod.fmt_p(0.03) == "0.030"
    assert mod.fmt_p(np.nan) == "—"


def test_length_bin_labels():
    bins = pd.cut([10, 15, 16, 20, 21, 25, 26, 30, 31, 40, 41],
                  mod.LENGTH_BINS, labels=mod.LENGTH_LABELS)
    got = list(bins.astype(str))
    assert got[0] == "≤15" and got[1] == "≤15" and got[2] == "16–20"
    assert got[-1] == ">40"
