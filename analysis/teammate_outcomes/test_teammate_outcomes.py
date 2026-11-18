"""T028 单元测试：聚类 SE、统计摘要、额外进站识别、策略构造、格式函数。

在项目根目录执行：

    .venv/bin/pytest -q -p no:cacheprovider analysis/teammate_outcomes/test_teammate_outcomes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_teammate_outcomes as mod  # noqa: E402


def test_ols_cluster():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 200)
    y = 1.5 * x + rng.normal(0, 0.2, 200)
    X = np.column_stack([np.ones(200), x])
    g = np.repeat(np.arange(20), 10)
    fit = mod.ols_cluster(y, X, g)
    assert fit["n_clusters"] == 20
    assert abs(fit["beta"][1] - 1.5) < 0.1
    assert fit["se"][1] > 0


def test_mean_stat():
    st = mod.mean_stat([1.0, 2.0, 3.0, 100.0])
    assert st["n"] == 4
    assert st["mean"] == pytest.approx(26.5)
    assert st["median"] == pytest.approx(2.5)
    assert st["sd"] > 0
    assert mod.mean_stat([])["n"] == 0


def test_find_extra_stop():
    # 多出的一次是最后一停
    assert mod.find_extra_stop([11, 29, 54], [12, 32]) == 54
    # 多出的一次是早停
    assert mod.find_extra_stop([5, 25, 42], [25, 42]) == 5
    # 无法唯一判断
    assert mod.find_extra_stop([5, 40], [20, 55]) is None
    # 正常一对一
    assert mod.find_extra_stop([14, 36], [11]) == 36


def test_strategy_from_row():
    row = pd.Series({
        "driver": "VER", "n_stints": 3, "total_laps": 57,
        "start_tyre_age": 0.0,
        "stint1_compound": "SOFT", "stint2_compound": "HARD", "stint3_compound": "HARD",
        "stint1_end": 14, "stint2_end": 36,
    })
    st = mod.strategy_from_row(row)
    assert st is not None
    assert st.start_compound == "SOFT"
    assert st.stops == [(14, "HARD"), (36, "HARD")]
    assert st.start_age == 1
    # 不合法（同一配方）返回 None
    bad = row.copy()
    bad["stint2_compound"] = "SOFT"
    bad["stint3_compound"] = "SOFT"
    assert mod.strategy_from_row(bad) is None


def test_fmt_p():
    assert mod.fmt_p(0.0004) == "<0.001"
    assert mod.fmt_p(0.2) == "0.200"
    assert mod.fmt_p(np.nan) == "—"
