"""T031 单元测试：分组优先级、统计摘要、车距计数、stint 定位。

在项目根目录执行：

    .venv/bin/pytest -q -p no:cacheprovider analysis/extra_stop/test_extra_stop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_extra_stop as mod  # noqa: E402


def test_mean_stat():
    st = mod.mean_stat([1.0, 2.0, 3.0, 100.0])
    assert st["n"] == 4
    assert st["mean"] == pytest.approx(26.5)
    assert st["median"] == pytest.approx(2.5)
    assert mod.mean_stat([])["n"] == 0


def test_assign_groups_priority():
    df = pd.DataFrame([
        {"slope_diff": 0.05, "rise_diff": np.nan, "stop_abnormal": True,
         "blocked_before": True, "tactical": False},   # 轮胎优先
        {"slope_diff": 0.0, "rise_diff": 0.2, "stop_abnormal": True,
         "blocked_before": True, "tactical": False},   # 停车异常
        {"slope_diff": np.nan, "rise_diff": np.nan, "stop_abnormal": False,
         "blocked_before": True, "tactical": False},   # 被堵
        {"slope_diff": np.nan, "rise_diff": np.nan, "stop_abnormal": False,
         "blocked_before": False, "tactical": True},   # 战术
        {"slope_diff": -0.1, "rise_diff": -1.0, "stop_abnormal": False,
         "blocked_before": False, "tactical": False},  # 主动
    ])
    out = mod.assign_groups(df)
    assert out["group"].tolist() == [
        "轮胎衰减更快（被迫）", "停车异常（被迫）", "被堵 / 战术",
        "被堵 / 战术", "无明显原因（主动）"]


def test_count_within():
    cum = {"A": 100.0, "B": 101.0, "C": 98.5, "D": 103.0, "E": 90.0}
    ahead, behind, total = mod.count_within(cum, "A", 100.0, 2.0)
    assert ahead == 1 and behind == 1 and total == 2


def test_stint_of_lap_and_pit_laps():
    stints = pd.DataFrame([
        {"year": 2023, "round": 1, "driver": "VER", "stint": 1,
         "lap_start": 1, "lap_end": 14},
        {"year": 2023, "round": 1, "driver": "VER", "stint": 2,
         "lap_start": 15, "lap_end": 36},
        {"year": 2023, "round": 1, "driver": "VER", "stint": 3,
         "lap_start": 37, "lap_end": 57},
    ])
    row = mod.stint_of_lap(stints, 2023, 1, "VER", 30)
    assert int(row["stint"]) == 2
    assert mod.stint_of_lap(stints, 2023, 1, "VER", 60) is None
    assert mod.pit_laps(stints, 2023, 1, "VER") == [14, 36]


def test_engine_helpers_not_imported():
    # 本单不调用模拟器，确认模块仍可独立导入
    assert hasattr(mod, "build_metrics")
