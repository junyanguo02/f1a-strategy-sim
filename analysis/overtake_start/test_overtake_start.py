"""T023 单元测试：位置/超车规则、Δ 分箱、Wilson 区间、赛道类型、起步筛选。

在项目根目录执行：

    .venv/bin/pytest -q -p no:cacheprovider analysis/overtake_start/test_overtake_start.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_overtake_start as mod  # noqa: E402


def test_delta_bin_boundaries():
    assert mod.delta_bin(-0.1) == "<0"
    assert mod.delta_bin(0.0) == "0–0.25"
    assert mod.delta_bin(0.249) == "0–0.25"
    assert mod.delta_bin(0.25) == "0.25–0.5"
    assert mod.delta_bin(0.5) == "0.5–1"
    assert mod.delta_bin(1.0) == ">1"
    assert mod.delta_bin(np.nan) is None


def test_wilson_ci():
    lo, hi = mod.wilson_ci(0, 100)
    assert lo == pytest.approx(0.0) and hi < 0.06
    lo, hi = mod.wilson_ci(50, 100)
    assert lo < 0.5 < hi
    assert abs((hi - lo) - 2 * 1.96 * 0.05) < 0.01
    lo, hi = mod.wilson_ci(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_track_type():
    assert mod.track_type("monaco") == "街道赛"
    assert mod.track_type("vegas") == "街道赛"
    assert mod.track_type("monza") == "常规赛道"
    assert mod.track_type("albert_park") == "常规赛道"


def test_race_cumulative_and_states():
    rows = []
    for drv, times in [("A", [90, 91, 92, 93]), ("B", [91, 92, 93, 94])]:
        for i, t in enumerate(times):
            rows.append({"driver": drv, "lap": i + 1, "lap_time_s": t})
    g = pd.DataFrame(rows)
    drivers, states = mod.race_cumulative(g)
    assert drivers["A"][4] == pytest.approx(366.0)
    assert states[4]["A"] == 1 and states[4]["B"] == 2
    # 缺圈后不再外推
    g2 = g[~((g["driver"] == "B") & (g["lap"] == 3))]
    drivers2, states2 = mod.race_cumulative(g2)
    assert 4 not in drivers2["B"]
    assert 3 not in drivers2["B"]


def test_pair_blocked():
    flags = {}
    for drv in ("A", "B"):
        for lap in (5, 6):
            flags[(drv, lap)] = {"pit_in": False, "pit_out": False,
                                 "sc": False, "vsc": False,
                                 "clean": True, "lap_time": 90.0}
    assert mod.pair_blocked(flags, "A", "B", 5) is False
    flags[("B", 6)]["pit_out"] = True
    assert mod.pair_blocked(flags, "A", "B", 5) is True
    flags[("B", 6)]["pit_out"] = False
    flags[("A", 5)]["sc"] = True
    assert mod.pair_blocked(flags, "A", "B", 5) is True
    del flags[("A", 6)]
    assert mod.pair_blocked(flags, "A", "B", 5) is True


def test_mean_stat():
    st = mod.mean_stat([1.0, 2.0, 3.0, 100.0])
    assert st["n"] == 4
    assert st["mean"] == pytest.approx(26.5)
    assert st["median"] == pytest.approx(2.5)
    assert mod.mean_stat([])["n"] == 0
