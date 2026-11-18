"""用项目真实数据的集成测试；数据缺失时跳过。"""
from pathlib import Path
import numpy as np
import pytest
from f1a.sim import RaceConfig, TyreParams, race_index_map, Strategy, evaluate

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "models" / "tyre_v3" / "outputs31_2024" / "posterior_core.npz"
ML = ROOT / "data" / "processed" / "model_laps.parquet"
need = pytest.mark.skipif(not (NPZ.exists() and ML.exists()), reason="缺少 v3.1 2024 后验或 model_laps")


@need
def test_race_index_map_matches_posterior():
    import pandas as pd
    z = np.load(NPZ)
    races = race_index_map(pd.read_parquet(ML), [2024])
    assert len(races) == z["k0"].shape[1] == z["gamma"].shape[1]


@need
def test_project_race_runs_and_is_sane():
    race = RaceConfig.from_project(ROOT, "2024-10")
    ty = TyreParams.from_posterior(NPZ, "2024-10", [2024], ML, n_draws=100)
    assert race.total_laps == 66 and 70 < race.base_lap_s < 100
    assert 1.0 < race.lap_mult["SC"] < 2.5 and 1.0 < race.lap_mult["VSC"] < 2.0
    df = evaluate(race, ty, [Strategy("MH", "MEDIUM", [(25, "HARD")]), Strategy("SMS", "SOFT", [(16, "MEDIUM"), (46, "SOFT")])],
                  n_sims=500, seed=0)
    # 总用时应接近 圈数×基础圈时 + 进站损失（量级检查）
    assert np.all(np.abs(df["mean"] - 66 * race.base_lap_s) < 400)


@need
def test_backtest_counts_cars_using_compound():
    import pandas as pd
    from f1a.sim.backtest import backtest_race

    entrants = pd.DataFrame(dict(year=2024, round=10, finisher=True, wet_tyre_used=False,
                                 strategy=["HSS", "H", "H", "H", "H"]))
    result = backtest_race(ROOT, "2024-10", 2024, entrants, n_sims=2, step=10)
    assert result["skipped"].endswith("['HARD']")


@need
def test_backtest_reports_circuit_id():
    import pandas as pd
    from f1a.sim.backtest import backtest_race

    entrants = pd.DataFrame(dict(year=2024, round=10, finisher=True, wet_tyre_used=False,
                                 strategy=["MH"] * 5, finish_position=range(1, 6),
                                 n_pit_stops=1, pit_laps="30"))
    result = backtest_race(ROOT, "2024-10", 2024, entrants, n_sims=5, step=10, max_stops=1)
    assert result["circuit"] == "catalunya"
