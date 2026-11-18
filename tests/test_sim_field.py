"""多车模拟器单元测试（手写参数）。"""
import numpy as np
import pytest
from f1a.sim import RaceConfig, TyreParams, Strategy, simulate
from f1a.sim.field import Car, FieldParams, simulate_field, evaluate_ego


def toy_race(n=20, lap1_sc=0.0):
    return RaceConfig(race_id="toy", total_laps=n, base_lap_s=90.0, pit_normal=np.array([20.0]),
                      pit_fcy={(t, p): np.array([10.0]) for t in ("SC", "VSC") for p in (1, 2, 3)},
                      lap1_prob={"SC": lap1_sc, "VSC": 0.0}, hazard={"SC": np.zeros(5), "VSC": np.zeros(5)},
                      durations={"SC": np.array([3]), "VSC": np.array([2])}, lap_mult={"SC": 1.5, "VSC": 1.3})


def toy_tyre():
    return TyreParams.from_dict(k0=[0, -0.3, -0.6], beta=[0.05, 0.08, 0.12], kappa=[0, 0, 0], gamma=0.06,
                                sigma=0.0, nu=5.0, sd_stint=0.0, rem_mean=9.5)


FP0 = FieldParams(start_offset={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}, grid_gap=0.0)


def test_single_car_matches_single_car_engine():
    r, t = toy_race(), toy_tyre()
    s = Strategy("a", "MEDIUM", [(10, "HARD")])
    single = simulate(r, t, s, n_sims=3, seed=0)["total"]
    multi = simulate_field(r, t, [Car("X", 0.0, 1, s)], FP0, n_sims=3, seed=0)["total"][:, 0]
    assert np.allclose(single, multi)


def test_blocked_when_pass_impossible_and_passes_when_certain():
    r, t = toy_race(), toy_tyre()
    s = Strategy("a", "MEDIUM", [(10, "HARD")])
    cars = [Car("SLOW", 0.5, 1, s), Car("FAST", 0.0, 2, s)]
    fp_block = FieldParams(start_offset={}, grid_gap=0.5, pass_a=-50, pass_b=0.0, d_gap_lt1=0.0, d_gap_1to2=0.0)
    r1 = simulate_field(r, t, cars, fp_block, n_sims=2, seed=0)
    assert (r1["position"][:, 1] == 2).all()                       # 快车被堵在后面
    gap = r1["total"][:, 1] - r1["total"][:, 0]
    assert np.allclose(gap, fp_block.min_gap)
    fp_pass = FieldParams(start_offset={}, grid_gap=0.5, pass_a=50, pass_b=0.0, d_gap_lt1=0.0, d_gap_1to2=0.0)
    r2 = simulate_field(r, t, cars, fp_pass, n_sims=2, seed=0)
    assert (r2["position"][:, 1] == 1).all()                       # 必定超过


def test_sc_compresses_gaps():
    r, t = toy_race(lap1_sc=1.0), toy_tyre()                        # 第 1 圈出 SC，持续 3 圈
    s = Strategy("a", "MEDIUM", [(12, "HARD")])
    cars = [Car(f"C{i}", 0.4 * i, i + 1, s) for i in range(4)]
    fp = FieldParams(start_offset={}, grid_gap=3.0, sc_restart_gap=0.5)
    # 只跑到 SC 结束后立即比较：用 3 圈的比赛
    r3 = RaceConfig(**{**r.__dict__, "total_laps": 3})
    out = simulate_field(r3, t, [Car(c.name, c.pace, c.grid, Strategy("b", "MEDIUM", [(1, "HARD")])) for c in cars], fp, n_sims=1, seed=0)
    T = np.sort(out["total"][0])
    assert np.all(np.diff(T) <= 0.5 + 1e-9)


def test_reproducible_and_ego_self_difference_zero():
    r, t = toy_race(), toy_tyre()
    s = Strategy("a", "MEDIUM", [(10, "HARD")])
    cars = [Car("A", 0.0, 1, s), Car("B", 0.1, 2, s), Car("C", 0.2, 3, s)]
    a = simulate_field(r, t, cars, FieldParams(), n_sims=50, seed=3)["total"]
    b = simulate_field(r, t, cars, FieldParams(), n_sims=50, seed=3)["total"]
    assert np.array_equal(a, b)
    ev = evaluate_ego(r, t, cars, 1, [s, Strategy("a2", "MEDIUM", [(10, "HARD")])], FieldParams(), n_sims=50, seed=3)
    assert np.allclose(ev.d_pos_vs_best, 0) and np.allclose(ev.d_time_vs_best, 0)


def test_sc_compression_caps_each_adjacent_gap():
    race = toy_race(n=3, lap1_sc=1)
    strategy = Strategy("same", "MEDIUM", [(1, "HARD")])
    cars = [Car(str(grid), 0, grid, strategy) for grid in (1, 2, 10)]
    fp = FieldParams(grid_gap=0.1, sc_restart_gap=0.5)
    times = simulate_field(race, toy_tyre(), cars, fp, n_sims=1)["total"][0]
    np.testing.assert_allclose(np.diff(times), [0.1, 0.5])


def test_pass_probability_handles_extreme_pace_difference():
    race = toy_race(n=3)
    strategy = Strategy("same", "MEDIUM", [(2, "HARD")])
    cars = [Car("front", 0, 1, strategy), Car("back", 500, 2, strategy)]
    with np.errstate(over="raise"):
        positions = simulate_field(race, toy_tyre(), cars, FieldParams(), n_sims=1)["position"]
    assert positions.tolist() == [[1, 2]]


@pytest.mark.parametrize("check", ["benchmark", "top"])
def test_backtest_actual_is_only_a_benchmark_and_top_is_official(monkeypatch, check):
    import pandas as pd
    from pathlib import Path
    from f1a.sim import field, optimize

    root = Path(__file__).resolve().parents[1]
    strategies = pd.DataFrame(dict(year=2024, round=10, driver=list("ABCDEFGHI"),
                                   finish_position=range(1, 10), finisher=True))
    # P1 lacks a usable lap history. P3 must not replace P1 in the official top two.
    actual = Strategy("SMSM", "SOFT", [(10, "MEDIUM"), (25, "SOFT"), (40, "MEDIUM")])
    cars = [Car(name, 0, grid, actual) for grid, name in enumerate("BCDEFGHI", 1)]
    monkeypatch.setattr(pd, "read_parquet", lambda *args: pd.DataFrame(
        {"race_id": ["2024-10"], "compound": ["SOFT"], "lap": [1], "stint_uid": [1]}))
    from f1a.sim import params
    monkeypatch.setattr(params, "estimated_compounds", lambda *args: ["SOFT", "MEDIUM"])
    monkeypatch.setattr(field.RaceConfig, "from_project", lambda *args: toy_race(n=66))
    monkeypatch.setattr(field.TyreParams, "from_posterior", lambda *args, **kwargs: toy_tyre())
    monkeypatch.setattr(field, "cars_from_project", lambda *args: cars)
    monkeypatch.setattr(optimize, "optimize_stops", lambda *args, **kwargs:
                        pd.DataFrame([dict(sequence="S-M", stops=(25,), mean=5000)]))

    def evaluation(race, tyre, cars, ego, options, *args, **kwargs):
        return pd.DataFrame([dict(strategy=s.name, label=s.label(),
                                  exp_pos=1 if s == actual else 2) for s in reversed(options)])

    monkeypatch.setattr(field, "evaluate_ego", evaluation)
    out = field.backtest_field_race(root, "2024-10", 2024, fp=FieldParams(), top=2, strategies=strategies)
    if check == "benchmark":
        assert list(out.model_stops) == [1] * len(out)
        assert out.actual_exp_pos.iloc[0] == 1
    else:
        assert list(out.driver) == ["B"]
