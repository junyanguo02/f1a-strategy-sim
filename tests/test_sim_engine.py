"""模拟器单元测试（不依赖项目数据，全部用手写参数）。"""
import numpy as np
import pytest
from f1a.sim import RaceConfig, TyreParams, Strategy, simulate, evaluate, common_draws, optimize_stops


def toy_race(n=20, hazard=0.0, lap1=0.0):
    return RaceConfig(race_id="toy", total_laps=n, base_lap_s=90.0, pit_normal=np.array([20.0]),
                      pit_fcy={(t, p): np.array([10.0]) for t in ("SC", "VSC") for p in (1, 2, 3)},
                      lap1_prob={"SC": lap1, "VSC": 0.0}, hazard={"SC": np.full(5, hazard), "VSC": np.zeros(5)},
                      durations={"SC": np.array([3]), "VSC": np.array([2])}, lap_mult={"SC": 1.5, "VSC": 1.3})


def toy_tyre(sigma=0.0, sd_stint=0.0):
    return TyreParams.from_dict(k0=[0, -0.3, -0.6], beta=[0.05, 0.08, 0.12], kappa=[0, 0, 0], gamma=0.06,
                                sigma=sigma, nu=5.0, sd_stint=sd_stint, rem_mean=9.5)


def test_reproducible_same_seed():
    r, t = toy_race(hazard=0.05), toy_tyre(sigma=0.5, sd_stint=0.2)
    s = Strategy("a", "MEDIUM", [(10, "HARD")])
    a = simulate(r, t, s, n_sims=200, seed=7)["total"]; b = simulate(r, t, s, n_sims=200, seed=7)["total"]
    assert np.array_equal(a, b)


def test_hand_computed_total_no_noise_no_fcy():
    n = 20; r, t = toy_race(n), toy_tyre()
    s = Strategy("a", "MEDIUM", [(10, "HARD")])
    expect = 0.0
    for l in range(1, n + 1):
        c, age = (1, l) if l <= 10 else (0, l - 10)
        k0 = [0, -0.3, -0.6][c]; beta = [0.05, 0.08, 0.12][c]
        expect += 90.0 + k0 + 0.06 * ((n - l) - 9.5) + beta * (age - 10)
    expect += 20.0
    got = simulate(r, t, s, n_sims=5, seed=0)["total"]
    assert np.allclose(got, expect)


def test_compliance_checks():
    with pytest.raises(ValueError):
        Strategy("x", "SOFT", [(10, "SOFT")]).validate(20)
    with pytest.raises(ValueError):
        Strategy("x", "SOFT", []).validate(20)
    with pytest.raises(ValueError):
        Strategy("x", "SOFT", [(12, "HARD"), (8, "MEDIUM")]).validate(20)
    with pytest.raises(ValueError):
        Strategy("x", "SOFT", [(20, "HARD")]).validate(20)
    Strategy("ok", "SOFT", [(8, "HARD")]).validate(20)


def test_zero_hazard_means_no_fcy():
    D = common_draws(toy_race(hazard=0.0), toy_tyre(), 500, 1)
    assert (D["status"] == 0).all()


def test_fcy_lowers_pit_cost_and_slows_laps():
    r = toy_race(hazard=0.0, lap1=1.0)  # 第 1 圈必出 SC，持续 3 圈
    s_fcy = Strategy("pit2", "MEDIUM", [(2, "HARD")]); s_green = Strategy("pit10", "MEDIUM", [(10, "HARD")])
    a = simulate(r, toy_tyre(), s_fcy, n_sims=3, seed=0)
    assert (a["n_sc"] == 1).all() and np.allclose(a["pit_loss"], 10.0)
    b = simulate(r, toy_tyre(), s_green, n_sims=3, seed=0)
    assert np.allclose(b["pit_loss"], 20.0)


def test_common_random_numbers_self_difference_zero():
    r, t = toy_race(hazard=0.05), toy_tyre(sigma=0.5, sd_stint=0.2)
    s = Strategy("a", "MEDIUM", [(10, "HARD")])
    df = evaluate(r, t, [s, Strategy("a2", "MEDIUM", [(10, "HARD")])], n_sims=300, seed=3)
    assert np.allclose(df.delta_mean, 0) and np.allclose(df.delta_lo, 0) and np.allclose(df.delta_hi, 0)


def test_optimizer_finds_known_optimum_without_noise():
    # 无噪声、无 FCY：中→硬 一停，最优进站圈可由穷举手算验证
    r, t = toy_race(20), toy_tyre()
    out = optimize_stops(r, t, [("MEDIUM", "HARD")], n_sims=2, seed=0, min_stint=3)
    best_lap, best_val = None, None
    for L in range(3, 18):
        v = simulate(r, t, Strategy("m", "MEDIUM", [(L, "HARD")]), n_sims=1, seed=0)["total"][0]
        if best_val is None or v < best_val:
            best_lap, best_val = L, v
    assert out.stops[0] == (best_lap,)


def test_optimizer_respects_max_stint():
    r, t = toy_race(30), toy_tyre()
    out = optimize_stops(r, t, [("MEDIUM", "HARD"), ("MEDIUM", "HARD", "MEDIUM")], n_sims=2, seed=0, min_stint=3, max_stint=12)
    row1 = out[out.sequence == "M-H"].iloc[0]
    assert row1.stops is None          # 一停时总有一段超过 12 圈
    laps = out[out.sequence == "M-H-M"].iloc[0].stops
    b = (0,) + tuple(laps) + (30,)
    assert all(y - x <= 12 for x, y in zip(b, b[1:]))


def test_fractional_stop_lap_rejected():
    with pytest.raises(ValueError):
        Strategy("fraction", "SOFT", [(8.5, "HARD")]).validate(20)


def test_adjacent_sc_episodes_counted_separately():
    race = toy_race(n=7, hazard=1, lap1=1)
    out = simulate(race, toy_tyre(), Strategy("a", "MEDIUM", [(3, "HARD")]), n_sims=2)
    assert np.all(out["n_sc"] == 3)


def test_posterior_remainder_can_use_later_draws():
    t = toy_tyre()
    t.gamma = np.zeros(20)
    t.nu = np.full(20, 5.0)
    indices = common_draws(toy_race(), t, 3, 11)["d"]
    assert len(set(indices)) == 3 and indices.max() >= 3


def test_simulate_supports_seven_stints():
    s = Strategy("seven", "MEDIUM", [(i, "HARD") for i in range(1, 7)])
    assert np.isfinite(simulate(toy_race(), toy_tyre(), s, n_sims=2)["total"]).all()


def test_optimizer_rejects_unknown_objective():
    with pytest.raises(ValueError, match="objective"):
        optimize_stops(toy_race(), toy_tyre(), [("MEDIUM", "HARD")], objective="median")
