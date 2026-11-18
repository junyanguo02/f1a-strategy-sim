"""单车模拟器演示：一场真实分站的策略比较与进站圈优化。

用法（在 projects/f1a-strategy-sim/ 下）：
    .venv/bin/python scripts/sim_demo.py --race 2024-10 --season 2024 --sims 5000
"""
import argparse, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1a.sim import RaceConfig, TyreParams, Strategy, evaluate, optimize_stops  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race", default="2024-10"); ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--sims", type=int, default=5000); ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261118)
    a = ap.parse_args()
    npz = ROOT / "models" / "tyre_v3" / f"outputs31_{a.season}" / "posterior_core.npz"
    race = RaceConfig.from_project(ROOT, a.race)
    tyre = TyreParams.from_posterior(npz, a.race, [a.season], ROOT / "data" / "processed" / "model_laps.parquet", n_draws=a.draws, seed=a.seed)
    print(f"分站 {a.race}：{race.total_laps} 圈，基础圈时 {race.base_lap_s:.3f} 秒；后验抽样 {tyre.n_draws} 个")
    for n in race.notes:
        print("  -", n)
    N = race.total_laps
    strategies = [
        Strategy("一停 中→硬", "MEDIUM", [(round(N * 0.40), "HARD")]),
        Strategy("一停 软→硬", "SOFT", [(round(N * 0.30), "HARD")]),
        Strategy("两停 软→中→软", "SOFT", [(round(N * 0.25), "MEDIUM"), (round(N * 0.68), "SOFT")]),
        Strategy("两停 软→中→中", "SOFT", [(round(N * 0.27), "MEDIUM"), (round(N * 0.64), "MEDIUM")]),
    ]
    t0 = time.time()
    df = evaluate(race, tyre, strategies, n_sims=a.sims, seed=a.seed)
    pd.set_option("display.width", 200)
    print("\n固定进站圈的策略比较（秒；delta 为相对平均最快策略的配对差及 95% 区间）：")
    print(df.round(2).to_string(index=False))
    print(f"模拟中出现 SC 的比例 {df.attrs['share_sims_with_sc']:.2f}，出现 VSC 的比例 {df.attrs['share_sims_with_vsc']:.2f}")
    opt = optimize_stops(race, tyre, [("MEDIUM", "HARD"), ("SOFT", "HARD"), ("SOFT", "MEDIUM"),
                                      ("SOFT", "MEDIUM", "SOFT"), ("SOFT", "MEDIUM", "MEDIUM"), ("MEDIUM", "HARD", "SOFT")],
                         n_sims=min(a.sims, 2000), seed=a.seed, step=1)
    print("\n各配方顺序的最优进站圈（按平均总用时）：")
    print(opt.round(2).to_string(index=False))
    print(f"\n用时 {time.time() - t0:.1f} 秒；种子 {a.seed}")


if __name__ == "__main__":
    main()
