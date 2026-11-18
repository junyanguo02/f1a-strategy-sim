"""事前回测：对一个赛季的干地、无红旗分站，比较模拟器推荐与前十名实际策略。

用法（在 projects/f1a-strategy-sim/ 下）：
    .venv/bin/python scripts/backtest.py --season 2024 --out analysis/backtest/backtest_2024.csv
"""
import argparse, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1a.sim.backtest import backtest_race  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sims", type=int, default=1000); ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--step", type=int, default=2); ap.add_argument("--fcy-by-circuit", action="store_true"); ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--prerace", action="store_true", help="使用严格日期截止的赛前输入")
    a = ap.parse_args()
    if a.prerace:
        from f1a.sim.prerace import run_season
        return run_season(ROOT, a, field=False)
    A = ROOT / "analysis" / "actual_strategies"
    strat = pd.read_parquet(A / "strategies.parquet")
    rs = pd.read_csv(A / "outputs" / "race_summary.csv")
    rs = rs[(rs.year == a.season) & rs.dry_for_analysis.astype(bool) & ~rs.has_red_flag.astype(bool) & rs.usable.astype(bool)]
    ml = pd.read_parquet(ROOT / "data" / "processed" / "model_laps.parquet")
    have = set(ml[ml.year == a.season].race_id)
    rows, t0 = [], time.time()
    for _, r in rs.iterrows():
        rid = f"{r.year}-{r['round']}"
        if rid not in have:
            rows.append(dict(race_id=rid, skipped="不在模型数据里")); continue
        out = backtest_race(ROOT, rid, a.season, strat, n_sims=a.sims, n_draws=a.draws, step=a.step, seed=a.seed,
                           fcy_by_circuit=a.fcy_by_circuit)
        out.update(event=r.event, n_sc=int(r.n_sc_events), n_vsc=int(r.n_vsc_events), sc_vsc_pit_share=float(r.sc_vsc_pit_share))
        rows.append(out)
        print(f"{rid} {r.event}: 模型 {out.get('model_best')} {out.get('model_best_stops')} | 前十 {out.get('top10_dominant')} "
              f"({out.get('top10_dominant_share', 0):.0%}) | SC {int(r.n_sc_events)} VSC {int(r.n_vsc_events)}", flush=True)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    ok = df[df.get("skipped").isna()] if "skipped" in df else df
    noSC = ok[ok.n_sc == 0]
    f = lambda d: (f"进站次数一致 {d.stops_match.mean():.0%}，配方组合一致 {d.combo_match.mean():.0%}，"
                   f"前十主流组合在模型里排名中位数 {d.dominant_combo_rank.median():.0f}，与模型最优差中位数 {d.dominant_combo_gap_s.median():.1f} 秒")
    print(f"\n全部 {len(ok)} 场：" + f(ok)); print(f"无 SC 的 {len(noSC)} 场：" + f(noSC))
    print(f"用时 {time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing local input: {exc.filename}. See scripts/download_data.md.") from None
