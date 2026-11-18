"""多车回测：每场前 N 名完赛车逐一做“本车”，其他车固定实际策略，比较模拟器推荐的进站次数与该车实际。

用法（在 projects/f1a-strategy-sim/ 下）：
    .venv/bin/python scripts/backtest_field.py --season 2024 --out analysis/backtest/multicar/field_2024.csv
"""
import argparse, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1a.sim.field import backtest_field_race  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sims", type=int, default=300); ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--top", type=int, default=10); ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--races", nargs="*", help="只跑指定 race_id（调试用）")
    ap.add_argument("--prerace", action="store_true", help="使用严格日期截止的赛前输入")
    ap.add_argument("--sensitivity", action="store_true", help="赛前多车四组单因素敏感性")
    ap.add_argument("--pit-loss-filter", choices=["range", "clean"])
    a = ap.parse_args()
    if a.sensitivity and not a.prerace:
        ap.error("--sensitivity requires --prerace")
    from f1a.sim.race import pit_loss_batch
    with pit_loss_batch(a.pit_loss_filter):
        return run(a)


def run(a):
    if a.prerace:
        from f1a.sim.prerace import run_season
        return run_season(ROOT, a, field=True)
    A = ROOT / "analysis" / "actual_strategies"
    strat = pd.read_parquet(A / "strategies.parquet")
    rs = pd.read_csv(A / "outputs" / "race_summary.csv")
    rs = rs[(rs.year == a.season) & rs.dry_for_analysis.astype(bool) & ~rs.has_red_flag.astype(bool) & rs.usable.astype(bool)]
    ml = pd.read_parquet(ROOT / "data" / "processed" / "model_laps.parquet")
    have = set(ml[ml.year == a.season].race_id)
    out, t0 = [], time.time()
    for _, r in rs.iterrows():
        rid = f"{r.year}-{r['round']}"
        if rid not in have or (a.races and rid not in a.races):
            continue
        d = backtest_field_race(ROOT, rid, a.season, n_sims=a.sims, n_draws=a.draws, seed=a.seed, top=a.top, strategies=strat)
        d["event"] = r.event; d["n_sc"] = int(r.n_sc_events)
        out.append(d)
        if "stops_match" in d:
            print(f"{rid} {r.event}: 多车一致 {d.stops_match.mean():.0%}，单车一致 {d.single_match.mean():.0%}（{len(d)} 辆）", flush=True)
        else:
            print(f"{rid} {r.event}: 跳过 {d.get('skipped', pd.Series(['?'])).iloc[0]}", flush=True)
    df = pd.concat(out, ignore_index=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    ok = df[df.get("stops_match").notna()] if "stops_match" in df else df
    less = (ok.model_stops < ok.actual_stops).mean(); more = (ok.model_stops > ok.actual_stops).mean()
    print(f"\n{a.season}：{len(ok)} 车次；进站次数一致 多车 {ok.stops_match.mean():.0%} / 单车 {ok.single_match.mean():.0%}；"
          f"多车少停 {less:.0%}、多停 {more:.0%}；用时 {time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing local input: {exc.filename or exc}. See scripts/download_data.md.") from None
