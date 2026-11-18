"""Run from the project root after each filter's seven season files are available."""
from pathlib import Path
import sys
import hashlib

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from f1a.sim.prerace import calendar
from f1a.sim.race import pit_tables, FCY_TYPES, PHASE_GROUPS

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
BASE = ROOT / "analysis/backtest"
KEY = ["stage", "race_id", "driver"]


def collect(variant, schedule):
    table, _ = pit_tables(ROOT, None if variant == "original" else variant)
    usable = table.loc[table.relative_usable & table.condition.eq("正常")]
    usable = usable.merge(schedule[["year", "round", "date"]], on=["year", "round"], how="left", validate="many_to_one")
    if usable.date.isna().any():
        raise ValueError("Normal pit sample outside race calendar")
    rows = []
    support = []
    for stage, years in (("post", range(2023, 2027)), ("pre", range(2024, 2027))):
        for year in years:
            folder = "prerace" if stage == "pre" else "multicar"
            path = BASE / folder / f"field_{year}.csv" if variant == "original" else OUT / f"{variant}-{stage}-{year}.csv"
            frame = pd.read_csv(path)
            for rid, cars in frame.groupby("race_id", sort=False):
                rnd = int(rid.split("-")[1])
                race = schedule.loc[schedule.year.eq(year) & schedule["round"].eq(rnd)].iloc[0]
                history = usable.loc[usable.date.lt(race.date)] if stage == "pre" else usable
                n = int(history.circuit_id.eq(race.circuit_id).sum())
                valid = cars.dropna(subset=["actual_stops", "model_stops"]).copy()
                pool = "circuit" if n >= 10 else "global"
                support.append(dict(variant=variant, stage=stage, year=year, race_id=rid, own_n=n,
                                    all_n=len(history), pool=pool, valid=len(valid), skipped=len(cars)-len(valid)))
                rows.append(valid.assign(variant=variant, stage=stage, year=year, pool=pool)[
                    KEY + ["variant", "year", "pool", "actual_stops", "model_stops"]])
    scored = pd.concat(rows, ignore_index=True)
    assert not scored.duplicated(KEY).any()
    return scored, pd.DataFrame(support)


def summarize(rows):
    delta = rows.model_stops - rows.actual_stops
    miss = int(delta.ne(0).sum())
    return dict(n=len(rows), agree=len(rows)-miss, mismatch=miss, fewer=int(delta.lt(0).sum()),
                one_fewer=int(delta.eq(-1).sum()))


def cell(row):
    def rate(a,b):
        return f"{a}/{b} ({100*a/b:.1f}%)" if b else "0/0 (—)"
    return " · ".join([rate(row.agree,row.n),rate(row.fewer,row.mismatch),rate(row.one_fewer,row.mismatch)])


def main():
    variants = ["original", "range"] + (["clean"] if "--clean" in sys.argv else [])
    inputs=[ROOT/"analysis/fcy_inputs/pit_loss_relative.parquet",
            ROOT/"analysis/sc_pit_timing/outputs/pit_match_detail.csv",
            ROOT/"analysis/actual_strategies/outputs/race_summary.csv"]
    inputs.extend(sorted((ROOT/"models/tyre_v3").glob("outputs31_*/posterior_core.npz")))
    if "clean" in variants:
        inputs.append(ROOT/"analysis/pitloss_audit/outputs/pit_loss_relative_clean.parquet")
    with (OUT/"input_hashes.sha256").open("w") as f:
        for path in inputs:
            raw=path.read_bytes()
            if not raw or len(raw)!=path.stat().st_size:
                raise OSError(f"Incomplete local input: {path}")
            f.write(hashlib.sha256(raw).hexdigest()+"  "+str(path.relative_to(ROOT))+"\n")
    schedule = calendar(ROOT)
    scored, pools = zip(*(collect(v, schedule) for v in variants))
    cars = pd.concat(scored, ignore_index=True)
    baseline = scored[0][KEY+["pool","actual_stops","model_stops"]].rename(columns={
        "pool":"original_pool","model_stops":"original_stops","actual_stops":"original_actual"})
    cars = cars.merge(baseline,on=KEY,how="left",validate="many_to_one")
    both = cars.original_stops.notna()
    assert cars.loc[both,"actual_stops"].eq(cars.loc[both,"original_actual"]).all()
    cars.to_csv(OUT/"paired.csv",index=False)
    pd.concat(pools).to_csv(OUT/"pool_sources.csv",index=False)
    rows=[]
    for (v, stage), group in cars.groupby(["variant","stage"],sort=False):
        for year, sample in [("all",group),*group.groupby("year")]:
            for pool in ("all","circuit","global"):
                subset=sample if pool=="all" else sample[sample.pool==pool]
                rows.append(dict(variant=v,stage=stage,year=year,pool=pool,**summarize(subset)))
    counts=pd.DataFrame(rows)
    counts.to_csv(OUT/"counts.csv",index=False)
    report=["# 进站损失敏感性回测", "", "每格依次为停数一致 / 全部有效车次、少停 / 不一致车次、恰少一停 / 不一致车次。按车次汇总；赛道样本达到 10 条时使用本赛道正常池，否则使用全体正常池。赛前只计比赛日前的记录。每列按该版本实际使用的池分组。", "",
            "| 模式 | 正常损失池 | 原始 | 10–60 秒 | T042 clean |","|---|---|---|---|---|"]
    stages={"post":"事后 2023–2026","pre":"赛前 2024–2026"}
    labels={"all":"全体","circuit":"本赛道","global":"回退全体"}
    for stage in stages:
        for pool in labels:
            result=counts[(counts.stage==stage)&(counts.year=="all")&(counts.pool==pool)]
            columns=[cell(r) for r in result.itertuples()]
            if "clean" not in variants:columns.append("待 T042")
            report.append("| "+" | ".join([stages[stage],labels[pool]]+columns)+" |")
    pit_stats=[]
    for variant in variants:
        normal, detail=pit_tables(ROOT, None if variant=="original" else variant)
        samples={"normal":normal.loc[normal.relative_usable & normal.condition.eq("正常"),"relative_loss_s"]}
        for flag in FCY_TYPES:
            for phase, groups in PHASE_GROUPS.items():
                samples[f"{flag}-{phase}"]=detail.loc[detail.condition.eq(flag) & detail.dt_group.isin(groups),"relative_loss_s"].dropna()
        for pool,values in samples.items():
            pit_stats.append(dict(variant=variant,pool=pool,n=len(values),mean_s=values.mean()))
    pd.DataFrame(pit_stats).to_csv(OUT/"pool_means.csv",index=False)
    report.extend(["","## 全体正常池",""])
    for row in pit_stats:
        if row["pool"]=="normal":
            report.append(f"- {row['variant']}：{row['n']} 条，均值 {row['mean_s']:.3f} 秒。")
    normal_means={row["variant"]:row["mean_s"] for row in pit_stats if row["pool"]=="normal"}
    change=normal_means["original"]-normal_means["range"]
    report.append(f"四季原始表按 relative_usable 为真且 condition 为“正常”取样，10–60 秒使全体池均值下降 {change:.3f} 秒。任务背景所列 2.54 秒未从这个四季分母复现。赛前各场只使用当时已完赛的历史，赛道池的数量和均值也随分站变化。")
    report.extend(["","## 同车次核对",""])
    for (v,stage),group in cars[cars.variant!="original"].groupby(["variant","stage"]):
        orig=baseline[baseline.stage==stage]
        paired=group.dropna(subset=["original_stops"])
        old_fewer=paired.original_stops.lt(paired.actual_stops)
        new_fewer=paired.model_stops.lt(paired.actual_stops)
        resolved=int((old_fewer & ~new_fewer).sum())
        introduced=int((~old_fewer & new_fewer).sum())
        report.append(f"- {stages[stage]}，{v}。配对 {len(paired)}，原始独有 {len(orig)-len(paired)}，过滤版独有 {len(group)-len(paired)}。推荐停数改变 {int(paired.model_stops.ne(paired.original_stops).sum())}，正常池来源切换 {int(paired.pool.ne(paired.original_pool).sum())}。原先少停而本版不再少停 {resolved}，反向变化 {introduced}。")
    report.extend(["","事后、赛前原始值分别来自 T025 和 T030 各季 field CSV。固定种子 20261118、200 组后验、每候选 300 次、多车前十。事后使用原 T025 入口；赛前使用 T030 的日期截止与对手规则。两类进站样本都按所选口径过滤，再进行赛道池回退和 SC/VSC 分阶段。各季分子、分母见 outputs/counts.csv，逐站池大小与跳过数见 outputs/pool_sources.csv，逐车推荐见 outputs/paired.csv。"])
    report.extend(["","## 复现","",
                   "在项目根目录执行。默认省略 --pit-loss-filter 时沿用原始池，可在 Python 构造器中通过 pit_loss_bounds=(lo, hi) 修改 range 的闭区间。","",
                   "```sh",'# 完成 scripts/download_data.md 的数据和后验准备后，重建原始池基线',
                   'for y in 2023 2024 2025 2026; do',
                   '  .venv/bin/python scripts/backtest_field.py --season "$y" --out "analysis/backtest/multicar/field_$y.csv"',
                   '  if [ "$y" != 2023 ]; then',
                   '    .venv/bin/python scripts/backtest_field.py --season "$y" --prerace --out "analysis/backtest/prerace/field_$y.csv"',
                   '  fi',
                   'done',"bash analysis/pitloss_sensitivity/run.sh range",
                   ".venv/bin/python analysis/pitloss_sensitivity/compare.py",
                   ".venv/bin/python analysis/pitloss_audit/run_pitloss_audit.py","bash analysis/pitloss_sensitivity/run.sh clean",
                   ".venv/bin/python analysis/pitloss_sensitivity/compare.py --clean","```"])
    (HERE/"report.md").write_text("\n".join(report)+"\n")


if __name__=="__main__":
    main()
