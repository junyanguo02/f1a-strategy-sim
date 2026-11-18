"""T031 独立读回校验。

在项目根目录执行：

    .venv/bin/python analysis/extra_stop/verify_extra_stop.py

检查：
  1. 输入哈希、输出文件、两次运行哈希一致；
  2. 主样本/对照车对数与分组计数；
  3. 停车异常、进站前被堵、分组规则可从原始数据抽对重算；
  4. 指标汇总与报告关键数字；
  5. 报告里关键数字存在。

只读，不写任何结果文件。
"""
from __future__ import annotations

import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROCESSED = ROOT / "data" / "processed"
T028 = ROOT / "analysis" / "teammate_outcomes" / "outputs" / "teammate_pairs.csv"
STINTS = ROOT / "analysis" / "actual_strategies" / "stints.csv"
STRATEGIES = ROOT / "analysis" / "actual_strategies" / "strategies.parquet"
LAPS = PROCESSED / "laps_pit_filled.parquet"
MODEL = PROCESSED / "model_laps.parquet"
RACE_SUMMARY = PROCESSED / "fcy_race_summary.csv"
PIT_GLOB = ROOT / "data" / "raw" / "openf1" / "pit_session_key=*.json"
OUT = HERE / "outputs"
FIG = HERE / "figures"
LOG = HERE / "logs"
FAILS: list[str] = []
PASSES: list[str] = []


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSES.append(name)
        print(f"PASS: {name}" + (f" — {detail}" if detail else ""))
    else:
        FAILS.append(name)
        print(f"FAIL: {name}" + (f" — {detail}" if detail else ""))


def load_cum():
    laps = pd.read_parquet(LAPS)
    cum = {}
    for (y, r, drv), g in laps.groupby(["year", "round", "driver"]):
        g = g.sort_values("lap")
        la = g["lap"].to_numpy(int)
        lt = g["lap_time_s"].to_numpy(float)
        cc = {}
        tot, prev = 0.0, 0
        for i in range(len(g)):
            k = int(la[i])
            if k != prev + 1 or not np.isfinite(lt[i]):
                break
            tot += float(lt[i])
            cc[k] = tot
            prev = k
        cum[(int(y), int(r), str(drv))] = cc
    return cum


def pit_duration_map():
    strat = pd.read_parquet(STRATEGIES)
    sess = strat.drop_duplicates(["year", "round"]).set_index("session_key")[["year", "round"]]
    rows = []
    for f in glob.glob(str(PIT_GLOB)):
        sk = int(Path(f).name.split("=")[1].split(".")[0])
        if sk not in sess.index:
            continue
        y, r = sess.loc[sk]
        for rec in json.load(open(f, encoding="utf-8")):
            rows.append({"year": int(y), "round": int(r),
                         "driver_number": rec.get("driver_number"),
                         "lap": rec.get("lap_number"),
                         "dur": rec.get("lane_duration") or rec.get("pit_duration")})
    df = pd.DataFrame(rows)
    circ = pd.read_csv(RACE_SUMMARY)[["year", "round", "circuit_id"]].drop_duplicates()
    df = df.merge(circ, on=["year", "round"], how="left").dropna(subset=["circuit_id"])
    return df, df.groupby("circuit_id")["dur"].median()


def check_files() -> None:
    required = ["report.md"]
    required += [f"outputs/{n}" for n in [
        "pair_metrics.csv", "metrics_summary.csv", "group_outcomes.csv",
        "flag_prevalence.csv", "input_sha256.csv"]]
    for rel in required:
        p = HERE / rel
        check(f"output exists: {rel}", p.exists() and p.stat().st_size > 0,
              f"{p.stat().st_size} bytes" if p.exists() else "missing")
    figs = sorted(p.name for p in FIG.glob("*.png"))
    check("three figures", len(figs) == 3, ", ".join(figs))
    for name in figs:
        p = FIG / name
        check(f"figure non-empty: {name}", p.stat().st_size > 10_000,
              f"{p.stat().st_size} bytes")
    hashes = pd.read_csv(OUT / "input_sha256.csv")
    for rel, path in [("analysis/teammate_outcomes/outputs/teammate_pairs.csv", T028),
                      ("analysis/stint_heterogeneity/outputs/stint_slopes.csv",
                       ROOT / "analysis" / "stint_heterogeneity" / "outputs" / "stint_slopes.csv"),
                      ("analysis/actual_strategies/stints.csv", STINTS),
                      ("data/processed/laps_pit_filled.parquet", LAPS),
                      ("data/processed/model_laps.parquet", MODEL)]:
        row = hashes[hashes["文件"] == rel]
        check(f"input hash listed: {rel}", len(row) == 1)
        if len(row) == 1:
            check(f"input hash matches: {rel}", row.iloc[0]["sha256"] == sha256(path),
                  row.iloc[0]["sha256"][:16])


def check_two_runs() -> None:
    r1 = LOG / "run1.sha256"
    r2 = LOG / "run2.sha256"
    check("run1 hash log exists", r1.exists() and r1.stat().st_size > 0)
    check("run2 hash log exists", r2.exists() and r2.stat().st_size > 0)
    if not (r1.exists() and r2.exists()):
        return
    t1 = r1.read_text(encoding="utf-8").strip()
    t2 = r2.read_text(encoding="utf-8").strip()
    check("two runs produce identical file hashes", t1 == t2)
    files = sorted(list(OUT.glob("*.csv")) + list(FIG.glob("*.png")) + [HERE / "report.md"])
    current = [f"{sha256(p)}  {p.relative_to(HERE).as_posix()}" for p in files]
    check("current outputs match the saved run hashes", "\n".join(current) == t1,
          f"{len(files)} files compared")
    marker = LOG / "two_runs.txt"
    check("two_runs.txt records identical", marker.exists()
          and "TWO RUNS IDENTICAL" in marker.read_text(encoding="utf-8"))


def check_sample() -> None:
    df = pd.read_csv(OUT / "pair_metrics.csv")
    main = df[df["extra_stop_sc"] == "绿旗下"]
    ctrl = df[df["extra_stop_sc"] == "SC/VSC 下"]
    check("main = 82", len(main) == 82, str(len(main)))
    check("control = 26", len(ctrl) == 26, str(len(ctrl)))
    summary = pd.read_csv(OUT / "group_outcomes.csv")
    check("group counts sum = 82", int(summary["车对数"].sum()) == 82,
          str(summary["车对数"].to_dict()))

    def grp(row):
        if pd.notna(row["slope_diff"]) and row["slope_diff"] >= 0.03:
            return "轮胎衰减更快（被迫）"
        if pd.notna(row["rise_diff"]) and row["rise_diff"] >= 0.60:
            return "轮胎衰减更快（被迫）"
        if bool(row["stop_abnormal"]):
            return "停车异常（被迫）"
        if bool(row["blocked_before"]) or bool(row["tactical"]):
            return "被堵 / 战术"
        return "无明显原因（主动）"

    re_groups = main.apply(grp, axis=1)
    check("group rule recompute matches",
          bool((re_groups.values == main["group"].values).all()))


def check_metrics() -> None:
    df = pd.read_csv(OUT / "pair_metrics.csv")
    main = df[df["extra_stop_sc"] == "绿旗下"]
    ctrl = df[df["extra_stop_sc"] == "SC/VSC 下"]
    ms = pd.read_csv(OUT / "metrics_summary.csv")
    row = ms[ms["样本"].str.startswith("主样本")].iloc[0]
    check("slope diff available = 44", int(main["slope_diff"].notna().sum()) == 44,
          str(int(main["slope_diff"].notna().sum())))
    check("slope diff mean ≈ +0.00945",
          abs(float(row["斜率差均值"]) - 0.00945) < 1e-4,
          f"{float(row['斜率差均值']):.5f}")
    check("rise diff available = 73", int(main["rise_diff"].notna().sum()) == 73,
          str(int(main["rise_diff"].notna().sum())))
    check("blocked pairs = 28", int(main["blocked_before"].sum()) == 28)
    check("tactical pairs = 26", int(main["tactical"].sum()) == 26)
    check("stop abnormal available = 73", int(main["stop_duration"].notna().sum()) == 73)
    check("stop abnormal = 6", int(main["stop_abnormal"].fillna(False).sum()) == 6)
    check("control mean time diff ≈ -11.61",
          abs(float(ctrl["time_diff"].mean()) - (-11.60596)) < 1e-3,
          f"{float(ctrl['time_diff'].mean()):.2f}")


def check_one_pair() -> None:
    df = pd.read_csv(OUT / "pair_metrics.csv")
    main = df[df["extra_stop_sc"] == "绿旗下"].reset_index(drop=True)
    # 选有停车时长的一对
    r = main[main["stop_duration"].notna()].iloc[0]
    pits, med = pit_duration_map()
    strat = pd.read_parquet(STRATEGIES)
    num = strat.drop_duplicates(["year", "round", "driver"])[
        ["year", "round", "driver", "driver_number"]]
    dnum = int(num[(num["year"] == r["year"]) & (num["round"] == r["round"])
                   & (num["driver"] == r["driver_more"])]["driver_number"].iloc[0])
    p = pits[(pits["year"] == r["year"]) & (pits["round"] == r["round"])
             & (pits["driver_number"] == dnum) & (pits["lap"] == r["extra_stop_lap"])]
    dur = float(p["dur"].mean())
    ab = dur > (float(med[r["circuit_id"]]) + 3.0)
    check("one pair stop duration & abnormal recompute",
          abs(dur - float(r["stop_duration"])) < 1e-6
          and bool(ab) == bool(r["stop_abnormal"]),
          f"dur={dur:.1f}, abnormal={ab}")
    # 被堵：用累计时间重算 L-3..L-1 的最小前车间距
    cum = load_cum()
    y, rd = int(r["year"]), int(r["round"])
    cm = cum[(y, rd, r["driver_more"])]
    L = int(r["extra_stop_lap"])
    gaps = []
    for lap in range(max(1, L - 3), L):
        if lap not in cm:
            continue
        drivers_lap = [d for (yy, rr, d), cc in cum.items()
                       if yy == y and rr == rd and lap in cc]
        rank = sorted(drivers_lap, key=lambda d: cum[(y, rd, d)][lap])
        i = rank.index(r["driver_more"])
        if i > 0:
            gaps.append(cm[lap] - cum[(y, rd, rank[i - 1])][lap])
    min_gap = min(gaps) if gaps else np.nan
    check("one pair blocked recompute",
          (pd.isna(min_gap) and pd.isna(r["min_gap_before"]))
          or abs(min_gap - float(r["min_gap_before"])) < 1e-6,
          f"{min_gap:.3f}" if pd.notna(min_gap) else "—")


def check_report() -> None:
    text = (HERE / "report.md").read_text(encoding="utf-8")
    for needle in ["82", "26", "34", "25", "17", "6", "23", "16.4", "-11.6",
                   "0.03", "0.60", "3.0"]:
        check(f"report contains {needle}", needle in text)


def main() -> None:
    check_files()
    check_two_runs()
    check_sample()
    check_metrics()
    check_one_pair()
    check_report()
    print()
    print(f"TOTAL: {len(PASSES)} passed, {len(FAILS)} failed")
    if FAILS:
        print("FAILED CHECKS:")
        for f in FAILS:
            print(" -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
