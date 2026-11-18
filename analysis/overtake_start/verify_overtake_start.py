"""T023 独立读回校验。

在项目根目录执行：

    .venv/bin/python analysis/overtake_start/verify_overtake_start.py

检查：
  1. 输入哈希、输出文件、两次运行哈希一致；
  2. 场次筛选、逐圈名次与赛道超车总数能从原始数据独立重算；
  3. 超车概率、被堵代价、起步、SC 重启间距的关键数字可重算；
  4. 报告里关键数字存在。

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
LAPS = PROCESSED / "laps_pit_filled.parquet"
EPISODES = PROCESSED / "fcy_episodes.parquet"
RACE_SUMMARY = PROCESSED / "fcy_race_summary.csv"
T020_RACE = ROOT / "analysis" / "actual_strategies" / "outputs" / "race_summary.csv"
T020_STRAT = ROOT / "analysis" / "actual_strategies" / "strategies.csv"
JOLPICA = ROOT / "data" / "raw" / "jolpica"
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


def build_positions(g: pd.DataFrame):
    drivers = {}
    for drv, d in g.groupby("driver"):
        d = d.sort_values("lap")
        laps = d["lap"].to_numpy(int)
        times = d["lap_time_s"].to_numpy(float)
        cum = {}
        s = 0.0
        prev = 0
        for i in range(len(d)):
            k = int(laps[i])
            if k != prev + 1 or not np.isfinite(times[i]):
                break
            s += float(times[i])
            cum[k] = s
            prev = k
        drivers[str(drv)] = cum
    states = {}
    all_laps = set()
    for cum in drivers.values():
        all_laps |= set(cum)
    for k in sorted(all_laps):
        pairs = sorted((drivers[d][k], d) for d in drivers if k in drivers[d])
        states[k] = {d: i + 1 for i, (_, d) in enumerate(pairs)}
    return drivers, states


def flags_map(g: pd.DataFrame) -> dict:
    out = {}
    for r in g.itertuples(index=False):
        out[(str(r.driver), int(r.lap))] = (
            bool(r.pit_in_filled), bool(r.pit_out), bool(r.safety_car),
            bool(r.virtual_safety_car), bool(r.is_clean),
            float(r.lap_time_s) if pd.notna(r.lap_time_s) else np.nan)
    return out


def blocked(flags, a, b, L):
    for drv in (a, b):
        for lap in (L, L + 1):
            f = flags.get((drv, lap))
            if f is None:
                return True
            if f[0] or f[1] or f[2] or f[3]:
                return True
    return False


# ---------------------------------------------------------------- 文件与哈希
def check_files() -> None:
    required = ["report.md"]
    required += [f"outputs/{n}" for n in [
        "race_overtakes.csv", "circuit_overtakes.csv", "overtake_probability.csv",
        "overtake_probability_prev_lap.csv", "blocked_cost.csv",
        "start_by_compound.csv", "start_grid_compound.csv",
        "start_lap1_by_grid.csv", "sc_restart_spacing.csv",
        "sc_restart_summary.csv", "input_sha256.csv"]]
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
    for rel, path in [("data/processed/laps_pit_filled.parquet", LAPS),
                      ("data/processed/fcy_episodes.parquet", EPISODES),
                      ("data/processed/fcy_race_summary.csv", RACE_SUMMARY),
                      ("analysis/actual_strategies/outputs/race_summary.csv", T020_RACE),
                      ("analysis/actual_strategies/strategies.csv", T020_STRAT)]:
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


# ---------------------------------------------------------------- 超车重算
def check_overtakes() -> None:
    laps = pd.read_parquet(LAPS)
    for c in ["pit_in_filled", "pit_out", "safety_car", "virtual_safety_car", "is_clean"]:
        laps[c] = laps[c].fillna(False).astype(bool)
    t020 = pd.read_csv(T020_RACE)
    keep = set(map(tuple, t020[t020["usable"] & t020["dry_for_analysis"]
                                & (~t020["has_red_flag"])][["year", "round"]].values))
    check("filtered races = 74", len(keep) == 74, str(len(keep)))
    race_tab = pd.read_csv(OUT / "race_overtakes.csv")
    check("races with positions = 73", len(race_tab) == 73, str(len(race_tab)))
    check("total overtakes = 1631", int(race_tab["n_overtakes"].sum()) == 1631,
          str(int(race_tab["n_overtakes"].sum())))
    check("mean overtakes per race = 22.34",
          abs(float(race_tab["n_overtakes"].mean()) - 1631 / 73) < 1e-9,
          f"{race_tab['n_overtakes'].mean():.2f}")
    # 独立重算两场
    for year, rnd, expect in [(2023, 1, 14), (2026, 13, 54)]:
        g = laps[(laps["year"] == year) & (laps["round"] == rnd)]
        drivers, states = build_positions(g)
        flags = flags_map(g)
        n = 0
        maxlap = max(states)
        for L in range(1, maxlap):
            r0, r1 = states.get(L), states.get(L + 1)
            if not r0 or not r1:
                continue
            inv = {pos: drv for drv, pos in r0.items()}
            for p in range(1, len(r0)):
                a, b = inv.get(p), inv.get(p + 1)
                if a is None or b is None:
                    continue
                if r1.get(a) != p + 1 or r1.get(b) != p:
                    continue
                if blocked(flags, a, b, L):
                    continue
                n += 1
        row = race_tab[(race_tab["year"] == year) & (race_tab["round"] == rnd)]
        check(f"overtake recount {year} R{rnd}", len(row) == 1 and int(row["n_overtakes"].iloc[0]) == n
              and n == expect, f"recount={n}")
    # 按赛道均值可重算
    circ = pd.read_csv(OUT / "circuit_overtakes.csv")
    rec = race_tab.groupby("circuit_id")["n_overtakes"].agg(["size", "mean"]).reset_index()
    merged = circ.merge(rec, on="circuit_id", how="inner")
    check("circuit means recompute",
          bool(np.allclose(merged["每场平均超车"], merged["mean"]))
          and bool((merged["场数"] == merged["size"]).all()))


# ---------------------------------------------------------------- 概率 / 被堵
def check_probability() -> None:
    laps = pd.read_parquet(LAPS)
    for c in ["pit_in_filled", "pit_out", "safety_car", "virtual_safety_car", "is_clean"]:
        laps[c] = laps[c].fillna(False).astype(bool)
    t020 = pd.read_csv(T020_RACE)
    keep = set(map(tuple, t020[t020["usable"] & t020["dry_for_analysis"]
                                & (~t020["has_red_flag"])][["year", "round"]].values))
    laps = laps[[(int(y), int(r)) in keep for y, r in zip(laps["year"], laps["round"])]]
    n_pairs = 0
    n_ov = 0
    deltas = []
    for (year, rnd), g in laps.groupby(["year", "round"]):
        drivers, states = build_positions(g)
        if not states:
            continue
        flags = flags_map(g)
        maxlap = max(states)
        for L in range(1, maxlap):
            r0, r1 = states.get(L), states.get(L + 1)
            if not r0 or not r1:
                continue
            inv = {pos: drv for drv, pos in r0.items()}
            for p in range(1, len(r0)):
                front, rear = inv.get(p), inv.get(p + 1)
                if front is None or rear is None:
                    continue
                gap = drivers[rear].get(L, np.nan) - drivers[front].get(L, np.nan)
                if not np.isfinite(gap) or gap >= 1.0:
                    continue
                if blocked(flags, front, rear, L):
                    continue
                f, r = flags.get((front, L + 1)), flags.get((rear, L + 1))
                if f is None or r is None or not (f[4] and r[4]):
                    continue
                n_pairs += 1
                deltas.append(f[5] - r[5])
                if r1.get(rear) == p and r1.get(front) == p + 1:
                    n_ov += 1
    check("probability pair count = 13938", n_pairs == 13938, str(n_pairs))
    check("probability overall overtake rate = 10.15%",
          abs(n_ov / n_pairs - 0.1015) < 5e-4, f"{n_ov}/{n_pairs} = {n_ov / n_pairs:.4%}")
    prob = pd.read_csv(OUT / "overtake_probability.csv")
    tot = prob[prob["赛道类型"] == "全部"]
    check("probability table total pair count", int(tot["车对数"].sum()) == n_pairs,
          str(int(tot["车对数"].sum())))
    check("probability table >1 bin ~81%",
          abs(float(tot.loc[tot["Δ 分箱"] == ">1", "超车比例"].iloc[0]) - 0.8106) < 1e-3)
    blocked_tab = pd.read_csv(OUT / "blocked_cost.csv")
    b_all = blocked_tab[blocked_tab["赛道类型"] == "全部"]
    check("blocked-cost rows", len(b_all) == 5)
    check("blocked <0 bin mean ~0.60s",
          abs(float(b_all.loc[b_all["Δ 分箱"] == "<0", "损失均值"].iloc[0]) - 0.6049) < 1e-3)


# ---------------------------------------------------------------- 起步
def check_start() -> None:
    laps = pd.read_parquet(LAPS)
    t020 = pd.read_csv(T020_RACE)
    strat = pd.read_csv(T020_STRAT)[["year", "round", "driver", "start_compound"]]
    keep = set(map(tuple, t020[t020["usable"] & t020["dry_for_analysis"]
                                & (~t020["has_red_flag"])][["year", "round"]].values))
    laps = laps[[(int(y), int(r)) in keep for y, r in zip(laps["year"], laps["round"])]]
    rows = []
    for (year, rnd), g in laps.groupby(["year", "round"]):
        g1 = g[g["lap"] == 1]
        t1 = {r.driver: float(r.lap_time_s) for r in g1.itertuples(index=False)
              if pd.notna(r.lap_time_s)}
        if len(t1) < 5:
            continue
        med = float(np.median(list(t1.values())))
        ranks = {d: i + 1 for i, d in enumerate(sorted(t1, key=lambda d: t1[d]))}
        for drv, pos in ranks.items():
            rows.append({"year": int(year), "round": int(rnd), "driver": drv,
                         "pos_lap1": pos, "lap1_rel": t1[drv] - med})
    start = pd.DataFrame(rows).merge(strat, on=["year", "round", "driver"], how="left")
    start = start[start["start_compound"].isin(["SOFT", "MEDIUM", "HARD"])]
    grids = []
    for f in glob.glob(str(JOLPICA / "*_results_limit=100.json")):
        stem = Path(f).name.replace("_results_limit=100.json", "")
        y, r = stem.split("_")
        race = json.load(open(f, encoding="utf-8"))["MRData"]["RaceTable"]["Races"][0]
        for x in race["Results"]:
            grids.append({"year": int(y), "round": int(r), "driver": x["Driver"].get("code"),
                          "grid": x.get("grid")})
    grids = pd.DataFrame(grids)
    grids["grid"] = pd.to_numeric(grids["grid"], errors="coerce")
    start = start.merge(grids, on=["year", "round", "driver"], how="left")
    start = start[(start["grid"].notna()) & (start["grid"] >= 1)]
    start["pos_change"] = start["pos_lap1"] - start["grid"]
    by_comp = pd.read_csv(OUT / "start_by_compound.csv")
    for comp, mean in [("SOFT", -0.3797), ("MEDIUM", -0.0299), ("HARD", -0.5508)]:
        got = float(start[start["start_compound"] == comp]["pos_change"].mean())
        check(f"start mean pos change {comp}", abs(got - mean) < 1e-3, f"{got:.4f}")
    check("start usable rows = 1438", len(start) == 1438, str(len(start)))
    grid_tab = pd.read_csv(OUT / "start_grid_compound.csv")
    check("grid x compound cells = 12", len(grid_tab) == 12)
    row = grid_tab[(grid_tab["发车位区间"] == "16–20") & (grid_tab["起步配方"] == "SOFT")].iloc[0]
    got = float(start[(start["grid"].between(16, 20))
                      & (start["start_compound"] == "SOFT")]["pos_change"].mean())
    check("16-20 SOFT mean recompute", abs(float(row["名次变化均值"]) - got) < 1e-9,
          f"{got:.3f}")


# ---------------------------------------------------------------- SC 重启
def check_restart() -> None:
    restart = pd.read_csv(OUT / "sc_restart_spacing.csv")
    sc = restart[restart["type"] == "SC"]
    check("SC restart rows = 41", len(sc) == 41, str(len(sc)))
    summary = pd.read_csv(OUT / "sc_restart_summary.csv")
    sc_row = summary[summary["类型"] == "SC"].iloc[0]
    check("SC restart summary median ~0.81",
          abs(float(sc_row["平均间距中位数"]) - 0.8146) < 1e-3)
    check("SC restart summary count = 41", int(sc_row["段数"]) == 41)
    laps = pd.read_parquet(LAPS)
    ep = pd.read_parquet(EPISODES)
    t020 = pd.read_csv(T020_RACE)
    keep = set(map(tuple, t020[t020["usable"] & t020["dry_for_analysis"]
                                & (~t020["has_red_flag"])][["year", "round"]].values))
    laps = laps[[(int(y), int(r)) in keep for y, r in zip(laps["year"], laps["round"])]]
    # 独立重算一段：2023 R2
    g = laps[(laps["year"] == 2023) & (laps["round"] == 2)]
    drivers, states = build_positions(g)
    ev = ep[(ep["year"] == 2023) & (ep["round"] == 2) & (ep["type"] == "SC")].iloc[0]
    lap = int(ev["end_lap"]) + 1
    order = sorted(states[lap].items(), key=lambda kv: kv[1])
    times = [drivers[d][lap] for d, _ in order if lap in drivers.get(d, {})]
    gaps = [times[i + 1] - times[i] for i in range(9)]
    got = float(np.mean(gaps))
    row = sc[(sc["year"] == 2023) & (sc["round"] == 2)].iloc[0]
    check("2023 R2 SC restart gap recompute", abs(got - float(row["mean_gap"])) < 1e-6,
          f"{got:.3f}")


# ---------------------------------------------------------------- 报告
def check_report() -> None:
    text = (HERE / "report.md").read_text(encoding="utf-8")
    for needle in ["74", "73", "1631", "22.3", "yas_marina", "monaco", "0.5–1",
                   "81.1%", "0.60", "SOFT", "MEDIUM", "HARD", "0.81", "VSC"]:
        check(f"report contains {needle}", needle in text)


def main() -> None:
    check_files()
    check_two_runs()
    check_overtakes()
    check_probability()
    check_start()
    check_restart()
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
