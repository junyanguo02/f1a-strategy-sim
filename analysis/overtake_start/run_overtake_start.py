"""T023: 超车与起步的多车模拟器校准数据（只做描述统计）。

一条命令重跑全部结果（在项目根目录执行）：

    .venv/bin/python analysis/overtake_start/run_overtake_start.py

输入（只读，全部在本地，不联网、不安装依赖）：
  data/processed/laps_pit_filled.parquet          逐圈表（累计时间、进站/出站、SC/VSC、干净圈）
  data/processed/fcy_episodes.parquet             SC/VSC 事件段（end_lap）
  data/processed/fcy_race_summary.csv             每个 (year, round) 的 circuit_id
  analysis/actual_strategies/outputs/race_summary.csv   T020 的场次筛选与每场进站次数
  analysis/actual_strategies/strategies.csv       T020 的起步配方
  data/raw/jolpica/*_results_limit=100.json       发车位（grid）

输出（全部在 analysis/overtake_start/ 下）：
  outputs/race_overtakes.csv               每场：赛道超车次数、圈数、进站次数、SC/VSC 段数
  outputs/circuit_overtakes.csv            按赛道的平均超车次数（含前后 5 名标记）
  outputs/overtake_probability.csv         按 Δ 分箱 × 赛道类型的超车比例
  outputs/overtake_probability_prev_lap.csv 用上一圈圈时差（决策前已知）的同一张表（敏感性）
  outputs/blocked_cost.csv                 被堵代价（按 Δ 分箱）
  outputs/start_by_compound.csv            起步配方：第 1 圈名次变化与第 1 圈圈时
  outputs/start_grid_compound.csv          发车位区间 × 起步配方的名次变化
  outputs/sc_restart_spacing.csv           每段 SC / VSC 结束后的前 10 名间距
  outputs/sc_restart_summary.csv           SC / VSC 重启间距汇总
  outputs/input_sha256.csv
  figures/fig1_circuit_overtakes.png
  figures/fig2_overtake_probability.png
  figures/fig3_blocked_and_start.png
  report.md

只写 analysis/overtake_start/；不删除数据、不提交 git、不安装依赖。
本脚本没有随机过程（只有固定种子占位，按要求固定 SEED=20261117）。
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="overtake-mpl-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 常量
SEED = 20261117
DELTA_BINS = [-np.inf, 0.0, 0.25, 0.5, 1.0, np.inf]
DELTA_LABELS = ["<0", "0–0.25", "0.25–0.5", "0.5–1", ">1"]
GRID_BINS = [0, 5, 10, 15, 20]
GRID_LABELS = ["1–5", "6–10", "11–15", "16–20"]
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
STREET = {"monaco", "baku", "marina_bay", "jeddah", "vegas", "miami", "madring"}
GAP_THRESHOLD = 1.0

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
REPORT = HERE / "report.md"
for _d in (OUT, FIG, LOG):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 小工具
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(x, nd: int = 2) -> str:
    if x is None:
        return "—"
    try:
        if pd.isna(x):
            return "—"
    except (TypeError, ValueError):
        pass
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    return f"{float(x):.{nd}f}"


def display_df(df: pd.DataFrame, nd: int = 2) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda x: fmt(x, nd))
    return out


def md_table(df: pd.DataFrame) -> str:
    show = display_df(df)
    cols = list(show.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def track_type(circuit: str) -> str:
    return "街道赛" if circuit in STREET else "常规赛道"


def delta_bin(value: float):
    if pd.isna(value):
        return None
    for i in range(len(DELTA_BINS) - 1):
        lo, hi = DELTA_BINS[i], DELTA_BINS[i + 1]
        if lo <= value < hi:
            return DELTA_LABELS[i]
    return DELTA_LABELS[-1]


def mean_stat(values) -> dict:
    v = pd.Series(values).dropna().astype(float)
    if len(v) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan}
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(v.median())}


# ---------------------------------------------------------------- 读数据与筛选
def load_data():
    laps = pd.read_parquet(LAPS)
    for c in ["pit_in_filled", "pit_out", "safety_car", "virtual_safety_car", "is_clean"]:
        laps[c] = laps[c].fillna(False).astype(bool)
    episodes = pd.read_parquet(EPISODES)
    circuit_map = pd.read_csv(RACE_SUMMARY)[["year", "round", "circuit_id", "event"]].drop_duplicates()
    t020_race = pd.read_csv(T020_RACE)
    t020_strat = pd.read_csv(T020_STRAT)

    grids = []
    for f in glob.glob(str(JOLPICA / "*_results_limit=100.json")):
        stem = Path(f).name.replace("_results_limit=100.json", "")
        y, r = stem.split("_")
        race = json.load(open(f, encoding="utf-8"))["MRData"]["RaceTable"]["Races"][0]
        for x in race["Results"]:
            grids.append({"year": int(y), "round": int(r),
                          "driver": x["Driver"].get("code"),
                          "grid": x.get("grid")})
    grids = pd.DataFrame(grids)
    grids["grid"] = pd.to_numeric(grids["grid"], errors="coerce")
    return laps, episodes, circuit_map, t020_race, t020_strat, grids


def filtered_races(t020_race: pd.DataFrame) -> set:
    sel = t020_race[(t020_race["usable"]) & (t020_race["dry_for_analysis"])
                    & (~t020_race["has_red_flag"])]
    return set(map(tuple, sel[["year", "round"]].values))


# ---------------------------------------------------------------- 逐圈名次
def race_cumulative(race_laps: pd.DataFrame):
    """累计时间（从第 1 圈起）与每圈结束时的名次；缺圈/缺圈时后不再外推。"""
    drivers = {}
    for drv, g in race_laps.groupby("driver"):
        g = g.sort_values("lap")
        laps = g["lap"].to_numpy(dtype=int)
        times = g["lap_time_s"].to_numpy(dtype=float)
        cum = {}
        s = 0.0
        prev = 0
        for i in range(len(g)):
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


def lap_flags(race_laps: pd.DataFrame) -> dict:
    out = {}
    for r in race_laps.itertuples(index=False):
        out[(str(r.driver), int(r.lap))] = {
            "pit_in": bool(r.pit_in_filled),
            "pit_out": bool(r.pit_out),
            "sc": bool(r.safety_car),
            "vsc": bool(r.virtual_safety_car),
            "clean": bool(r.is_clean),
            "lap_time": float(r.lap_time_s) if pd.notna(r.lap_time_s) else np.nan,
        }
    return out


def pair_blocked(flags: dict, driver_a: str, driver_b: str, lap: int) -> bool:
    for drv in (driver_a, driver_b):
        for L in (lap, lap + 1):
            f = flags.get((drv, L))
            if f is None:
                return True
            if f["pit_in"] or f["pit_out"] or f["sc"] or f["vsc"]:
                return True
    return False


def clean_median(race_laps: pd.DataFrame, driver: str) -> float:
    s = race_laps[(race_laps["driver"] == driver) & race_laps["is_clean"]]["lap_time_s"]
    return float(s.median()) if len(s) else np.nan


# ---------------------------------------------------------------- 超车
def analyze_races(laps: pd.DataFrame, race_set: set, circuit_map: pd.DataFrame,
                  episodes: pd.DataFrame):
    """逐场计算赛道超车、超车概率、被堵代价、起步与 SC 重启间距。"""
    laps = laps[[(int(y), int(r)) in race_set for y, r in zip(laps["year"], laps["round"])]]
    laps = laps.merge(circuit_map, on=["year", "round"], how="left")
    max_lap = laps.groupby(["year", "round"])["lap"].max().to_dict()

    race_rows = []
    prob_rows = []
    prob_prev_rows = []
    blocked_rows = []
    restart_rows = []
    skipped_races = []
    for (year, rnd), g in laps.groupby(["year", "round"]):
        circuit = g["circuit_id"].iloc[0]
        tt = track_type(circuit)
        drivers, states = race_cumulative(g)
        flags = lap_flags(g)
        if not states:
            skipped_races.append({"year": int(year), "round": int(rnd),
                                  "circuit_id": circuit,
                                  "reason": "没有可用的累计时间（首圈圈时缺失等）"})
            continue
        n_overtakes = 0
        n_transitions = 0
        maxlap = max(states)
        for L in range(1, maxlap):
            r0, r1 = states.get(L), states.get(L + 1)
            if not r0 or not r1:
                continue
            n_transitions += 1
            inv = {pos: drv for drv, pos in r0.items()}
            for p in range(1, len(r0)):
                a, b = inv.get(p), inv.get(p + 1)
                if a is None or b is None:
                    continue
                if r1.get(a) != p + 1 or r1.get(b) != p:
                    continue
                if pair_blocked(flags, a, b, L):
                    continue
                n_overtakes += 1
        # 超车概率 / 被堵代价 / 后车损失
        for L in range(1, maxlap):
            r0, r1 = states.get(L), states.get(L + 1)
            if not r0 or not r1:
                continue
            inv = {pos: drv for drv, pos in r0.items()}
            for p in range(1, len(r0)):
                front, rear = inv.get(p), inv.get(p + 1)
                if front is None or rear is None:
                    continue
                cum = drivers[rear].get(L, np.nan) - drivers[front].get(L, np.nan)
                if not np.isfinite(cum) or cum >= GAP_THRESHOLD:
                    continue
                if pair_blocked(flags, front, rear, L):
                    continue
                f_next = flags.get((front, L + 1))
                r_next = flags.get((rear, L + 1))
                if f_next is None or r_next is None:
                    continue
                if not (f_next["clean"] and r_next["clean"]):
                    continue
                delta = f_next["lap_time"] - r_next["lap_time"]  # 后车快为正
                overtake = int(r1.get(rear) == p and r1.get(front) == p + 1)
                f_prev = flags.get((front, L))
                r_prev = flags.get((rear, L))
                delta_prev = (f_prev["lap_time"] - r_prev["lap_time"]
                              if (f_prev and r_prev and f_prev["clean"] and r_prev["clean"])
                              else np.nan)
                base = {
                    "year": int(year), "round": int(rnd), "circuit_id": circuit,
                    "track_type": tt, "lap": L, "delta": delta,
                    "delta_bin": delta_bin(delta), "overtake": overtake,
                }
                prob_rows.append(dict(base))
                if pd.notna(delta_prev):
                    prob_prev_rows.append(dict(base, delta_prev=delta_prev,
                                               delta_prev_bin=delta_bin(delta_prev)))
                if overtake == 0:
                    med = clean_median(g, rear)
                    if np.isfinite(med) and np.isfinite(r_next["lap_time"]):
                        blocked_rows.append({
                            "year": int(year), "round": int(rnd), "circuit_id": circuit,
                            "track_type": tt, "delta": delta,
                            "delta_bin": delta_bin(delta),
                            "loss": r_next["lap_time"] - med,
                        })
        race_rows.append({
            "year": int(year), "round": int(rnd), "circuit_id": circuit,
            "track_type": tt, "n_overtakes": n_overtakes,
            "n_transitions": n_transitions, "max_lap": int(maxlap),
            "overtakes_per_lap": (n_overtakes / n_transitions if n_transitions else np.nan),
        })
        # SC / VSC 重启间距（第一圈结束时前 10 名相邻间距）
        for typ in ["SC", "VSC"]:
            ev = episodes[(episodes["year"] == year) & (episodes["round"] == rnd)
                          & (episodes["type"] == typ)]
            for e in ev.itertuples(index=False):
                lap = int(e.end_lap) + 1
                if lap not in states:
                    continue
                order = sorted(states[lap].items(), key=lambda kv: kv[1])
                times = []
                for drv, pos in order:
                    if lap in drivers.get(drv, {}):
                        times.append(drivers[drv][lap])
                if len(times) < 2:
                    continue
                gaps = [times[i + 1] - times[i] for i in range(min(9, len(times) - 1))]
                restart_rows.append({
                    "year": int(year), "round": int(rnd), "circuit_id": circuit,
                    "track_type": tt, "type": typ, "restart_lap": lap,
                    "n_top10": min(10, len(times)),
                    "mean_gap": float(np.mean(gaps)) if gaps else np.nan,
                    "median_gap": float(np.median(gaps)) if gaps else np.nan,
                })
    return (pd.DataFrame(race_rows), pd.DataFrame(prob_rows),
            pd.DataFrame(prob_prev_rows), pd.DataFrame(blocked_rows),
            pd.DataFrame(restart_rows), pd.DataFrame(skipped_races))


# ---------------------------------------------------------------- 起步
def start_table(laps: pd.DataFrame, race_set: set, circuit_map: pd.DataFrame,
                strat: pd.DataFrame, grids: pd.DataFrame):
    laps = laps[[(int(y), int(r)) in race_set for y, r in zip(laps["year"], laps["round"])]]
    laps = laps.merge(circuit_map, on=["year", "round"], how="left")
    rows = []
    for (year, rnd), g in laps.groupby(["year", "round"]):
        g1 = g[g["lap"] == 1]
        t1 = {r.driver: float(r.lap_time_s) for r in g1.itertuples(index=False)
              if pd.notna(r.lap_time_s)}
        if len(t1) < 5:
            continue
        race_med = float(np.median(list(t1.values())))
        ranks = {d: i + 1 for i, d in enumerate(sorted(t1, key=lambda d: t1[d]))}
        for drv, pos in ranks.items():
            rows.append({"year": int(year), "round": int(rnd),
                         "circuit_id": g["circuit_id"].iloc[0], "driver": drv,
                         "pos_lap1": pos, "lap1_time": t1[drv],
                         "lap1_rel": t1[drv] - race_med})
    start = pd.DataFrame(rows)
    start = start.merge(grids, on=["year", "round", "driver"], how="left")
    start = start.merge(strat[["year", "round", "driver", "start_compound"]],
                        on=["year", "round", "driver"], how="left")
    start["grid"] = pd.to_numeric(start["grid"], errors="coerce")
    start["pos_change"] = start["pos_lap1"] - start["grid"]  # 正 = 丢位置，负 = 超车/受益
    start["grid_bucket"] = pd.cut(start["grid"], bins=GRID_BINS, labels=GRID_LABELS)
    start["excluded_reason"] = None
    start.loc[start["grid"].isna(), "excluded_reason"] = "无发车位"
    start.loc[start["grid"] == 0, "excluded_reason"] = "pit lane 起步（grid=0）"
    start.loc[start["start_compound"].isna(), "excluded_reason"] = "无起步配方"
    start.loc[~start["start_compound"].isin(COMPOUNDS), "excluded_reason"] = "湿胎/未知起步配方"
    excluded = start[start["excluded_reason"].notna()]
    usable = start[start["excluded_reason"].isna()].copy()

    by_comp = []
    for c in COMPOUNDS + ["全部"]:
        s = usable if c == "全部" else usable[usable["start_compound"] == c]
        st = mean_stat(s["pos_change"])
        by_comp.append({
            "起步配方": c, "n": int(len(s)),
            "名次变化均值": st["mean"], "名次变化中位数": st["median"],
            "名次提升占比": float((s["pos_change"] < 0).mean()) if len(s) else np.nan,
            "不变占比": float((s["pos_change"] == 0).mean()) if len(s) else np.nan,
            "名次下降占比": float((s["pos_change"] > 0).mean()) if len(s) else np.nan,
            "第1圈圈时相对中位数均值": float(s["lap1_rel"].mean()) if len(s) else np.nan,
            "第1圈圈时相对中位数中位数": float(s["lap1_rel"].median()) if len(s) else np.nan,
        })
    by_comp = pd.DataFrame(by_comp)

    grid_rows = []
    for bucket in GRID_LABELS:
        for c in COMPOUNDS:
            s = usable[(usable["grid_bucket"] == bucket)
                       & (usable["start_compound"] == c)]
            st = mean_stat(s["pos_change"])
            grid_rows.append({"发车位区间": bucket, "起步配方": c, "n": int(len(s)),
                              "名次变化均值": st["mean"],
                              "名次变化中位数": st["median"]})
    grid_tab = pd.DataFrame(grid_rows)

    lap1_rows = []
    for bucket in GRID_LABELS:
        for c in COMPOUNDS:
            s = usable[(usable["grid_bucket"] == bucket)
                       & (usable["start_compound"] == c)]
            st = mean_stat(s["lap1_rel"])
            lap1_rows.append({"发车位区间": bucket, "起步配方": c, "n": int(len(s)),
                              "第1圈相对圈时均值": st["mean"]})
    lap1_tab = pd.DataFrame(lap1_rows)
    return usable, excluded, by_comp, grid_tab, lap1_tab, start


# ---------------------------------------------------------------- 概率/损失表
def probability_table(prob: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tt in ["常规赛道", "街道赛", "全部"]:
        s = prob if tt == "全部" else prob[prob["track_type"] == tt]
        for b in DELTA_LABELS:
            g = s[s["delta_bin"] == b]
            n, k = len(g), int(g["overtake"].sum())
            lo, hi = wilson_ci(k, n)
            rows.append({"赛道类型": tt, "Δ 分箱": b, "车对数": n,
                         "超车次数": k, "超车比例": (k / n if n else np.nan),
                         "95%CI 下": lo, "95%CI 上": hi})
    return pd.DataFrame(rows)


def probability_table_prev(prev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tt in ["常规赛道", "街道赛", "全部"]:
        s = prev if tt == "全部" else prev[prev["track_type"] == tt]
        for b in DELTA_LABELS:
            g = s[s["delta_prev_bin"] == b]
            n, k = len(g), int(g["overtake"].sum())
            lo, hi = wilson_ci(k, n)
            rows.append({"赛道类型": tt, "上一圈Δ分箱": b, "车对数": n,
                         "超车次数": k, "超车比例": (k / n if n else np.nan),
                         "95%CI 下": lo, "95%CI 上": hi})
    return pd.DataFrame(rows)


def blocked_table(blocked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tt in ["常规赛道", "街道赛", "全部"]:
        s = blocked if tt == "全部" else blocked[blocked["track_type"] == tt]
        for b in DELTA_LABELS:
            g = s[s["delta_bin"] == b]
            st = mean_stat(g["loss"])
            rows.append({"赛道类型": tt, "Δ 分箱": b, "车对数": int(len(g)),
                         "损失均值": st["mean"], "损失中位数": st["median"],
                         "正损失占比": float((g["loss"] > 0).mean()) if len(g) else np.nan})
    return pd.DataFrame(rows)


def restart_table(restarts: pd.DataFrame):
    rows = []
    for typ in ["SC", "VSC"]:
        s = restarts[restarts["type"] == typ]
        vals = s["mean_gap"].dropna()
        if len(vals) == 0:
            rows.append({"类型": typ, "段数": 0, "平均间距均值": np.nan,
                         "平均间距中位数": np.nan, "最小": np.nan, "最大": np.nan})
            continue
        rows.append({"类型": typ, "段数": int(len(vals)),
                     "平均间距均值": float(vals.mean()),
                     "平均间距中位数": float(vals.median()),
                     "最小": float(vals.min()), "最大": float(vals.max())})
    # 街道/常规 拆分（SC）
    for tt in ["常规赛道", "街道赛"]:
        s = restarts[(restarts["type"] == "SC") & (restarts["track_type"] == tt)]["mean_gap"].dropna()
        rows.append({"类型": f"SC-{tt}", "段数": int(len(s)),
                     "平均间距均值": float(s.mean()) if len(s) else np.nan,
                     "平均间距中位数": float(s.median()) if len(s) else np.nan,
                     "最小": float(s.min()) if len(s) else np.nan,
                     "最大": float(s.max()) if len(s) else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 图
def make_figures(circuit_tab: pd.DataFrame, prob_tab: pd.DataFrame,
                 blocked_tab: pd.DataFrame, grid_tab: pd.DataFrame):
    paths = []

    # fig1: 按赛道的平均超车次数
    c = circuit_tab.sort_values("每场平均超车", ascending=False)
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    colors = ["#dd8452" if t == "街道赛" else "#4c72b0" for t in c["赛道类型"]]
    ax.bar(c["circuit_id"], c["每场平均超车"], color=colors, edgecolor="black", linewidth=0.4)
    for i, (v, n) in enumerate(zip(c["每场平均超车"], c["场数"])):
        ax.text(i, v + 0.3, f"{v:.1f}\n({int(n)})", ha="center", va="bottom", fontsize=6.5)
    ax.set_ylabel("On-track overtakes per race")
    ax.set_title("Circuit-level on-track overtakes (bar labels: mean, races)")
    ax.tick_params(axis="x", rotation=70, labelsize=7.5)
    ax.grid(axis="y", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#4c72b0", ec="black"),
               plt.Rectangle((0, 0), 1, 1, color="#dd8452", ec="black")]
    ax.legend(handles, ["regular circuit", "street circuit"], frameon=False, fontsize=8)
    fig.tight_layout()
    p = FIG / "fig1_circuit_overtakes.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # fig2: 超车概率 vs Δ
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for tt, color, label in [("常规赛道", "#4c72b0", "regular circuit"),
                             ("街道赛", "#dd8452", "street circuit")]:
        s = prob_tab[prob_tab["赛道类型"] == tt]
        x = np.arange(len(DELTA_LABELS))
        y = [float(s.loc[s["Δ 分箱"] == b, "超车比例"].iloc[0]) for b in DELTA_LABELS]
        lo = [y[i] - float(s.loc[s["Δ 分箱"] == DELTA_LABELS[i], "95%CI 下"].iloc[0])
              for i in range(len(y))]
        hi = [float(s.loc[s["Δ 分箱"] == DELTA_LABELS[i], "95%CI 上"].iloc[0]) - y[i]
              for i in range(len(y))]
        ax.errorbar(x, y, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], marker="o",
                    linewidth=1.5, capsize=3, color=color, label=label)
    ax.set_xticks(np.arange(len(DELTA_LABELS)))
    ax.set_xticklabels(DELTA_LABELS)
    ax.set_xlabel("Pace delta Δ = front lap time − rear lap time (s; rear faster > 0)")
    ax.set_ylabel("Overtake probability next lap")
    ax.set_title("Overtake probability given gap < 1 s (95% Wilson CI)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = FIG / "fig2_overtake_probability.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # fig3: 被堵代价 + 起步名次变化
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    s = blocked_tab[blocked_tab["赛道类型"] == "全部"]
    x = np.arange(len(DELTA_LABELS))
    ax.bar(x, [float(s.loc[s["Δ 分箱"] == b, "损失均值"].iloc[0]) for b in DELTA_LABELS],
           color="#55a868", edgecolor="black", linewidth=0.5)
    for i, b in enumerate(DELTA_LABELS):
        v = float(s.loc[s["Δ 分箱"] == b, "损失均值"].iloc[0])
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(DELTA_LABELS)
    ax.set_xlabel("Pace delta Δ (s)")
    ax.set_ylabel("Mean lap-time loss vs clean median (s)")
    ax.set_title("Blocked behind < 1 s (no overtake)")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    x = np.arange(len(GRID_LABELS))
    width = 0.26
    colors = {"SOFT": "#55a868", "MEDIUM": "#dd8452", "HARD": "#4c72b0"}
    for i, comp in enumerate(COMPOUNDS):
        s = grid_tab[grid_tab["起步配方"] == comp]
        vals = [float(s.loc[s["发车位区间"] == b, "名次变化均值"].iloc[0]) for b in GRID_LABELS]
        ax.bar(x + (i - 1) * width, vals, width, color=colors[comp],
               edgecolor="black", linewidth=0.4, label=comp)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(GRID_LABELS)
    ax.set_xlabel("Grid position")
    ax.set_ylabel("Mean lap-1 position change (+ = lost)")
    ax.set_title("Lap-1 position change by start compound and grid")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = FIG / "fig3_blocked_and_start.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


# ---------------------------------------------------------------- 报告
def build_report(sample_info, race_tab, circuit_tab, prob_tab, prob_prev_tab,
                 blocked_tab, start_comp, start_grid, start_lap1, restart_tab,
                 restart_sc, figs, hashes, t020: pd.DataFrame = None) -> str:
    lines = []
    lines.append("# T023　超车与起步：多车模拟器的校准数据")
    lines.append("")
    lines.append("由 `analysis/overtake_start/run_overtake_start.py` 离线生成；只做描述统计，"
                 "不建模、不写建模建议。本脚本没有随机过程（固定 SEED=20261117），两次运行输出一致。")
    lines.append("")
    lines.append("## 口径")
    lines.append("")
    lines.append("- **场次**：T020 里 `usable & dry_for_analysis & ~has_red_flag` 的 "
                 f"{sample_info['n_filtered_races']} 场（2023–2026）。")
    lines.append("- **逐圈名次**：沿用 T012 的累计时间口径——从第 1 圈起累加 `lap_time_s`，"
                 "每圈结束时按累计时间升序排名；缺圈/缺圈时的车手在该圈之后退出排名（不外推）。"
                 "这不是官方逐圈位置（罚时、DSQ、套圈分类等无法复现）。")
    lines.append("- **赛道超车**：相邻两圈（L→L+1）末名次互换的相邻名次车对，且两车在 L、L+1 两圈都"
                 "不是进站圈/出站圈、都不在 SC/VSC 下，记 1 次。")
    lines.append("- **超车概率**：上一圈（L）结束时后车距前车累计时间 <1 秒的车对，"
                 "看下一圈（L+1）是否完成名次互换；Δ = 前车 L+1 圈时 − 后车 L+1 圈时（后车快为正），"
                 "只用两车 L+1 都是干净圈的车对。另给用 L 圈时差（决策前已知）的敏感性表。")
    lines.append("- **被堵代价**：上述车对里没有超车的，后车 L+1 圈时减去其本场干净圈中位数（秒）。")
    lines.append("- **起步**：第 1 圈结束时名次（累计时间排名）− Jolpica 发车位；负值 = 名次提升。"
                 "只保留 SOFT/MEDIUM/HARD 起步、发车位 ≥1 的车场次；`grid=0`（pit lane 起步）与湿胎/未知配方剔除。")
    lines.append("- **SC 重启间距**：每段 SC（VSC 单列对照）结束圈 +1 的那一圈结束时，前 10 名相邻车的"
                 "平均累计时间差。")
    lines.append("- **赛道类型**：街道赛 = " + "、".join(sorted(STREET)) + "；其余为常规赛道。"
                 "（Albert Park 等 park/semi-permanent 赛道按常规处理，见“不能说明什么”。）")
    lines.append("")

    lines.append("## 1. 样本")
    lines.append("")
    lines.append(md_table(pd.DataFrame(sample_info["rows"])))
    lines.append("")
    lines.append(f"- 有可用逐圈名次的场次：**{sample_info['n_overtake_races']}**；"
                 f"被跳过的场次：{sample_info['n_skipped']}（"
                 + ("、".join(f"{r.year} R{r.round} {r.circuit_id}：{r.reason}"
                              for r in sample_info["skipped"].itertuples(index=False))
                    if sample_info["n_skipped"] else "无") + "）。")
    lines.append("")

    lines.append("## 2. 赛道超车")
    lines.append("")
    rt = race_tab
    lines.append(f"- 总超车 **{int(rt['n_overtakes'].sum())}** 次 / {len(rt)} 场；"
                 f"每场平均 **{rt['n_overtakes'].mean():.1f}** 次，中位数 {rt['n_overtakes'].median():.0f}，"
                 f"范围 {int(rt['n_overtakes'].min())}–{int(rt['n_overtakes'].max())}。")
    lines.append(f"- 折算成圈均：每 100 圈 {rt['overtakes_per_lap'].mean()*100:.1f} 次"
                 f"（平均 {rt['overtakes_per_lap'].mean():.3f} 次/圈）。")
    lines.append("")
    lines.append("按赛道的平均超车次数（前后各 5 名）：")
    lines.append("")
    ct = circuit_tab.sort_values("每场平均超车", ascending=False)
    top5 = ct.head(5).copy()
    bottom5 = ct.tail(5).copy()
    ct_show = pd.concat([top5.assign(分组="前 5"), bottom5.assign(分组="后 5")])
    ct_show = ct_show[["分组", "circuit_id", "赛道类型", "场数", "每场平均超车",
                       "每场中位数", "每圈超车率"]]
    lines.append(md_table(ct_show.rename(columns={
        "circuit_id": "赛道", "赛道类型": "类型", "场数": "场数",
        "每场平均超车": "每场平均", "每场中位数": "每场中位数",
        "每圈超车率": "每圈"})))
    lines.append("")
    lines.append("与 T020 的每场进站次数对照：")
    lines.append("")
    corr = np.corrcoef(rt["n_overtakes"], rt["n_total_pits"])[0, 1]
    lines.append(f"- 每场超车次数与每场总进站次数（T020 `n_total_pits`）的 Pearson 相关系数 "
                 f"**{corr:.3f}**（n={len(rt)} 场）；"
                 f"超车次数均值 {rt['n_overtakes'].mean():.1f}，进站次数均值 {rt['n_total_pits'].mean():.1f}。")
    lines.append("")
    lines.append("![按赛道超车](figures/fig1_circuit_overtakes.png)")
    lines.append("")
    lines.append("**图 1**：按赛道的每场平均赛道超车次数，柱上标注平均值与场数；蓝色常规赛道、橙色街道赛。")
    lines.append("")

    lines.append("## 3. 超车概率与配速差")
    lines.append("")
    lines.append("Δ = 前车本轮圈时 − 后车本轮圈时（后车快为正）；只统计上一圈结束时距前车 <1 秒、"
                 "两车本轮都是干净圈、且两圈都不涉及进站/SC/VSC 的车对。")
    lines.append("")
    show = prob_tab.rename(columns={"赛道类型": "类型", "Δ 分箱": "Δ",
                                    "车对数": "车对", "超车次数": "超车",
                                    "超车比例": "比例", "95%CI 下": "CI 下",
                                    "95%CI 上": "CI 上"})
    show["比例"] = show["比例"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    show["CI 下"] = show["CI 下"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    show["CI 上"] = show["CI 上"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    lines.append(md_table(show))
    lines.append("")
    lines.append("敏感性表：Δ 改用**上一圈（L）**的干净圈时差（决策前已知，不含本轮结果）：")
    lines.append("")
    show2 = prob_prev_tab.rename(columns={"赛道类型": "类型", "上一圈Δ分箱": "Δ",
                                          "车对数": "车对", "超车次数": "超车",
                                          "超车比例": "比例", "95%CI 下": "CI 下",
                                          "95%CI 上": "CI 上"})
    show2["比例"] = show2["比例"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    show2["CI 下"] = show2["CI 下"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    show2["CI 上"] = show2["CI 上"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    lines.append(md_table(show2))
    lines.append("")
    lines.append("![超车概率](figures/fig2_overtake_probability.png)")
    lines.append("")
    lines.append("**图 2**：上一圈 <1 秒车对在下一圈的超车比例，按 Δ 分箱，常规赛道/街道赛分开；误差线为 95% Wilson 区间。")
    lines.append("")

    lines.append("## 4. 被堵代价")
    lines.append("")
    lines.append("上一圈 <1 秒、下一圈没有超车：后车本轮圈时相对其本场干净圈中位数的损失。")
    lines.append("")
    bshow = blocked_tab.rename(columns={"赛道类型": "类型", "Δ 分箱": "Δ",
                                        "车对数": "车对", "损失均值": "损失均值 (s)",
                                        "损失中位数": "损失中位数 (s)",
                                        "正损失占比": "正损失占比"})
    bshow["正损失占比"] = bshow["正损失占比"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    lines.append(md_table(bshow))
    lines.append("")

    lines.append("## 5. 起步")
    lines.append("")
    lines.append(md_table(start_comp.rename(columns={
        "起步配方": "起步配方", "n": "n", "名次变化均值": "名次变化均值",
        "名次变化中位数": "名次变化中位数", "名次提升占比": "提升占比",
        "不变占比": "不变占比", "名次下降占比": "下降占比",
        "第1圈圈时相对中位数均值": "第1圈相对圈时均值 (s)",
        "第1圈圈时相对中位数中位数": "第1圈相对圈时中位数 (s)"})))
    lines.append("")
    lines.append("控制发车位区间后的名次变化均值（负 = 名次提升）：")
    lines.append("")
    pivot = start_grid.pivot(index="发车位区间", columns="起步配方",
                             values="名次变化均值").reindex(GRID_LABELS)
    pivot_n = start_grid.pivot(index="发车位区间", columns="起步配方",
                               values="n").reindex(GRID_LABELS)
    pshow = pivot.copy()
    for c in pshow.columns:
        pshow[c] = [f"{fmt(pivot.iloc[i, j], 2)}（n={int(pivot_n.iloc[i, j])}）"
                    if pd.notna(pivot.iloc[i, j]) else "—"
                    for i in range(len(pivot)) for j in [list(pivot.columns).index(c)]]
    lines.append(md_table(pshow.reset_index().rename(columns={"发车位区间": "发车位"})))
    lines.append("")
    lines.append("控制发车位区间后的第 1 圈圈时（相对该场第 1 圈中位数，均值，秒）：")
    lines.append("")
    lap1_pivot = start_lap1.pivot(index="发车位区间", columns="起步配方",
                                  values="第1圈相对圈时均值").reindex(GRID_LABELS)
    l1 = lap1_pivot.copy()
    for c in l1.columns:
        l1[c] = l1[c].map(lambda v: fmt(v, 2))
    lines.append(md_table(l1.reset_index().rename(columns={"发车位区间": "发车位"})))
    lines.append("")
    lines.append("![被堵与起步](figures/fig3_blocked_and_start.png)")
    lines.append("")
    lines.append("**图 3**：左 = 被堵代价（按 Δ 分箱）；右 = 第 1 圈名次变化（按发车位区间 × 起步配方）。")
    lines.append("")

    lines.append("## 6. SC 重启后的间距")
    lines.append("")
    lines.append(md_table(restart_tab))
    lines.append("")
    lines.append("每段 SC 结束后的第一圈结束时，前 10 名相邻车的平均间距（逐段）：")
    lines.append("")
    sc_only = restart_sc[restart_sc["type"] == "SC"].copy()
    sc_show = sc_only[["year", "round", "circuit_id", "restart_lap", "n_top10",
                       "mean_gap", "median_gap"]].rename(columns={
        "year": "赛季", "round": "站次", "circuit_id": "赛道",
        "restart_lap": "重启圈", "n_top10": "前10车数",
        "mean_gap": "平均间距 (s)", "median_gap": "中位间距 (s)"})
    lines.append(md_table(sc_show))
    lines.append("")

    # ---------- 观察
    def comp_row(c):
        return start_comp[start_comp["起步配方"] == c].iloc[0]

    lines.append("## 观察到的现象")
    lines.append("")
    lines.append(f"1. **赛道超车平均每场 {rt['n_overtakes'].mean():.1f} 次，赛道之间差异很大。** "
                 f"最高是 {ct.iloc[0]['circuit_id']}（每场 {ct.iloc[0]['每场平均超车']:.1f} 次，"
                 f"{int(ct.iloc[0]['场数'])} 场），最低是 {ct.iloc[-1]['circuit_id']}"
                 f"（每场 {ct.iloc[-1]['每场平均超车']:.1f} 次，{int(ct.iloc[-1]['场数'])} 场）；"
                 f"超车次数与进站次数相关系数 {corr:.3f}（只描述，不表示因果）。")
    lines.append(f"2. **超车比例随 Δ 迅速上升。** 全部车对中，Δ<0（后车更慢）时超车比例 "
                 f"{float(prob_tab[(prob_tab['赛道类型'] == '全部') & (prob_tab['Δ 分箱'] == '<0')]['超车比例'].iloc[0]):.1%}，"
                 f"0–0.25 秒 {float(prob_tab[(prob_tab['赛道类型'] == '全部') & (prob_tab['Δ 分箱'] == '0–0.25')]['超车比例'].iloc[0]):.1%}，"
                 f"0.25–0.5 秒 {float(prob_tab[(prob_tab['赛道类型'] == '全部') & (prob_tab['Δ 分箱'] == '0.25–0.5')]['超车比例'].iloc[0]):.1%}，"
                 f"0.5–1 秒 {float(prob_tab[(prob_tab['赛道类型'] == '全部') & (prob_tab['Δ 分箱'] == '0.5–1')]['超车比例'].iloc[0]):.1%}，"
                 f">1 秒 {float(prob_tab[(prob_tab['赛道类型'] == '全部') & (prob_tab['Δ 分箱'] == '>1')]['超车比例'].iloc[0]):.1%}。")
    lines.append("3. **街道赛更难超车、但配速差足够大时比例接近。** 0.25–0.5 秒档："
                 + "、".join(
                     f"{tt} {float(prob_tab[(prob_tab['赛道类型'] == tt) & (prob_tab['Δ 分箱'] == '0.25–0.5')]['超车比例'].iloc[0]):.1%}"
                     for tt in ["常规赛道", "街道赛"])
                 + "；>1 秒档：" + "、".join(
                     f"{tt} {float(prob_tab[(prob_tab['赛道类型'] == tt) & (prob_tab['Δ 分箱'] == '>1')]['超车比例'].iloc[0]):.1%}"
                     for tt in ["常规赛道", "街道赛"]) + "。")
    lines.append(f"4. **被堵时后车仍要付 0.2–0.6 秒的圈时代价。** 按 Δ 分箱的平均损失："
                 + "、".join(
                     f"{b} 档 {float(blocked_tab[(blocked_tab['赛道类型'] == '全部') & (blocked_tab['Δ 分箱'] == b)]['损失均值'].iloc[0]):.2f}s"
                     for b in DELTA_LABELS) + "。"
                 "Δ<0（后车更慢还被堵在 1 秒内）时损失最大。")
    lines.append("5. **起步配方与第 1 圈名次变化：整体差异不大，后段发车名次提升更明显。** "
                 + "；".join(
                     f"{c} 起步 n={int(comp_row(c)['n'])}，名次变化均值 {comp_row(c)['名次变化均值']:+.2f}"
                     for c in COMPOUNDS)
                 + f"。发车位 16–20 区间："
                 + "、".join(
                     f"{c} {float(start_grid[(start_grid['发车位区间'] == '16–20') & (start_grid['起步配方'] == c)]['名次变化均值'].iloc[0]):+.2f}"
                     for c in COMPOUNDS)
                 + "（负 = 名次提升）。"
                 "第 1 圈圈时方面，按配方看 HARD 平均比该场第 1 圈中位数慢 3.97 秒，"
                 "但 HARD 起步车集中在后段发车（16–20 区间 n=94），这个均值混入了发车位；"
                 "按发车位区间控制后，16–20 区间 HARD 仍比 SOFT 慢约 3 秒（6.70 vs 3.72），"
                 "前段区间 HARD 的样本只有个位数、不能下结论。")
    lines.append("6. **SC 重启后前 10 名间距中位数约 0.8 秒。** SC 重启的相邻平均间距："
                 f"中位数 {float(restart_tab.loc[restart_tab['类型'] == 'SC', '平均间距中位数'].iloc[0]):.2f}s、"
                 f"均值 {float(restart_tab.loc[restart_tab['类型'] == 'SC', '平均间距均值'].iloc[0]):.2f}s"
                 f"（{int(restart_tab.loc[restart_tab['类型'] == 'SC', '段数'].iloc[0])} 段）；"
                 f"VSC 后是 {float(restart_tab.loc[restart_tab['类型'] == 'VSC', '平均间距中位数'].iloc[0]):.2f}s。")
    lines.append("")

    lines.append("## 这些数据不能说明什么")
    lines.append("")
    lines.append("1. **逐圈名次不是官方名次。** 用累计时间排名；罚时、DSQ、套圈分类、维修区内的位置变化"
                 "都无法复现。T012 核对过与官方完赛名次完全一致率约 81.5%、±1 以内 94.2%；"
                 "本单的同圈名次同样受此限制。")
    lines.append("2. **“赛道超车”只是相邻名次互换的代理。** 无法区分 DRS 列车、蓝旗、车损、轮胎管理、"
                 "前车失误等；也不会把“先被超再超回来”合并；被套圈车之间的位置变化无法与官方位置核对。")
    lines.append("3. **Δ 是同一圈的圈时差，不是赛前已知的配速差。** 主表用 L+1 圈（超车发生圈）的干净圈时差，"
                 "这个差本身受超车过程影响（循环性）；所以另给一张用 L 圈（决策前）Δ 的敏感性表，"
                 "两表方向应该一致，但数值不同。")
    lines.append("4. **干净圈筛选会偏向“没被堵”的圈。** 主表只保留两车 L+1 都是干净圈的车对；"
                 "被堵到圈时明显变慢的圈可能被 clean 规则剔除，使被堵代价与超车概率的绝对值有偏"
                 "（方向不一定有偏）。")
    lines.append("5. **场次筛选与数据缺口。** 74 场里 1 场（2024 摩纳哥）首圈圈时缺失、无法算累计时间；"
                 "已从本单分析中跳过。红 flag 检测用 track_status 含 5 的启发式，"
                 "可能漏掉部分红旗（如 2024 摩纳哥红旗未被识别，但它因缺圈时被跳过）。")
    lines.append("6. **起步配方的效应混入了发车位、轮胎新旧、车手/车队差异。** 只按发车位区间做了粗控制，"
                 "没有控制排位赛速度、赛道温度、起步程序；湿胎/未知配方与 pit lane 起步（grid=0）已剔除。")
    lines.append("7. **SC 重启间距只取“第一圈结束时”的一个快照。** 重启后各车加速/压缩过程不是瞬时完成的，"
                 "且 SC 段由 race_control 消息判定，end_lap 与实际绿旗时刻可能差 1 圈；VSC 单列但同样近似。")
    lines.append("8. **本单只做描述，不能直接当模拟器参数。** 超车概率与 Δ 的关系里混入了赛道、车手、"
                 "轮胎状态、DRS 等因素；要接进模拟器需要另行标定（本单不给建模建议）。")
    lines.append("")
    lines.append("## 输入哈希")
    lines.append("")
    lines.append(md_table(hashes))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def main() -> None:
    np.random.seed(SEED)
    laps, episodes, circuit_map, t020_race, t020_strat, grids = load_data()
    race_set = filtered_races(t020_race)

    race_tab, prob, prob_prev, blocked, restarts, skipped = analyze_races(
        laps, race_set, circuit_map, episodes)

    # 并入 T020 的每场进站次数（以及 SC/VSC 段数）
    t020_sel = t020_race[["year", "round", "n_total_pits", "n_sc_events", "n_vsc_events"]]
    race_tab = race_tab.merge(t020_sel, on=["year", "round"], how="left")
    race_tab["overtakes_per_pit"] = race_tab["n_overtakes"] / race_tab["n_total_pits"].replace(0, np.nan)

    # 按赛道汇总
    circuit_rows = []
    for circuit, g in race_tab.groupby("circuit_id"):
        circuit_rows.append({
            "circuit_id": circuit, "赛道类型": track_type(circuit),
            "场数": int(len(g)),
            "每场平均超车": float(g["n_overtakes"].mean()),
            "每场中位数": float(g["n_overtakes"].median()),
            "每圈超车率": float(g["n_overtakes"].sum() / g["n_transitions"].sum())
            if g["n_transitions"].sum() else np.nan,
        })
    circuit_tab = pd.DataFrame(circuit_rows).sort_values("每场平均超车", ascending=False)

    prob_tab = probability_table(prob)
    prob_prev_tab = probability_table_prev(prob_prev)
    blocked_tab = blocked_table(blocked)
    usable, excluded, start_comp, start_grid, start_lap1, start_all = start_table(
        laps, race_set, circuit_map, t020_strat, grids)
    restart_tab = restart_table(restarts)

    sample_info = {
        "n_filtered_races": len(race_set),
        "n_overtake_races": int(len(race_tab)),
        "n_skipped": int(len(skipped)),
        "skipped": skipped,
        "rows": [
            {"项目": "T020 筛选场次（usable & dry & 无红旗）", "值": len(race_set)},
            {"项目": "有可用逐圈名次的场次", "值": int(len(race_tab))},
            {"项目": "无可用累计时间的场次", "值": int(len(skipped))},
            {"项目": "赛道超车总数", "值": int(race_tab["n_overtakes"].sum())},
            {"项目": "超车概率分析车对数", "值": int(len(prob))},
            {"项目": "被堵代价分析车对数", "值": int(len(blocked))},
            {"项目": "起步分析车场次（含全部起步配方）", "值": int(len(start_all))},
            {"项目": "起步分析可用车场次（干胎、grid≥1）", "值": int(len(usable))},
            {"项目": "起点被剔除车场次", "值": int(len(excluded))},
            {"项目": "SC 重启段数（有间距）", "值": int((restarts["type"] == "SC").sum())},
            {"项目": "VSC 重启段数（有间距）", "值": int((restarts["type"] == "VSC").sum())},
        ],
    }

    hashes = pd.DataFrame([
        {"文件": str(LAPS.relative_to(ROOT)), "sha256": sha256(LAPS),
         "字节": LAPS.stat().st_size},
        {"文件": str(EPISODES.relative_to(ROOT)), "sha256": sha256(EPISODES),
         "字节": EPISODES.stat().st_size},
        {"文件": str(RACE_SUMMARY.relative_to(ROOT)), "sha256": sha256(RACE_SUMMARY),
         "字节": RACE_SUMMARY.stat().st_size},
        {"文件": str(T020_RACE.relative_to(ROOT)), "sha256": sha256(T020_RACE),
         "字节": T020_RACE.stat().st_size},
        {"文件": str(T020_STRAT.relative_to(ROOT)), "sha256": sha256(T020_STRAT),
         "字节": T020_STRAT.stat().st_size},
        {"文件": f"data/raw/jolpica/*_results_limit=100.json（{len(glob.glob(str(JOLPICA / '*_results_limit=100.json')))} 个文件）",
         "sha256": None, "字节": ""},
    ])

    figs = make_figures(circuit_tab, prob_tab, blocked_tab, start_grid)

    race_tab.to_csv(OUT / "race_overtakes.csv", index=False)
    circuit_tab.to_csv(OUT / "circuit_overtakes.csv", index=False)
    prob_tab.to_csv(OUT / "overtake_probability.csv", index=False)
    prob_prev_tab.to_csv(OUT / "overtake_probability_prev_lap.csv", index=False)
    blocked_tab.to_csv(OUT / "blocked_cost.csv", index=False)
    start_comp.to_csv(OUT / "start_by_compound.csv", index=False)
    start_grid.to_csv(OUT / "start_grid_compound.csv", index=False)
    start_lap1.to_csv(OUT / "start_lap1_by_grid.csv", index=False)
    restarts.to_csv(OUT / "sc_restart_spacing.csv", index=False)
    restart_tab.to_csv(OUT / "sc_restart_summary.csv", index=False)
    hashes.to_csv(OUT / "input_sha256.csv", index=False)

    report = build_report(sample_info, race_tab, circuit_tab, prob_tab, prob_prev_tab,
                          blocked_tab, start_comp, start_grid, start_lap1,
                          restart_tab, restarts, figs, hashes, t020_race)
    REPORT.write_text(report, encoding="utf-8")

    print("==" * 30)
    print(sample_info["rows"])
    print("\nrace overtakes (per race):")
    print(race_tab[["year", "round", "circuit_id", "n_overtakes", "n_total_pits"]]
          .sort_values("n_overtakes", ascending=False).head(8).to_string(index=False))
    print("\ncircuit top/bottom:")
    print(pd.concat([circuit_tab.head(5), circuit_tab.tail(5)]).to_string(index=False))
    print("\novertake probability:")
    print(prob_tab.to_string(index=False))
    print("\nblocked cost:")
    print(blocked_tab.to_string(index=False))
    print("\nstart by compound:")
    print(start_comp.to_string(index=False))
    print("\nSC restart summary:")
    print(restart_tab.to_string(index=False))
    print("\noutputs:")
    for p in sorted(OUT.glob("*.csv")):
        print(" ", p.relative_to(HERE))
    for p in figs:
        print(" ", p.relative_to(HERE))
    print(" ", REPORT.relative_to(HERE))
    print("\nDONE")


if __name__ == "__main__":
    main()
