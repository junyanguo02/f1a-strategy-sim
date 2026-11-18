"""T031: 多出来的那一停：被迫还是主动（只做描述）。

一条命令重跑全部结果（在项目根目录执行）：

    .venv/bin/python analysis/extra_stop/run_extra_stop.py

输入（只读，全部在本地，不联网、不安装依赖）：
  analysis/teammate_outcomes/outputs/teammate_pairs.csv   T028 车对明细（主样本/SC 对照）
  analysis/stint_heterogeneity/outputs/stint_slopes.csv   T026 每段斜率
  analysis/actual_strategies/stints.csv                  stint 起止圈（识别 stint/进站圈）
  analysis/actual_strategies/strategies.parquet          车手号码、session_key
  data/processed/laps_pit_filled.parquet                 逐圈时间（累计时间/名次/交通）
  data/processed/model_laps.parquet                      干净圈（前 5 圈均值/上升量）
  data/processed/fcy_race_summary.csv                    赛道映射
  data/raw/openf1/pit_session_key=*.json                 OpenF1 进站时长（stop/lane duration）

输出（全部在 analysis/extra_stop/ 下）：
  outputs/pair_metrics.csv        每对车对：1–4 条指标 + 分组 + 用时差
  outputs/metrics_summary.csv     1–4 条指标的关键数字（主样本 vs SC 对照）
  outputs/group_outcomes.csv      第 5 条分组结果
  outputs/flag_prevalence.csv     各标志/分组占比（主样本 vs SC 对照）
  outputs/input_sha256.csv
  figures/fig1_group_outcomes.png
  figures/fig2_tyre_metrics.png
  figures/fig3_flags_control.png
  report.md

只写 analysis/extra_stop/；不删除数据、不提交 git、不安装依赖。
本脚本没有随机过程；两次运行输出一致。
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="extrastop-mpl-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 阈值（事先固定）
TYRE_SLOPE_DIFF = 0.03       # s/lap：多停方第一段斜率 − 少停方 ≥ 0.03 → “轮胎衰减更快”
TYRE_RISE_DIFF = 0.60        # s：多停方进站前 5 圈相对该段前 5 圈的上升量 − 少停方 ≥ 0.60
STOP_ABNORMAL_MARGIN = 3.0   # s：进站时长 > 本站中位数 + 3.0 → “停车异常”
BLOCK_GAP = 1.0              # s：进站前 3 圈中任一圈距前车 < 1.0 → “被堵”
POST_STOP_WINDOW = 2.0       # s：出站圈前后 2 秒内的车数
TACTICAL_WINDOW = 2          # 圈：对手/队友进站后 0–2 圈内跟进/回应；或对手进站前 0–2 圈 undercut

MAIN_SC = "绿旗下"
CTRL_SC = "SC/VSC 下"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROCESSED = ROOT / "data" / "processed"
T028 = ROOT / "analysis" / "teammate_outcomes" / "outputs" / "teammate_pairs.csv"
T026 = ROOT / "analysis" / "stint_heterogeneity" / "outputs" / "stint_slopes.csv"
STINTS = ROOT / "analysis" / "actual_strategies" / "stints.csv"
STRATEGIES = ROOT / "analysis" / "actual_strategies" / "strategies.parquet"
LAPS = PROCESSED / "laps_pit_filled.parquet"
MODEL = PROCESSED / "model_laps.parquet"
RACE_SUMMARY = PROCESSED / "fcy_race_summary.csv"
PIT_GLOB = ROOT / "data" / "raw" / "openf1" / "pit_session_key=*.json"

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


def display_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(fmt)
    return out


def md_table(df: pd.DataFrame) -> str:
    show = display_df(df)
    cols = list(show.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def mean_stat(x) -> dict:
    v = pd.Series(x).dropna().astype(float)
    if len(v) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "sd": np.nan}
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(v.median()),
            "sd": float(v.std(ddof=1))}


# ---------------------------------------------------------------- 数据准备
def load_data():
    pairs = pd.read_csv(T028)
    pairs = pairs[(~pairs["same_stops"]) & (pairs["stop_diff"] == 1)].copy()
    pairs = pairs[pairs["extra_stop_sc"].isin([MAIN_SC, CTRL_SC])].copy()

    laps = pd.read_parquet(LAPS)
    for c in ["pit_in_filled", "safety_car", "virtual_safety_car"]:
        laps[c] = laps[c].fillna(False).astype(bool)

    # 累计时间、每圈名次、前车间距
    cum, states = {}, {}
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
    # 用 pair 涉及场次构造 states
    for (y, r) in pairs[["year", "round"]].drop_duplicates().itertuples(index=False):
        drivers = {d: v for (yy, rr, d), v in cum.items() if yy == y and rr == r}
        all_laps = set()
        for cc in drivers.values():
            all_laps |= set(cc)
        for k in all_laps:
            order = sorted((drivers[d][k], d) for d in drivers if k in drivers[d])
            states[(int(y), int(r), int(k))] = {d: i + 1 for i, (_, d) in enumerate(order)}

    # T026 每段斜率（第一段）
    slopes = pd.read_csv(T026)
    first_slope = slopes[slopes["stint"] == 1][["year", "round", "driver", "slope", "se"]]

    # stints（进站圈/段号）
    stints = pd.read_csv(STINTS).rename(columns={"stint_number": "stint"})

    # 车手号码
    strat = pd.read_parquet(STRATEGIES)
    num = strat.drop_duplicates(["year", "round", "driver"])[
        ["year", "round", "driver", "driver_number", "session_key"]]

    # OpenF1 进站时长
    sess = strat.drop_duplicates(["year", "round"]).set_index("session_key")[["year", "round"]]
    pit_rows = []
    for f in glob.glob(str(PIT_GLOB)):
        sk = int(Path(f).name.split("=")[1].split(".")[0])
        if sk not in sess.index:
            continue
        y, r = sess.loc[sk]
        for rec in json.load(open(f, encoding="utf-8")):
            pit_rows.append({"year": int(y), "round": int(r),
                             "driver_number": rec.get("driver_number"),
                             "lap": rec.get("lap_number"),
                             "stop_duration": rec.get("stop_duration"),
                             "pit_duration": rec.get("pit_duration"),
                             "lane_duration": rec.get("lane_duration")})
    pits = pd.DataFrame(pit_rows)
    pits["dur"] = pits["lane_duration"].fillna(pits["pit_duration"])
    circ = pd.read_csv(RACE_SUMMARY)[["year", "round", "circuit_id"]].drop_duplicates()
    pits = pits.merge(circ, on=["year", "round"], how="left")
    pits = pits.dropna(subset=["circuit_id"])
    circuit_median = pits.groupby("circuit_id")["dur"].median()

    # 干净圈（上升量）
    ml = pd.read_parquet(MODEL)
    ml["t_corr"] = (ml["lap_time_s"].astype(float)
                    - 0.06 * ml["laps_remaining"].astype(float))
    return pairs, cum, states, first_slope, stints, num, pits, circuit_median, ml


# ---------------------------------------------------------------- 指标
def stint_of_lap(stints: pd.DataFrame, year: int, rnd: int, driver: str, lap: int):
    s = stints[(stints["year"] == year) & (stints["round"] == rnd)
               & (stints["driver"] == driver)]
    s = s[(s["lap_start"] <= lap) & (s["lap_end"] >= lap)]
    return s.iloc[0] if len(s) else None


def rise_metric(ml: pd.DataFrame, year: int, rnd: int, driver: str,
                stint_row) -> float:
    """该段先取前 5 个干净圈均值，再取进站圈及之前最后 5 个干净圈均值，差 = 上升量。"""
    if stint_row is None:
        return np.nan
    start = int(stint_row["lap_start"])
    end = int(stint_row["lap_end"])
    s = ml[(ml["year"] == year) & (ml["round"] == rnd) & (ml["driver"] == driver)
           & (ml["lap"] >= start) & (ml["lap"] <= end)].sort_values("lap")
    if len(s) < 6:
        return np.nan
    first = s.head(5)
    last = s[s["lap"] <= int(stint_row["lap_end"])].tail(5)
    if len(first) < 3 or len(last) < 3:
        return np.nan
    return float(last["t_corr"].mean() - first["t_corr"].mean())


def count_within(window_cum: dict, self_driver: str, self_time: float,
                 window: float) -> tuple[int, int, int]:
    """在给定圈的所有车累计时间里，数 |Δt|≤window 的车数（前/后）。"""
    ahead = behind = 0
    for d, t in window_cum.items():
        if d == self_driver:
            continue
        dt = t - self_time
        if 0 < dt <= window:
            behind += 1
        elif -window <= dt < 0:
            ahead += 1
    return ahead, behind, ahead + behind


def pit_laps(stints: pd.DataFrame, year: int, rnd: int, driver: str) -> list[int]:
    s = stints[(stints["year"] == year) & (stints["round"] == rnd)
               & (stints["driver"] == driver)].sort_values("stint")
    return [int(x) for x in s["lap_end"].iloc[:-1]]


def build_metrics(pairs: pd.DataFrame, cum, states, first_slope, stints, num,
                  pits, circuit_median, ml) -> pd.DataFrame:
    rows = []
    for r in pairs.itertuples(index=False):
        L = int(r.extra_stop_lap) if pd.notna(r.extra_stop_lap) else None
        y, rd = int(r.year), int(r.round)
        more, less = str(r.driver_more), str(r.driver_less)
        row = {"year": y, "round": rd, "team": r.team,
               "driver_more": more, "driver_less": less,
               "extra_stop_lap": L, "extra_stop_sc": r.extra_stop_sc,
               "time_diff": float(r.time_diff), "aligned": bool(r.aligned),
               "grid_diff": float(r.grid_diff), "pace_diff": float(r.pace_diff)}
        if L is None:
            rows.append(row)
            continue
        # 1) T026 第一段斜率差
        sm = first_slope[(first_slope["year"] == y) & (first_slope["round"] == rd)
                         & (first_slope["driver"] == more)]
        sl = first_slope[(first_slope["year"] == y) & (first_slope["round"] == rd)
                         & (first_slope["driver"] == less)]
        row["slope_first_more"] = float(sm["slope"].iloc[0]) if len(sm) else np.nan
        row["slope_first_less"] = float(sl["slope"].iloc[0]) if len(sl) else np.nan
        if len(sm) and len(sl):
            row["slope_diff"] = float(sm["slope"].iloc[0] - sl["slope"].iloc[0])
        else:
            row["slope_diff"] = np.nan
        # 1b) 进站前 5 圈相对该段前 5 圈的上升量
        stm = stint_of_lap(stints, y, rd, more, L)
        stl_row = stints[(stints["year"] == y) & (stints["round"] == rd)
                         & (stints["driver"] == less)]
        stl = None
        if len(stl_row):
            inside = stl_row[(stl_row["lap_start"] <= L) & (stl_row["lap_end"] >= L)]
            stl = inside.iloc[0] if len(inside) else None
        rise_more = rise_metric(ml, y, rd, more, stm)
        rise_less = rise_metric(ml, y, rd, less, stl)
        row["rise_more"] = rise_more
        row["rise_less"] = rise_less
        row["rise_diff"] = (rise_more - rise_less
                            if pd.notna(rise_more) and pd.notna(rise_less) else np.nan)
        # 2) 停车时长异常
        num_m = num[(num["year"] == y) & (num["round"] == rd) & (num["driver"] == more)]
        dnum = int(num_m["driver_number"].iloc[0]) if len(num_m) else None
        pm = pits[(pits["year"] == y) & (pits["round"] == rd)
                  & (pits["driver_number"] == dnum) & (pits["lap"] == L)]
        dur = float(pm["dur"].mean()) if len(pm) else np.nan
        # circuit_id 在 build_metrics 末尾统一 merge
        row["stop_duration"] = dur
        row["stop_duration_median"] = np.nan
        row["stop_abnormal"] = np.nan
        # 3) 交通与位置
        blocked = False
        min_gap = np.nan
        cm = cum.get((y, rd, more), {})
        for lap in range(max(1, L - 3), L):
            if lap not in cm:
                continue
            pos = states.get((y, rd, lap), {}).get(more)
            if pos is None or pos == 1:
                continue
            drivers_lap = [d for (yy, rr, d), cc in cum.items()
                           if yy == y and rr == rd and lap in cc]
            rank = sorted(drivers_lap, key=lambda d: cum[(y, rd, d)][lap])
            i = rank.index(more)
            if i > 0:
                gap = cm[lap] - cum[(y, rd, rank[i - 1])][lap]
                if np.isfinite(gap):
                    min_gap = gap if pd.isna(min_gap) else min(min_gap, gap)
                    if gap < BLOCK_GAP:
                        blocked = True
        row["min_gap_before"] = min_gap
        row["blocked_before"] = bool(blocked)
        out = L + 1
        ahead2 = behind2 = total2 = np.nan
        if out in cm:
            drivers_out = [d for (yy, rr, d), cc in cum.items()
                           if yy == y and rr == rd and out in cc]
            window_cum = {d: cum[(y, rd, d)][out] for d in drivers_out}
            a, b, t = count_within(window_cum, more, cm[out], POST_STOP_WINDOW)
            ahead2, behind2, total2 = a, b, t
        row["post_stop_within2_ahead"] = ahead2
        row["post_stop_within2_behind"] = behind2
        row["post_stop_within2_total"] = total2
        # 4) 战术时机
        rank_prev = None
        st_prev = states.get((y, rd, L - 1), {})
        if more in st_prev:
            rank_prev = st_prev[more]
        drivers_lap = [d for (yy, rr, d), cc in cum.items()
                       if yy == y and rr == rd and (L - 1) in cc]
        rank = sorted(drivers_lap, key=lambda d: cum[(y, rd, d)][L - 1])
        ahead_drv = rank[rank.index(more) - 1] if more in rank and rank.index(more) > 0 else None
        behind_drv = (rank[rank.index(more) + 1]
                      if more in rank and rank.index(more) + 1 < len(rank) else None)
        row["ahead_driver"] = ahead_drv
        row["behind_driver"] = behind_drv
        ahead_pits = pit_laps(stints, y, rd, ahead_drv) if ahead_drv else []
        behind_pits = pit_laps(stints, y, rd, behind_drv) if behind_drv else []
        # teammate 用 pairs 的 less 车手（同队）
        mate_pits = pit_laps(stints, y, rd, less)

        def in_window(laps, lo, hi):
            return any(lo <= x <= hi for x in laps)

        response_ahead = in_window(ahead_pits, L - TACTICAL_WINDOW, L)
        response_behind = in_window(behind_pits, L - TACTICAL_WINDOW, L)
        response_mate = in_window(mate_pits, L - TACTICAL_WINDOW, L + TACTICAL_WINDOW)
        undercut = in_window(ahead_pits, L + 1, L + TACTICAL_WINDOW)
        row["response_ahead"] = bool(response_ahead)
        row["response_behind"] = bool(response_behind)
        row["teammate_within2"] = bool(response_mate)
        row["undercut_ahead"] = bool(undercut)
        row["tactical"] = bool(response_ahead or response_behind or response_mate or undercut)
        rows.append(row)
    df = pd.DataFrame(rows)

    # 停车异常（需要 circuit_id）
    circ = pd.read_csv(RACE_SUMMARY)[["year", "round", "circuit_id"]].drop_duplicates()
    df = df.merge(circ, on=["year", "round"], how="left")
    df["stop_duration_median"] = df["circuit_id"].map(circuit_median)
    df["stop_abnormal"] = df["stop_duration"] > (df["stop_duration_median"] + STOP_ABNORMAL_MARGIN)
    return df


# ---------------------------------------------------------------- 分组
def assign_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tyre = (df["slope_diff"] >= TYRE_SLOPE_DIFF) | (df["rise_diff"] >= TYRE_RISE_DIFF)
    tyre = tyre.fillna(False)
    stop = df["stop_abnormal"].fillna(False).astype(bool)
    traffic = df["blocked_before"].fillna(False).astype(bool) | df["tactical"].fillna(False).astype(bool)

    def grp(row):
        if bool(tyre.loc[row.name]):
            return "轮胎衰减更快（被迫）"
        if bool(stop.loc[row.name]):
            return "停车异常（被迫）"
        if bool(traffic.loc[row.name]):
            return "被堵 / 战术"
        return "无明显原因（主动）"

    df["group"] = df.apply(grp, axis=1)
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, s in df.groupby("group"):
        st = mean_stat(s["time_diff"])
        rows.append({"组": g, "车对数": int(len(s)),
                     "用时差均值 (s)": st["mean"], "中位数 (s)": st["median"],
                     "多停更快占比": float((s["time_diff"] < 0).mean()) if len(s) else np.nan,
                     "进站前被堵占比": float(s["blocked_before"].mean()) if len(s) else np.nan,
                     "轮胎斜率差均值": float(s["slope_diff"].mean()) if len(s) else np.nan,
                     "上升量差均值": float(s["rise_diff"].mean()) if len(s) else np.nan,
                     "停车异常占比": float(s["stop_abnormal"].mean()) if len(s) else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 图
def make_figures(df: pd.DataFrame, summary: pd.DataFrame):
    paths = []
    df = df[df["extra_stop_sc"] == MAIN_SC] if False else df
    groups = ["轮胎衰减更快（被迫）", "停车异常（被迫）", "被堵 / 战术", "无明显原因（主动）"]
    colors = {"轮胎衰减更快（被迫）": "#c44e52", "停车异常（被迫）": "#dd8452",
              "被堵 / 战术": "#4c72b0", "无明显原因（主动）": "#55a868"}
    en = {"轮胎衰减更快（被迫）": "tyre faster (forced)",
          "停车异常（被迫）": "stop abnormal (forced)",
          "被堵 / 战术": "traffic / tactical",
          "无明显原因（主动）": "no obvious cause (active)"}

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    vals, ns, cs = [], [], []
    main_df = df[df["extra_stop_sc"] == MAIN_SC]
    for g in groups:
        s = main_df[main_df["group"] == g]
        if len(s) == 0:
            continue
        vals.append(float(s["time_diff"].mean()))
        ns.append(len(s))
        cs.append(colors[g])
    x = np.arange(len(vals))
    ax.bar(x, vals, color=cs, edgecolor="black", linewidth=0.5)
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.text(i, v + (1 if v >= 0 else -1), f"{v:+.1f}s\n(n={n})",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([en.get(g, g) for g in groups[:len(vals)]], fontsize=7.5)
    ax.set_ylabel("Mean time difference: more - fewer stops (s)")
    ax.set_ylim(0, max(vals) * 1.18 if vals else 1)
    ax.set_title("Green-flag extra stop: outcomes by exclusive group")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = FIG / "fig1_group_outcomes.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for g in groups:
        s = main_df[(main_df["group"] == g) & main_df["slope_diff"].notna()
                    & main_df["rise_diff"].notna()]
        if len(s) == 0:
            continue
        ax.scatter(s["slope_diff"], s["rise_diff"], s=32, color=colors[g],
                   alpha=0.75, edgecolor="black", linewidth=0.3, label=en.get(g, g))
    ax.axvline(TYRE_SLOPE_DIFF, color="#333333", linestyle="--", linewidth=0.8)
    ax.axhline(TYRE_RISE_DIFF, color="#333333", linestyle="--", linewidth=0.8)
    ax.set_xlabel("First-stint slope difference (s/lap; more - fewer)")
    ax.set_ylabel("Rise difference before extra stop (s; more - fewer)")
    ax.set_title("Tyre-state metrics by group")
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = FIG / "fig2_tyre_metrics.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax = axes[0]
    flags = ["轮胎衰减更快", "停车异常", "进站前被堵", "战术（跟进/回应/undercut）"]
    main = [float((df["group"] == "轮胎衰减更快（被迫）").mean()),
            float(df["stop_abnormal"].fillna(False).mean()),
            float(df["blocked_before"].fillna(False).mean()),
            float(df["tactical"].fillna(False).mean())]
    ctrl = df[df["extra_stop_sc"] == CTRL_SC]
    main_df = df[df["extra_stop_sc"] == MAIN_SC]
    ctrl_vals = [
        float((((ctrl["slope_diff"] >= TYRE_SLOPE_DIFF)
                | (ctrl["rise_diff"] >= TYRE_RISE_DIFF))).fillna(False).mean()),
        float(ctrl["stop_abnormal"].fillna(False).mean()),
        float(ctrl["blocked_before"].fillna(False).mean()),
        float(ctrl["tactical"].fillna(False).mean()),
    ]
    x = np.arange(len(flags))
    ax.bar(x - 0.2, main[:4], 0.4, color="#c44e52", edgecolor="black", linewidth=0.4,
           label="green-flag extra stop")
    ax.bar(x + 0.2, ctrl_vals, 0.4, color="#4c72b0", edgecolor="black", linewidth=0.4,
           label="SC/VSC extra stop")
    ax.set_xticks(x)
    ax.set_xticklabels(["tyre faster", "stop abnormal", "blocked before", "tactical"],
                       fontsize=8)
    ax.set_ylabel("Share of pairs")
    ax.set_title("Flag prevalence: main vs control")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    vals_main = summary["用时差均值 (s)"].to_numpy(float)
    labels = summary["组"].tolist()
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals_main, color=[colors.get(g, "#999999") for g in labels],
            edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{en.get(g, g)} (n={int(n)})"
                        for g, n in zip(labels, summary["车对数"])], fontsize=7.5)
    ax.set_xlabel("Mean time difference (s)")
    ax.set_title("Main sample: mean time difference by group")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p = FIG / "fig3_flags_control.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


# ---------------------------------------------------------------- 报告
def build_report(df: pd.DataFrame, summary: pd.DataFrame, figs, hashes) -> str:
    main = df[df["extra_stop_sc"] == MAIN_SC]
    ctrl = df[df["extra_stop_sc"] == CTRL_SC]
    lines = []
    lines.append("# T031　多出来的那一停：被迫还是主动")
    lines.append("")
    lines.append("由 `analysis/extra_stop/run_extra_stop.py` 离线生成；只做描述统计，不评价策略、"
                 "也不写建模建议。阈值在脚本里事先固定（见下）；两次运行输出一致。")
    lines.append("")
    lines.append("## 口径与事先固定的阈值")
    lines.append("")
    lines.append("- 主样本：T028 里“绿旗下多停一次”的车对；对照组：T028 里“SC/VSC 下多停一次”的车对。")
    lines.append("- **轮胎衰减更快**：T026 第一段斜率差（多停 − 少停）≥ "
                 f"**{TYRE_SLOPE_DIFF:.2f} s/圈**；或进站前 5 圈相对该段前 5 圈的上升量差 ≥ "
                 f"**{TYRE_RISE_DIFF:.2f} 秒**。")
    lines.append(f"- **停车异常**：多出那次进站的 lane/pit duration > 该赛道中位数 + "
                 f"**{STOP_ABNORMAL_MARGIN:.1f} 秒**。")
    lines.append(f"- **被堵 / 战术**：进站前 3 圈中任一圈距前车 < **{BLOCK_GAP:.1f} 秒**；"
                 f"或同队/前后名次对手在本次进站前 0–{TACTICAL_WINDOW} 圈内进站（跟进/回应），"
                 f"或前车在本次进站后 0–{TACTICAL_WINDOW} 圈内进站（undercut 尝试）。")
    lines.append("- **分组优先级**：轮胎衰减更快 → 停车异常 → 被堵 / 战术 → 无明显原因（主动）；"
                 "一个车对可能同时满足多个，只按优先级归入一组，前 4 条的标志仍各自统计。")
    lines.append("- 用时差 = T028 的“多停方 − 少停方”累计用时差（正 = 多停更慢）。")
    lines.append("")

    lines.append("## 1. 主样本与对照")
    lines.append("")
    rows = [{"样本": "绿旗下多停一次（主样本）", "车对数": int(len(main)),
             "用时差均值 (s)": float(main["time_diff"].mean()),
             "中位数 (s)": float(main["time_diff"].median())},
            {"样本": "SC/VSC 下多停一次（对照）", "车对数": int(len(ctrl)),
             "用时差均值 (s)": float(ctrl["time_diff"].mean()),
             "中位数 (s)": float(ctrl["time_diff"].median())}]
    lines.append(md_table(pd.DataFrame(rows)))
    lines.append("")

    lines.append("## 2. 轮胎状态（第 1 条）")
    lines.append("")
    for name, s in [("主样本（绿旗下）", main), ("对照（SC/VSC 下）", ctrl)]:
        sd = s["slope_diff"].dropna()
        rd = s["rise_diff"].dropna()
        lines.append(f"**{name}**：第一段斜率差可用 {len(sd)}/{len(s)} 对"
                     f"（均值 {sd.mean():+.3f}、中位数 {sd.median():+.3f} s/圈，"
                     f"正 = 多停方衰减更快）；上升量差可用 {len(rd)}/{len(s)} 对"
                     f"（均值 {rd.mean():+.2f}、中位数 {rd.median():+.2f} 秒）。")
        lines.append("")
    lines.append("![轮胎指标](figures/fig2_tyre_metrics.png)")
    lines.append("")
    lines.append("**图 2**：主样本的第一段斜率差（x）与进站前上升量差（y），颜色为最终分组；"
                 "虚线为事先固定的阈值。")
    lines.append("")

    lines.append("## 3. 车损 / 停车异常（第 2 条）")
    lines.append("")
    rows = []
    for name, s in [("主样本（绿旗下）", main), ("对照（SC/VSC 下）", ctrl)]:
        av = s["stop_duration"].notna().sum()
        ab = int(s["stop_abnormal"].fillna(False).sum())
        rows.append({"样本": name, "有停车时长": int(av), "停车异常": ab,
                     "异常占比": ab / av if av else np.nan,
                     "异常组用时差均值 (s)": float(s[s["stop_abnormal"] == True]["time_diff"].mean())
                     if ab else np.nan})
    lines.append(md_table(pd.DataFrame(rows)))
    lines.append("")

    lines.append("## 4. 交通与位置（第 3 条）")
    lines.append("")
    rows = []
    for name, s in [("主样本（绿旗下）", main), ("对照（SC/VSC 下）", ctrl)]:
        blocked = int(s["blocked_before"].fillna(False).sum())
        rows.append({"样本": name, "有进站前 3 圈数据": int(s["min_gap_before"].notna().sum()),
                     "进站前被堵": blocked,
                     "被堵占比": float(s["blocked_before"].mean()) if len(s) else np.nan,
                     "出站后 ±2 秒车数均值": float(s["post_stop_within2_total"].mean()),
                     "出站后 ±2 秒车数中位数": float(s["post_stop_within2_total"].median())})
    lines.append(md_table(pd.DataFrame(rows)))
    lines.append("")

    lines.append("## 5. 战术时机（第 4 条）")
    lines.append("")
    rows = []
    for name, s in [("主样本（绿旗下）", main), ("对照（SC/VSC 下）", ctrl)]:
        rows.append({"样本": name, "跟进/回应 前车": int(s["response_ahead"].fillna(False).sum()),
                     "跟进/回应 后车": int(s["response_behind"].fillna(False).sum()),
                     "同队 0–2 圈内": int(s["teammate_within2"].fillna(False).sum()),
                     "undercut 前车": int(s["undercut_ahead"].fillna(False).sum()),
                     "任意战术": int(s["tactical"].fillna(False).sum())})
    lines.append(md_table(pd.DataFrame(rows)))
    lines.append("")

    lines.append("## 6. 分组结果（第 5 条）")
    lines.append("")
    lines.append(md_table(summary.rename(columns={
        "组": "分组", "车对数": "车对数", "用时差均值 (s)": "用时差均值 (s)",
        "中位数 (s)": "中位数 (s)", "多停更快占比": "多停更快占比",
        "进站前被堵占比": "进站前被堵占比", "轮胎斜率差均值": "轮胎斜率差均值",
        "上升量差均值": "上升量差均值", "停车异常占比": "停车异常占比"})))
    lines.append("")
    lines.append("![分组结果](figures/fig1_group_outcomes.png)")
    lines.append("")
    lines.append("**图 1**：主样本各分组（按优先级唯一分组）的平均用时差与车对数。")
    lines.append("")
    lines.append("![标志与对照](figures/fig3_flags_control.png)")
    lines.append("")
    lines.append("**图 3**：左 = 主样本 vs SC 对照的各标志占比；右 = 主样本各分组的平均用时差。")
    lines.append("")

    forced = summary[summary["组"].str.contains("被迫")]
    active = summary[summary["组"].str.contains("主动")]
    traffic = summary[summary["组"].str.contains("被堵")]
    n_forced = int(forced["车对数"].sum())
    n_active = int(active["车对数"].sum()) if len(active) else 0
    n_traffic = int(traffic["车对数"].sum()) if len(traffic) else 0
    lines.append("## 观察到的现象")
    lines.append("")
    lines.append(f"1. **被迫组只占约三成：** 主样本 {len(main)} 对里，轮胎衰减更快 "
                 f"{int(summary.loc[summary['组']=='轮胎衰减更快（被迫）','车对数'].sum()) if (summary['组']=='轮胎衰减更快（被迫）').any() else 0} 对、"
                 f"停车异常 {int(summary.loc[summary['组']=='停车异常（被迫）','车对数'].sum()) if (summary['组']=='停车异常（被迫）').any() else 0} 对，"
                 f"合计被迫 **{n_forced} 对（{n_forced/len(main)*100:.0f}%）**；被堵/战术 **{n_traffic} 对**；"
                 f"无明显原因（主动）**{n_active} 对**。")
    lines.append(f"2. **各组的用时差都为正（多停更慢）；停车异常组最慢，主动组也不快。** "
                 + "；".join(f"{r['组']}：n={int(r['车对数'])}、均值 {r['用时差均值 (s)']:+.1f} 秒"
                            for _, r in summary.iterrows())
                 + "。")
    lines.append(f"3. **对照（SC/VSC 下多停）整体更快**：均值 {float(ctrl['time_diff'].mean()):+.1f} 秒"
                 f"（n={len(ctrl)}）；主样本均值 {float(main['time_diff'].mean()):+.1f} 秒。"
                 "这与 T028 一致：SC 下多出的那一次进站更像“免费进站”，与绿旗下多停的性质不同。")
    lines.append(f"4. **轮胎/交通指标并不完全重合：** 第一段斜率差为正（多停方衰减更快）的比例 "
                 f"{float((main['slope_diff'] > 0).mean())*100:.0f}%（{int(main['slope_diff'].notna().sum())} 对可用）；"
                 f"进站前被堵占比 {float(main['blocked_before'].mean())*100:.0f}%；"
                 f"任意战术信号占比 {float(main['tactical'].mean())*100:.0f}%。")
    lines.append("")

    lines.append("## 这些数据不能说明什么")
    lines.append("")
    lines.append("1. **T026 第一段斜率覆盖率有限**：只有第一段 ≥10 个干净圈的 stint 才有斜率，"
                 f"主样本 82 对里只有 {int(main['slope_diff'].notna().sum())} 对；其余靠上升量或其它标志判断，"
                 "轮胎状态的分类并不完整。")
    lines.append("2. **“被迫”是代理，不是直接观测。** 轮胎衰减差、停车异常、被堵、战术都只是代理；"
                 "车损、前翼更换、引擎模式、队内指令等无法从这些数据里直接看到。")
    lines.append("3. **阈值是事先固定的经验值，不是估计出来的最优切点。** 换阈值会改变分组人数（例如斜率差 0.03→0.05），"
                 "所以分组结果应看作“按预先定义的规则做的描述”，不是唯一解。")
    lines.append("4. **停车时长用 lane/pit duration 代理车损。** OpenF1 `stop_duration` 半数缺失，"
                 "这里优先用 `lane_duration`、否则 `pit_duration`；它包含进站通道时间，长停车不一定等于车损。")
    lines.append("5. **交通与位置用累计时间/名次重算。** 与官方名次有已知差异（T023）；"
                 "“出站后 ±2 秒车数”只算同圈完成的车，被套圈车/进出站中的车可能被漏掉。")
    lines.append("6. **战术窗口是粗窗口。** 只检查同队和前后各一名车手在 0–2 圈内的进站；"
                 "更远的车、车队整体计划、轮胎状态变化都没有纳入。")
    lines.append("7. **样本小、选择偏差仍在。** 主样本只有 82 对，分组后每组十几到几十对；"
                 "“被迫 vs 主动”组间比较仍是观察性的，不能当因果。")
    lines.append("8. **本单只做描述，不能直接当模拟器参数**，也不回答“车队该不该少停”（不给建模建议）。")
    lines.append("")
    lines.append("## 输入哈希")
    lines.append("")
    lines.append(md_table(hashes))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def main() -> None:
    np.random.seed(20261118)
    pairs, cum, states, first_slope, stints, num, pits, circuit_median, ml = load_data()

    # 把 circuit_id 加进 pairs（用于赛道中位数）
    circ = pd.read_csv(RACE_SUMMARY)[["year", "round", "circuit_id"]].drop_duplicates()
    pairs = pairs.merge(circ, on=["year", "round"], how="left")

    df = build_metrics(pairs, cum, states, first_slope, stints, num,
                       pits, circuit_median, ml)
    df = assign_groups(df)

    main = df[df["extra_stop_sc"] == MAIN_SC]
    summary = summarise(main).sort_values("车对数", ascending=False).reset_index(drop=True)

    # 指标汇总表
    metric_rows = []
    for name, s in [("主样本（绿旗下）", main),
                    ("对照（SC/VSC 下）", df[df["extra_stop_sc"] == CTRL_SC])]:
        metric_rows.append({
            "样本": name, "车对数": int(len(s)),
            "斜率差可用": int(s["slope_diff"].notna().sum()),
            "斜率差均值": float(s["slope_diff"].mean()),
            "上升量差可用": int(s["rise_diff"].notna().sum()),
            "上升量差均值": float(s["rise_diff"].mean()),
            "停车时长可用": int(s["stop_duration"].notna().sum()),
            "停车异常数": int(s["stop_abnormal"].fillna(False).sum()),
            "被堵数": int(s["blocked_before"].fillna(False).sum()),
            "战术数": int(s["tactical"].fillna(False).sum()),
            "出站±2秒车数均值": float(s["post_stop_within2_total"].mean()),
        })
    metrics = pd.DataFrame(metric_rows)

    # 各标志占比（主 vs 对照）
    flag_rows = []
    for name, s in [("主样本（绿旗下）", main),
                    ("对照（SC/VSC 下）", df[df["extra_stop_sc"] == CTRL_SC])]:
        flag_rows.append({
            "样本": name, "车对数": int(len(s)),
            "轮胎更快占比": float(((s["slope_diff"] >= TYRE_SLOPE_DIFF)
                                 | (s["rise_diff"] >= TYRE_RISE_DIFF)).fillna(False).mean()),
            "停车异常占比": float(s["stop_abnormal"].fillna(False).mean()),
            "进站前被堵占比": float(s["blocked_before"].fillna(False).mean()),
            "战术占比": float(s["tactical"].fillna(False).mean()),
            "无明显原因占比": float((s["group"] == "无明显原因（主动）").mean()),
        })
    flags = pd.DataFrame(flag_rows)

    hashes = pd.DataFrame([
        {"文件": str(T028.relative_to(ROOT)), "sha256": sha256(T028),
         "字节": T028.stat().st_size},
        {"文件": str(T026.relative_to(ROOT)), "sha256": sha256(T026),
         "字节": T026.stat().st_size},
        {"文件": str(STINTS.relative_to(ROOT)), "sha256": sha256(STINTS),
         "字节": STINTS.stat().st_size},
        {"文件": str(STRATEGIES.relative_to(ROOT)), "sha256": sha256(STRATEGIES),
         "字节": STRATEGIES.stat().st_size},
        {"文件": str(LAPS.relative_to(ROOT)), "sha256": sha256(LAPS),
         "字节": LAPS.stat().st_size},
        {"文件": str(MODEL.relative_to(ROOT)), "sha256": sha256(MODEL),
         "字节": MODEL.stat().st_size},
        {"文件": f"data/raw/openf1/pit_session_key=*.json（{len(glob.glob(str(PIT_GLOB)))} 个文件）",
         "sha256": None, "字节": ""},
    ])

    figs = make_figures(df, summary)

    df.to_csv(OUT / "pair_metrics.csv", index=False)
    metrics.to_csv(OUT / "metrics_summary.csv", index=False)
    summary.to_csv(OUT / "group_outcomes.csv", index=False)
    flags.to_csv(OUT / "flag_prevalence.csv", index=False)
    hashes.to_csv(OUT / "input_sha256.csv", index=False)

    report = build_report(df, summary, figs, hashes)
    REPORT.write_text(report, encoding="utf-8")

    print("==" * 30)
    print("main", len(main), "control", len(df[df["extra_stop_sc"] == CTRL_SC]))
    print(metrics.to_string(index=False))
    print("\nflag prevalence:")
    print(flags.to_string(index=False))
    print("\ngroups:")
    print(summary.to_string(index=False))
    print("\noutputs:")
    for p in sorted(OUT.glob("*.csv")):
        print(" ", p.relative_to(HERE))
    for p in figs:
        print(" ", p.relative_to(HERE))
    print(" ", REPORT.relative_to(HERE))
    print("\nDONE")


if __name__ == "__main__":
    main()
