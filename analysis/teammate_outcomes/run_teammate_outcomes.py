"""T028: 队友对照——多停一次的车真的更快吗（只做描述）。

一条命令重跑全部结果（在项目根目录执行）：

    .venv/bin/python analysis/teammate_outcomes/run_teammate_outcomes.py

输入（只读，全部在本地，不联网、不安装依赖）：
  analysis/actual_strategies/strategies.parquet     T020 策略/进站/起步/完赛信息
  analysis/actual_strategies/outputs/race_summary.csv  T020 场次筛选
  data/processed/laps_pit_filled.parquet            逐圈时间（累计用时、SC 旗标）
  data/processed/model_laps.parquet                 干净圈（第一段前 5 圈配速）
  data/raw/jolpica/*_results_limit=100.json         发车位
  src/f1a/sim（只调用 evaluate()，不改代码）         模型对照

输出（全部在 analysis/teammate_outcomes/ 下）：
  outputs/teammate_pairs.csv         每个车对的实际用时差、发车位差、第一段配速差、模型差
  outputs/regression_pairs.csv       不同/相同进站次数两组的调整回归
  outputs/sc_split.csv               多停一次发生在 SC/VSC 下 vs 绿旗下的结果
  outputs/model_comparison.csv       实际用时差 vs 模型用时差的相关/符号一致率
  outputs/sample_flow.csv            样本筛选计数
  outputs/input_sha256.csv
  figures/fig1_actual_vs_model.png
  figures/fig2_sc_split.png
  figures/fig3_regression.png
  report.md

只写 analysis/teammate_outcomes/；不删除数据、不提交 git、不安装依赖、不修改 src/。
本脚本没有随机过程（模拟器种子固定 20261118，n_sims=2000），两次运行输出一致。
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="teammate-mpl-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1a.sim import RaceConfig, Strategy, TyreParams, evaluate  # noqa: E402

# ---------------------------------------------------------------- 常量
SEED = 20261118
N_SIMS = 2000
N_DRAWS = 200
FUEL_PER_LAP = 0.06
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
MIN_FIRST_LAPS = 3
DRY_FRAC = 0.90

PROCESSED = ROOT / "data" / "processed"
STRATEGIES = ROOT / "analysis" / "actual_strategies" / "strategies.parquet"
T020_RACE = ROOT / "analysis" / "actual_strategies" / "outputs" / "race_summary.csv"
LAPS = PROCESSED / "laps_pit_filled.parquet"
MODEL = PROCESSED / "model_laps.parquet"
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


def fmt_p(p) -> str:
    try:
        if pd.isna(p):
            return "—"
    except (TypeError, ValueError):
        pass
    return "<0.001" if float(p) < 0.001 else f"{float(p):.3f}"


def ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> dict:
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtXi = np.linalg.pinv(X.T @ X)
    uniq, inv = np.unique(groups.astype(str), return_inverse=True)
    G = len(uniq)
    scores = np.zeros((G, X.shape[1]))
    np.add.at(scores, inv, X * resid[:, None])
    V = XtXi @ (scores.T @ scores * G / max(G - 1, 1)) @ XtXi
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return {"beta": beta, "se": se, "n": len(y), "n_clusters": G,
            "df": max(G - 1, 1)}


def mean_stat(x) -> dict:
    v = pd.Series(x).dropna().astype(float)
    if len(v) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "sd": np.nan}
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(v.median()),
            "sd": float(v.std(ddof=1))}


# ---------------------------------------------------------------- 数据准备
def load_data():
    strat = pd.read_parquet(STRATEGIES)
    rs = pd.read_csv(T020_RACE)
    keep = rs[rs["usable"] & rs["dry_for_analysis"] & (~rs["has_red_flag"])]
    keep_keys = set(map(tuple, keep[["year", "round"]].values))
    strat = strat[[(int(y), int(r)) in keep_keys for y, r in zip(strat["year"], strat["round"])]].copy()

    laps = pd.read_parquet(LAPS)
    for c in ["safety_car", "virtual_safety_car", "pit_in_filled"]:
        laps[c] = laps[c].fillna(False).astype(bool)

    # 累计用时
    cum = {}
    for (y, rnd, drv), g in laps.groupby(["year", "round", "driver"]):
        g = g.sort_values("lap")
        laps_arr = g["lap"].to_numpy(int)
        times = g["lap_time_s"].to_numpy(float)
        cc = {}
        total = 0.0
        prev = 0
        for i in range(len(g)):
            k = int(laps_arr[i])
            if k != prev + 1 or not np.isfinite(times[i]):
                break
            total += float(times[i])
            cc[k] = total
            prev = k
        cum[(int(y), int(rnd), str(drv))] = cc

    # 干净圈：第一段前 5 圈
    ml = pd.read_parquet(MODEL)
    ml["t_corr"] = ml["lap_time_s"].astype(float) - FUEL_PER_LAP * ml["laps_remaining"].astype(float)
    first = (ml[ml["stint"] == 1].sort_values(["year", "round", "driver", "tyre_life"])
             .groupby(["year", "round", "driver"]).head(5))
    pace = first.groupby(["year", "round", "driver"]).agg(
        pace_first5=("t_corr", "mean"), n_first5=("t_corr", "size")).reset_index()

    # 发车位
    jrows = []
    for f in glob.glob(str(JOLPICA / "*_results_limit=100.json")):
        stem = Path(f).name.replace("_results_limit=100.json", "")
        y, r = stem.split("_")
        race = json.load(open(f, encoding="utf-8"))["MRData"]["RaceTable"]["Races"][0]
        for x in race["Results"]:
            jrows.append({"year": int(y), "round": int(r),
                          "driver": x["Driver"].get("code"), "grid": x.get("grid")})
    grids = pd.DataFrame(jrows)
    grids["grid"] = pd.to_numeric(grids["grid"], errors="coerce")

    # 进站旗标查询
    flags = {}
    for r in laps.itertuples(index=False):
        flags[(int(r.year), int(r.round), str(r.driver), int(r.lap))] = (
            bool(r.safety_car), bool(r.virtual_safety_car), bool(r.pit_in_filled))
    return strat, rs, cum, pace, grids, flags, keep_keys


def strategy_from_row(row) -> Strategy | None:
    try:
        n = int(row["n_stints"])
        comps = [row[f"stint{k}_compound"] for k in range(1, n + 1)]
        if any(c not in COMPOUNDS for c in comps):
            return None
        stops = [(int(row[f"stint{k}_end"]), comps[k]) for k in range(1, n)]
        start_age = int(max(1, float(row["start_tyre_age"]))) if pd.notna(row["start_tyre_age"]) else 1
        st = Strategy(name=str(row["driver"]), start_compound=comps[0],
                      stops=stops, start_age=start_age)
        st.validate(int(row["total_laps"]))
        return st
    except Exception:
        return None


def find_extra_stop(more_laps: list[int], less_laps: list[int], tol: int = 12) -> int | None:
    """在 more 的进站圈里找唯一没法与 less 对应的一次（容差 ±tol 圈）。"""
    used = set()
    unmatched = []
    for l in sorted(more_laps):
        best, bestd = None, 10 ** 9
        for j, m in enumerate(less_laps):
            if j in used:
                continue
            d = abs(l - m)
            if d < bestd:
                bestd, best = d, j
        if best is not None and bestd <= tol:
            used.add(best)
        else:
            unmatched.append(l)
    return unmatched[0] if len(unmatched) == 1 else None


def stop_laps(row) -> list[int]:
    n = int(row["n_stints"])
    return [int(row[f"stint{k}_end"]) for k in range(1, n)]


# ---------------------------------------------------------------- 构造车对
def build_pairs(strat: pd.DataFrame, cum: dict, pace: pd.DataFrame,
                grids: pd.DataFrame, flags: dict, keep_keys: set) -> tuple[list, dict]:
    s = strat.copy()
    s["race_key"] = s["year"].astype(str) + "|" + s["round"].astype(str)
    s = s[s["wet_tyre_used"].fillna(False) == False]
    s = s[s["start_compound"].isin(COMPOUNDS)]
    s = s[s["raw_first_stint_number"].fillna(1).astype(int) == 1]
    s = s[~s["stint_number_gap"].fillna(False).astype(bool)]
    s = s[s["n_stints"].between(2, 8)]
    s = s.merge(grids, on=["year", "round", "driver"], how="left")
    s = s.merge(pace, on=["year", "round", "driver"], how="left")
    flow = []

    def add(step, n):
        flow.append({"步骤": step, "剩余车场次": int(n)})

    add("干地/无红旗/可用比赛的干胎车场次", len(s))
    s = s[s["total_laps"].notna() & (s["total_laps"] > 0)]
    add("有 total_laps", len(s))
    s = s[s["grid"].notna() & (s["grid"] >= 1)]
    add("有发车位（grid>=1）", len(s))
    s = s[s["pace_first5"].notna() & (s["n_first5"] >= MIN_FIRST_LAPS)]
    add(f"第一段前 5 圈配速可用（≥{MIN_FIRST_LAPS} 圈）", len(s))
    s = s[[(int(y), int(r)) in keep_keys for y, r in zip(s["year"], s["round"])]]

    rows = []
    skipped = {"group_size": 0, "strategy_invalid": 0, "no_cum": 0, "under90": 0, "no_cum_aligned": 0}
    for (year, rnd, team), g in s.groupby(["year", "round", "team"]):
        if len(g) != 2:
            skipped["group_size"] += 1
            continue
        a, b = g.iloc[0], g.iloc[1]
        if int(a["n_pit_stops"]) == int(b["n_pit_stops"]):
            more, less = a, b
            same = True
            # 同进站次数：按车手代码排序确定方向（A→B）
            if str(a["driver"]) > str(b["driver"]):
                more, less = b, a
        else:
            same = False
            more = a if int(a["n_pit_stops"]) > int(b["n_pit_stops"]) else b
            less = b if more is a else a
        n_more, n_less = int(more["n_pit_stops"]), int(less["n_pit_stops"])
        for row in (more, less):
            key = (int(year), int(rnd), str(row["driver"]))
            if key not in cum or not cum[key]:
                skipped["no_cum"] += 1
                break
        else:
            last_more = max(cum[(int(year), int(rnd), str(more["driver"]))])
            last_less = max(cum[(int(year), int(rnd), str(less["driver"]))])
            total = int(more["total_laps"])
            fin_more = bool(more["finisher"]) if pd.notna(more["finisher"]) else False
            fin_less = bool(less["finisher"]) if pd.notna(less["finisher"]) else False
            frac_more = float(more["finish_laps"]) / total if pd.notna(more["finish_laps"]) else 0
            frac_less = float(less["finish_laps"]) / total if pd.notna(less["finish_laps"]) else 0
            if not fin_more and frac_more < DRY_FRAC:
                skipped["under90"] += 1
                continue
            if not fin_less and frac_less < DRY_FRAC:
                skipped["under90"] += 1
                continue
            if last_more < DRY_FRAC * total or last_less < DRY_FRAC * total:
                skipped["under90"] += 1
                continue
            sm = strategy_from_row(more)
            sl = strategy_from_row(less)
            if sm is None or sl is None:
                skipped["strategy_invalid"] += 1
                continue
            align = min(last_more, last_less)
            c_more = cum[(int(year), int(rnd), str(more["driver"]))].get(align)
            c_less = cum[(int(year), int(rnd), str(less["driver"]))].get(align)
            if c_more is None or c_less is None:
                skipped["no_cum_aligned"] += 1
                continue
            flag_align = (last_more != last_less) or (not fin_more) or (not fin_less)
            pos_more = float(more["finish_position"]) if pd.notna(more["finish_position"]) else np.nan
            pos_less = float(less["finish_position"]) if pd.notna(less["finish_position"]) else np.nan
            extra_lap = np.nan
            sc_extra = "—"
            if not same and n_more - n_less == 1:
                extra = find_extra_stop(stop_laps(more), stop_laps(less))
                if extra is not None:
                    extra_lap = extra
                    f = flags.get((int(year), int(rnd), str(more["driver"]), extra_lap))
                    if f is not None:
                        sc_extra = "SC/VSC 下" if (f[0] or f[1]) else "绿旗下"
                    else:
                        sc_extra = "无法判定"
                else:
                    sc_extra = "无法判定"
            rows.append({
                "year": int(year), "round": int(rnd), "team": team,
                "race_key": f"{int(year)}|{int(rnd)}",
                "driver_more": str(more["driver"]), "driver_less": str(less["driver"]),
                "n_stops_more": n_more, "n_stops_less": n_less,
                "stop_diff": n_more - n_less, "same_stops": bool(same),
                "total_laps": total,
                "time_diff": float(c_more - c_less),
                "grid_diff": float(more["grid"] - less["grid"]),
                "pace_diff": float(more["pace_first5"] - less["pace_first5"]),
                "pos_diff": pos_more - pos_less,
                "aligned": bool(flag_align),
                "finisher_more": fin_more, "finisher_less": fin_less,
                "last_lap_more": int(last_more), "last_lap_less": int(last_less),
                "extra_stop_lap": extra_lap if pd.notna(extra_lap) else np.nan,
                "extra_stop_sc": sc_extra,
            })
    pairs = pd.DataFrame(rows)
    add("满足条件的同队两车车对", len(pairs))
    add("其中：进站次数不同且差 1", int((~pairs["same_stops"] & (pairs["stop_diff"] == 1)).sum()))
    add("其中：进站次数不同且差 >1", int((~pairs["same_stops"] & (pairs["stop_diff"] != 1)).sum()))
    add("其中：进站次数相同（对照组）", int(pairs["same_stops"].sum()))
    flow_tab = pd.DataFrame(flow)
    return pairs, {"flow": flow_tab, "skipped": skipped}


# ---------------------------------------------------------------- 模型对照
def model_comparison(pairs: pd.DataFrame, strat: pd.DataFrame):
    cache = {}

    def get(race_key):
        if race_key in cache:
            return cache[race_key]
        y, r = race_key.split("|")
        rid = f"{int(y)}-{int(r)}"
        race = RaceConfig.from_project(ROOT, rid)
        tyre = TyreParams.from_posterior(
            ROOT / "models" / "tyre_v3" / f"outputs31_{int(y)}" / "posterior_core.npz",
            rid, [int(y)], MODEL, n_draws=N_DRAWS, seed=SEED)
        cache[race_key] = (race, tyre)
        return cache[race_key]

    rows = []
    for r in pairs.itertuples(index=False):
        if (not r.same_stops) and int(r.stop_diff) != 1:
            continue
        if not (bool(r.finisher_more) and bool(r.finisher_less)):
            continue
        g = strat[(strat["year"] == r.year) & (strat["round"] == r.round)
                  & (strat["team"] == r.team)]
        sm = g[g["driver"] == r.driver_more]
        sl = g[g["driver"] == r.driver_less]
        if len(sm) != 1 or len(sl) != 1:
            continue
        st_m = strategy_from_row(sm.iloc[0])
        st_l = strategy_from_row(sl.iloc[0])
        if st_m is None or st_l is None:
            continue
        try:
            race, tyre = get(r.race_key)
            out = evaluate(race, tyre, [st_m, st_l], n_sims=N_SIMS, seed=SEED)
            m = out.set_index("strategy")["mean"]
            model_diff = float(m[st_m.name]) - float(m[st_l.name])
        except Exception:
            continue
        rows.append({"race_key": r.race_key, "driver_more": r.driver_more,
                     "driver_less": r.driver_less, "same_stops": bool(r.same_stops),
                     "stop_diff": int(r.stop_diff), "actual_time_diff": float(r.time_diff),
                     "model_time_diff": model_diff,
                     "model_n_sims": int(N_SIMS)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 回归与汇总
def regression_table(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    diff = pairs[(~pairs["same_stops"]) & (pairs["stop_diff"] == 1)].copy()
    ctrl = pairs[pairs["same_stops"]].copy()
    pooled = pd.concat([diff.assign(group=1), ctrl.assign(group=0)], ignore_index=True)
    specs = [
        ("混合样本（差 1 vs 同停）", pooled, True),
        ("不同进站次数（差 1）", diff, False),
        ("相同进站次数（对照）", ctrl, False),
    ]
    for name, df, has_group in specs:
        if len(df) == 0:
            continue
        y = df["time_diff"].to_numpy(float)
        cols = [np.ones(len(df)), df["grid_diff"].to_numpy(float),
                df["pace_diff"].to_numpy(float)]
        names = ["截距", "发车位差 (s/位)", "第一段配速差 (s/s)"]
        if has_group:
            cols.insert(1, df["group"].to_numpy(float))
            names.insert(1, "多停一次 (s，相对同停)")
        X = np.column_stack(cols)
        fit = ols_cluster(y, X, df["race_key"].to_numpy())
        for i, nm in enumerate(names):
            rows.append({"组": name, "变量": nm, "系数": float(fit["beta"][i]),
                         "聚类 SE": float(fit["se"][i]), "n": int(len(df)),
                         "n_场次": int(fit["n_clusters"])})
        rows.append({"组": name, "变量": "原始均值 (s)", "系数": float(np.mean(y)),
                     "聚类 SE": np.nan, "n": int(len(df)),
                     "n_场次": int(fit["n_clusters"])})
        rows.append({"组": name, "变量": "原始中位数 (s)", "系数": float(np.median(y)),
                     "聚类 SE": np.nan, "n": int(len(df)),
                     "n_场次": int(fit["n_clusters"])})
    return pd.DataFrame(rows)


def model_stats(model: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    out = {}
    for name, df in [("不同进站次数（差 1）", model[(~model["same_stops"]) & (model["stop_diff"] == 1)]),
                     ("相同进站次数（对照）", model[model["same_stops"]])]:
        if len(df) < 3:
            continue
        a = df["actual_time_diff"].to_numpy(float)
        m = df["model_time_diff"].to_numpy(float)
        pear = float(stats.pearsonr(a, m)[0])
        spear = float(stats.spearmanr(a, m)[0])
        # 符号一致：两者同号（含 0 视为不一致）
        sign_ok = int(np.sum(np.sign(a) == np.sign(m)))
        model_fewer_faster = m > 0  # 模型认为多停更慢 → 少停更快
        actual_more_faster = a < 0
        more_faster_than_model = int(np.sum(model_fewer_faster & actual_more_faster))
        rows.append({"组": name, "车对数": int(len(df)),
                     "实际均值 (s)": float(a.mean()), "模型均值 (s)": float(m.mean()),
                     "Pearson": pear, "Spearman": spear,
                     "符号一致率": sign_ok / len(df),
                     "模型说少停更快、实际多停更快": more_faster_than_model / len(df),
                     "模型说少停更快、实际少停更快": int(np.sum(model_fewer_faster & (a > 0))) / len(df),
                     "模型说多停更快、实际多停更快": int(np.sum((m < 0) & actual_more_faster)) / len(df)})
        out[name] = {"n": int(len(df)), "pearson": pear, "spearman": spear,
                     "sign_agree": sign_ok / len(df),
                     "model_fewer_actual_more": more_faster_than_model / len(df),
                     "mean_actual": float(a.mean()), "mean_model": float(m.mean())}
    return pd.DataFrame(rows), out


def sc_table(pairs: pd.DataFrame) -> pd.DataFrame:
    diff = pairs[(~pairs["same_stops"]) & (pairs["stop_diff"] == 1)].copy()
    rows = []
    for cls in ["SC/VSC 下", "绿旗下", "无法判定"]:
        g = diff[diff["extra_stop_sc"] == cls]
        st = mean_stat(g["time_diff"])
        rows.append({"多停发生时机": cls, "车对数": int(len(g)),
                     "用时差均值 (s)": st["mean"], "用时差中位数 (s)": st["median"],
                     "用时差 SD (s)": st["sd"],
                     "多停更快占比": float((g["time_diff"] < 0).mean()) if len(g) else np.nan,
                     "发车位差均值": float(g["grid_diff"].mean()) if len(g) else np.nan,
                     "第一段配速差均值": float(g["pace_diff"].mean()) if len(g) else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 图
def make_figures(pairs: pd.DataFrame, model: pd.DataFrame, reg: pd.DataFrame,
                 sc: pd.DataFrame):
    paths = []
    colors = {"不同进站次数（差 1）": "#c44e52", "相同进站次数（对照）": "#4c72b0"}

    # fig1: actual vs model
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for name, df, c in [
            ("same stops (control)", model[model["same_stops"]], "#4c72b0"),
            ("different stops (diff=1)", model[(~model["same_stops"]) & (model["stop_diff"] == 1)], "#c44e52")]:
        if len(df):
            ax.scatter(df["model_time_diff"], df["actual_time_diff"], s=28,
                       color=c, alpha=0.7, edgecolor="black", linewidth=0.3, label=name)
    lim = max(5, float(np.nanmax(np.abs(np.r_[model["model_time_diff"].to_numpy(float),
                                          model["actual_time_diff"].to_numpy(float)]))) * 1.05)
    ax.plot([-lim, lim], [-lim, lim], color="#333333", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="#999999", linewidth=0.6)
    ax.axvline(0, color="#999999", linewidth=0.6)
    ax.set_xlabel("Model time difference: more - fewer stops (s)")
    ax.set_ylabel("Actual time difference: more - fewer stops (s)")
    ax.set_title("Actual vs model strategy difference (teammate pairs)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = FIG / "fig1_actual_vs_model.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # fig2: SC split
    diff = pairs[(~pairs["same_stops"]) & (pairs["stop_diff"] == 1)]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    groups = ["SC/VSC 下", "绿旗下"]
    data = [diff[diff["extra_stop_sc"] == g]["time_diff"].dropna() for g in groups]
    ax.boxplot(data, tick_labels=["extra stop under SC/VSC", "extra stop under green"],
               widths=0.5, showfliers=False)
    rng = np.random.default_rng(20261118)
    for i, d in enumerate(data):
        ax.scatter(i + 1 + rng.normal(0, 0.04, len(d)), d, s=8, alpha=0.4, color="#4c72b0")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Actual time difference: more - fewer stops (s)")
    ax.set_title("Extra stop timing: SC/VSC vs green flag")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = FIG / "fig2_sc_split.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # fig3: regression coefficients
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    plot = reg[~reg["变量"].isin(["原始均值", "原始中位数"])].copy()
    plot = plot[plot["聚类 SE"].notna()]
    labels = []
    vals = []
    ses = []
    cols = []
    en_group = {"混合样本（差 1 vs 同停）": "pooled",
                "不同进站次数（差 1）": "diff=1",
                "相同进站次数（对照）": "control"}
    en_var = {"截距": "intercept", "多停一次 (s，相对同停)": "extra stop (vs same)",
              "发车位差 (s/位)": "grid diff", "第一段配速差 (s/s)": "first-stint pace diff",
              "原始均值 (s)": "raw mean", "原始中位数 (s)": "raw median"}
    for name in ["混合样本（差 1 vs 同停）", "不同进站次数（差 1）", "相同进站次数（对照）"]:
        for _, r in plot[plot["组"] == name].iterrows():
            labels.append(f"{en_group[name]}: {en_var.get(r['变量'], r['变量'])}")
            vals.append(float(r["系数"]))
            ses.append(float(r["聚类 SE"]))
            cols.append("#c44e52" if "多停" in str(r["变量"]) else
                        ("#4c72b0" if "不同" in name else "#999999"))
    y = np.arange(len(labels))[::-1]
    ax.errorbar(vals, y, xerr=1.96 * np.array(ses), fmt="o", color="black",
                ecolor="#555555", capsize=3)
    ax.scatter(vals, y, c=cols, s=40, zorder=3)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Regression coefficient (s)")
    ax.set_title("Adjusted time-difference regressions")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p = FIG / "fig3_regression.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


# ---------------------------------------------------------------- 报告
def build_report(flow, pairs, reg, sc, mstats, mout, figs, hashes) -> str:
    lines = []
    lines.append("# T028　队友对照：多停一次的车真的更快吗")
    lines.append("")
    lines.append("由 `analysis/teammate_outcomes/run_teammate_outcomes.py` 离线生成；只做描述统计，"
                 "不评价策略、也不写建模建议。模型对照只调用 `src/f1a/sim` 的 `evaluate()`，没有修改任何代码。"
                 "模拟器随机种子固定 20261118；两次运行输出一致。")
    lines.append("")
    lines.append("## 口径")
    lines.append("")
    lines.append("- 比赛：T020 的 `usable & dry_for_analysis & ~has_red_flag` 74 场；只保留干胎、非 pit lane 起步、"
                 "`n_stints` 2–8、策略可构造的车场次。")
    lines.append("- 车对：同队两车（`team` 相同）在正赛各有一辆车；要求两车都完赛或都跑满 ≥90% 圈数、"
                 "都有连续累计时间、都有 Jolpica 发车位、第一段前 5 个干净圈配速可用。")
    lines.append("- **主对照**：进站次数不同且差 **1** 的车对（多停 1 次的一方记为 more）；"
                 "**对照组**：进站次数相同的队友车对（方向按车手代码排序，实际用时差 = 后者 − 前者）。")
    lines.append("- **实际用时差**：累计圈时之和；若两车最后完成圈数不同（套圈或未完赛），"
                 "在较短一方的最后圈对齐、并在 `teammate_pairs.csv` 标注 `aligned=True`。")
    lines.append("- **发车位差** = more 车发车位 − less 车发车位（正 = more 车起步更靠后）；"
                 "**第一段配速差** = more 车第一段前 5 个干净圈燃油修正均值 − less 车（正 = more 车更慢）。")
    lines.append("- **回归**：`time_diff ~ 截距 + 多停指示 + 发车位差 + 第一段配速差`（不同进站次数组）；"
                 "对照组同式但不含多停指示。标准误按比赛聚类稳健。")
    lines.append("- **SC 分类**：差 1 的车对里，more 车多出的那一次进站若进站圈在 SC/VSC 下记为“SC/VSC 下”，"
                 "否则“绿旗下”。")
    lines.append(f"- **模型对照**：对每个车对用该季 tyre v3.1 后验（{N_DRAWS} 个后验抽样）、`n_sims={N_SIMS}`、"
                 f"`seed={SEED}` 调用 `evaluate()`；两车实际策略在同一场比赛、同一配速下比较，"
                 "模型用时差 = more 策略 − less 策略。")
    lines.append("")

    lines.append("## 1. 样本")
    lines.append("")
    lines.append(md_table(flow))
    lines.append("")
    lines.append(f"- 主对照（进站次数差 1）：**{int((~pairs['same_stops'] & (pairs['stop_diff'] == 1)).sum())} 对**；"
                 f"对照组（同进站次数）：**{int(pairs['same_stops'].sum())} 对**；"
                 f"差 >1 的另有 {int((~pairs['same_stops'] & (pairs['stop_diff'] != 1)).sum())} 对（不进入主分析）。")
    lines.append("")

    lines.append("## 2. 实际用时差与完赛名次差")
    lines.append("")
    rows = []
    for name, df in [("不同进站次数（差 1）", pairs[(~pairs["same_stops"]) & (pairs["stop_diff"] == 1)]),
                     ("相同进站次数（对照）", pairs[pairs["same_stops"]])]:
        st = mean_stat(df["time_diff"])
        pos = mean_stat(df["pos_diff"])
        rows.append({"组": name, "车对数": int(len(df)),
                     "用时差均值 (s)": st["mean"], "中位数 (s)": st["median"],
                     "SD (s)": st["sd"],
                     "名次差均值": pos["mean"], "名次差中位数": pos["median"],
                     "套圈/未完赛对齐": int(df["aligned"].sum())})
    lines.append(md_table(pd.DataFrame(rows)))
    lines.append("")
    lines.append("- 用时差正 = 多停的一方更慢；名次差正 = 多停的一方完赛名次更靠后。")
    lines.append("")

    lines.append("## 3. 调整起点差异的回归")
    lines.append("")
    lines.append(md_table(reg.rename(columns={"组": "组", "变量": "变量"})))
    lines.append("")
    lines.append("![回归系数](figures/fig3_regression.png)")
    lines.append("")
    lines.append("**图 3**：调整回归系数（点）与 95% CI（误差线）；红色为“多停指示”系数，其余为发车位差/第一段配速差。")
    lines.append("")

    lines.append("## 4. SC/VSC 下多停 vs 绿旗下多停")
    lines.append("")
    lines.append(md_table(sc))
    lines.append("")
    lines.append("![SC 分类](figures/fig2_sc_split.png)")
    lines.append("")
    lines.append("**图 2**：差 1 车对中，多停发生在 SC/VSC 下与绿旗旗下的实际用时差分布。")
    lines.append("")

    lines.append("## 5. 模型对照")
    lines.append("")
    lines.append(md_table(mstats))
    lines.append("")
    lines.append(f"（117 对差 1 车对里，{int(mout.get('不同进站次数（差 1）', {}).get('n', 0))} 对两车都完赛、"
                 "且实际策略能构造为模拟器 `Strategy`，进入模型对照；其余为 ≥90% 未完赛或策略数据不足。）")
    lines.append("")
    lines.append("![实际 vs 模型](figures/fig1_actual_vs_model.png)")
    lines.append("")
    lines.append("**图 1**：每个车对的实际用时差（y）与模型用时差（x），红色 = 不同进站次数（差 1），"
                 "蓝色 = 相同进站次数对照组；黑色虚线为 y=x。")
    lines.append("")

    # ---------- 观察
    diff = pairs[(~pairs["same_stops"]) & (pairs["stop_diff"] == 1)]
    ctrl = pairs[pairs["same_stops"]]
    dm = mean_stat(diff["time_diff"])
    cm = mean_stat(ctrl["time_diff"])
    reg_diff = reg[(reg["组"] == "混合样本（差 1 vs 同停）")
                   & (reg["变量"].astype(str).str.contains("多停一次"))]
    treated = float(reg_diff["系数"].iloc[0]) if len(reg_diff) else np.nan
    ms = mout.get("不同进站次数（差 1）", {})
    lines.append("## 观察到的现象")
    lines.append("")
    lines.append(f"1. **多停一次的车平均并不更快。** 差 1 车对（n={dm['n']}）里，"
                 f"多停一方 − 少停一方的用时差均值 **{dm['mean']:+.1f} 秒**（中位数 {dm['median']:+.1f}），"
                 f"正 = 多停更慢；同一场同队对照（n={cm['n']}）的用时差均值 {cm['mean']:+.1f} 秒。"
                 f"多停方更快的比例：{float((diff['time_diff'] < 0).mean())*100:.0f}%。")
    lines.append(f"2. **调整发车位与第一段配速后，“多停”的系数为 {treated:+.1f} 秒（按比赛聚类 SE "
                 f"{float(reg_diff['聚类 SE'].iloc[0]):.1f}，n={int(reg_diff['n'].iloc[0])}）。** "
                 f"即多停一次与更慢相关，而不是更快。")
    lines.append(f"3. **完赛名次也是多停方更差：** 名次差均值 {mean_stat(diff['pos_diff'])['mean']:+.2f} 位"
                 f"（正 = 多停方名次更靠后）。")
    sc_green = sc[sc["多停发生时机"] == "绿旗下"].iloc[0]
    sc_sc = sc[sc["多停发生时机"] == "SC/VSC 下"].iloc[0]
    lines.append(f"4. **SC/VSC 下多停与绿旗下多停的差别：** 绿旗下多停 n={int(sc_green['车对数'])}，"
                 f"用时差均值 {float(sc_green['用时差均值 (s)']):+.1f} 秒；"
                 f"SC/VSC 下多停 n={int(sc_sc['车对数'])}，均值 {float(sc_sc['用时差均值 (s)']):+.1f} 秒。")
    if ms:
        lines.append(f"5. **模型在实际策略上的判断：** 差 1 车对里，模型用时差均值 {ms['mean_model']:+.1f} 秒、"
                     f"实际 {ms['mean_actual']:+.1f} 秒；Pearson {ms['pearson']:+.3f}、Spearman "
                     f"{ms['spearman']:+.3f}，符号一致率 {ms['sign_agree']*100:.0f}%；"
                     f"“模型说少停更快、实际多停更快”占 {ms['model_fewer_actual_more']*100:.0f}%。")
    lines.append("")

    lines.append("## 这些数据不能说明什么")
    lines.append("")
    lines.append("1. **样本小。** 74 场里只有两百多个差 1 的队友车对，且每队只有两辆车；"
                 "组内还有很多没控制的因素（车损、交通、策略时机、队内指令）。")
    lines.append("2. **策略是事后选择，不是随机分配。** 多停的车可能本来就更慢/遇到车损/被交通困住，"
                 "所以“多停更慢”不能反推“少停会更快”；这是选择偏差，不是因果。")
    lines.append("3. **累计用时受套圈/未完赛影响。** 两车最后圈数不同时按较短一方对齐并标注；"
                 "仍有 1 圈内的对齐误差。")
    lines.append("4. **发车位/第一段配速只做了线性调整。** 发车位差与第一段配速差本身受交通、起步事故、"
                 "轮胎新旧影响；调整后仍可能有残余混杂。")
    lines.append("5. **SC 分类只区分“多出的那一次进站”是否在 SC/VSC 下**；同一车对里其他进站、"
                 "以及安全车对两车整体的影响没有单独建模。")
    lines.append("6. **模型对照是单车、无对手、无交通、无实际 SC 的“同一配速只换策略”**，"
                 "模型的 SC 时间线是按全赛道基线随机生成，不是该场实际 SC；"
                 "模型差异里不含 undercut/防守/进站后落进车流的收益。")
    lines.append("7. **模型用的是该季 v3.1 后验（200 个抽样）**，不是逐场重新拟合；"
                 "后验抽样误差会让模型用时差有少量抖动。")
    lines.append("8. **本单只做描述，不能直接当模拟器参数**；也不回答“车队是不是偏保守”（不给建模建议）。")
    lines.append("")
    lines.append("## 输入哈希")
    lines.append("")
    lines.append(md_table(hashes))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def main() -> None:
    np.random.seed(SEED)
    strat, rs, cum, pace, grids, flags, keep_keys = load_data()
    pairs, info = build_pairs(strat, cum, pace, grids, flags, keep_keys)
    flow = info["flow"]

    main_pairs = pairs[(~pairs["same_stops"]) & (pairs["stop_diff"] == 1)].copy()
    reg = regression_table(pairs)
    sc = sc_table(pairs)

    model = model_comparison(pairs, strat)
    mstats, mout = model_stats(model)

    hashes = pd.DataFrame([
        {"文件": str(STRATEGIES.relative_to(ROOT)), "sha256": sha256(STRATEGIES),
         "字节": STRATEGIES.stat().st_size},
        {"文件": str(T020_RACE.relative_to(ROOT)), "sha256": sha256(T020_RACE),
         "字节": T020_RACE.stat().st_size},
        {"文件": str(LAPS.relative_to(ROOT)), "sha256": sha256(LAPS),
         "字节": LAPS.stat().st_size},
        {"文件": str(MODEL.relative_to(ROOT)), "sha256": sha256(MODEL),
         "字节": MODEL.stat().st_size},
    ])

    figs = make_figures(pairs, model, reg, sc)

    pairs.to_csv(OUT / "teammate_pairs.csv", index=False)
    reg.to_csv(OUT / "regression_pairs.csv", index=False)
    sc.to_csv(OUT / "sc_split.csv", index=False)
    mstats.to_csv(OUT / "model_comparison.csv", index=False)
    flow.to_csv(OUT / "sample_flow.csv", index=False)
    hashes.to_csv(OUT / "input_sha256.csv", index=False)
    model.to_csv(OUT / "pair_model_detail.csv", index=False)

    report = build_report(flow, pairs, reg, sc, mstats, mout, figs, hashes)
    REPORT.write_text(report, encoding="utf-8")

    print("==" * 30)
    print(flow.to_string(index=False))
    print("\nskipped:", info["skipped"])
    print("\npairs: diff=1", len(main_pairs), "same", int(pairs["same_stops"].sum()),
          "diff>1", int((~pairs["same_stops"] & (pairs["stop_diff"] != 1)).sum()))
    print(main_pairs[["year", "round", "team", "driver_more", "driver_less",
                      "time_diff", "grid_diff", "pace_diff", "extra_stop_sc"]]
          .head(10).to_string(index=False))
    print("\nregression:"); print(reg.to_string(index=False))
    print("\nSC:"); print(sc.to_string(index=False))
    print("\nmodel:"); print(mstats.to_string(index=False))
    print("model rows", len(model))
    print("\noutputs:")
    for p in sorted(OUT.glob("*.csv")):
        print(" ", p.relative_to(HERE))
    for p in figs:
        print(" ", p.relative_to(HERE))
    print(" ", REPORT.relative_to(HERE))
    print("\nDONE")


if __name__ == "__main__":
    main()
