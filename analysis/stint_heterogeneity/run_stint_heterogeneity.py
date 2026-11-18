"""T026: 段与段之间的衰减差异，以及它和进站决定的关系（只做描述）。

一条命令重跑全部结果（在项目根目录执行）：

    .venv/bin/python analysis/stint_heterogeneity/run_stint_heterogeneity.py

输入（只读，全部在本地，不联网、不安装依赖）：
  data/processed/model_laps.parquet              干净圈（81 场，77,652 行，3,957 段）
  analysis/actual_strategies/stints.csv          OpenF1 stint 原始起止圈（段长、最后一段判定）
  analysis/actual_strategies/strategies.csv      每车每场进站次数、起步配方、完赛信息
  analysis/actual_strategies/outputs/race_summary.csv  T020 场次筛选
  data/processed/laps_pit_filled.parquet         进站/SC 旗标（SC 提前进站段）
  data/processed/fcy_race_summary.csv            赛道映射（备用）

输出（全部在 analysis/stint_heterogeneity/ 下）：
  outputs/stint_slopes.csv              每段 stint 的斜率、SE、前 8 圈斜率、段长等
  outputs/slope_distribution.csv        按配方的斜率分布与扣除估计误差后的真实 SD
  outputs/variance_decomposition.csv    总方差 = 站×配方之间 + 站×配方之内 + 估计误差（含按站自助 CI）
  outputs/first8_vs_length.csv          前 8 圈斜率与最终段长的相关与分箱
  outputs/pitstop_highlow.csv           同场同配方高/低斜率车的进站次数比较
  outputs/extra_stop.csv                计划外多停一次 vs 主流进站次数的前 8 圈斜率比较
  outputs/driver_team_variance.csv      车手/车队同一赛季平均水平能解释多少段斜率
  outputs/input_sha256.csv
  figures/fig1_slope_distributions.png
  figures/fig2_variance_decomposition.png
  figures/fig3_first8_pit.png
  report.md

只写 analysis/stint_heterogeneity/；不删除数据、不提交 git、不安装依赖。
唯一随机过程是按赛道分块自助抽样（固定种子 SEED=20261117）。
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="stinthet-mpl-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------- 常量
SEED = 20261117
FUEL_PER_LAP = 0.06
MIN_STINT_LAPS = 10
MAX_SLOPE_AGE = 20
COMPOUNDS = ["HARD", "MEDIUM", "SOFT"]
BOOT_REPS = 1000

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROCESSED = ROOT / "data" / "processed"
MODEL = PROCESSED / "model_laps.parquet"
STINTS = ROOT / "analysis" / "actual_strategies" / "stints.csv"
STRATEGIES = ROOT / "analysis" / "actual_strategies" / "strategies.csv"
T020_RACE = ROOT / "analysis" / "actual_strategies" / "outputs" / "race_summary.csv"
LAPS = PROCESSED / "laps_pit_filled.parquet"

OUT = HERE / "outputs"
FIG = HERE / "figures"
LOG = HERE / "logs"
REPORT = HERE / "report.md"
for _d in (OUT, FIG, LOG):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 小工具
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(x, nd: int = 3) -> str:
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


def display_df(df: pd.DataFrame, nd: int = 3) -> pd.DataFrame:
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


def ols_slope(age: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """单段 OLS 斜率及其标准误。"""
    age = np.asarray(age, float)
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(age)), age])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(age) - 2
    if dof <= 0 or np.sum((age - age.mean()) ** 2) == 0:
        return np.nan, np.nan
    se = float(np.sqrt((resid @ resid / dof) / np.sum((age - age.mean()) ** 2)))
    return float(beta[1]), se


def ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> dict:
    """OLS + 按 cluster 的 CR0 稳健 SE。"""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtXi = np.linalg.pinv(X.T @ X)
    uniq, inv = np.unique(groups, return_inverse=True)
    G = len(uniq)
    scores = np.zeros((G, X.shape[1]))
    np.add.at(scores, inv, X * resid[:, None])
    meat = scores.T @ scores * G / max(G - 1, 1)
    V = XtXi @ meat @ XtXi
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return {"beta": beta, "se": se, "n": len(y), "n_clusters": G,
            "df": max(G - 1, 1)}


def r2_ols(y: np.ndarray, X: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(y)), X])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))


# ---------------------------------------------------------------- 数据准备
def load_and_prepare():
    ml = pd.read_parquet(MODEL)
    st = pd.read_csv(STINTS).rename(columns={"stint_number": "stint"})
    strat = pd.read_csv(STRATEGIES)
    t020 = pd.read_csv(T020_RACE)
    laps = pd.read_parquet(LAPS)
    for c in ["pit_in_filled", "safety_car", "virtual_safety_car"]:
        laps[c] = laps[c].fillna(False).astype(bool)

    ml["t_corr"] = ml["lap_time_s"].astype(float) - FUEL_PER_LAP * ml["laps_remaining"].astype(float)

    # stint 元数据：计划段长、最后一段、因 SC/VSC 提前进站
    stm = st[["year", "round", "driver", "stint", "lap_start", "lap_end"]].copy()
    stm["length"] = stm["lap_end"] - stm["lap_start"] + 1
    mx = stm.groupby(["year", "round", "driver"])["stint"].max().rename("max_stint").reset_index()
    stm = stm.merge(mx, on=["year", "round", "driver"], how="left")
    stm["is_last"] = stm["stint"] == stm["max_stint"]
    fl = laps[["year", "round", "driver", "lap", "pit_in_filled",
               "safety_car", "virtual_safety_car"]].copy()
    nxt = fl[["year", "round", "driver", "lap", "safety_car", "virtual_safety_car"]].copy()
    nxt["lap"] = nxt["lap"] - 1
    nxt = nxt.rename(columns={"safety_car": "out_sc", "virtual_safety_car": "out_vsc"})
    fl = fl.merge(nxt, on=["year", "round", "driver", "lap"], how="left")
    fl["sc_end"] = (fl["safety_car"] | fl["virtual_safety_car"]
                    | fl["out_sc"].fillna(False) | fl["out_vsc"].fillna(False))
    fl = fl.rename(columns={"lap": "lap_end", "pit_in_filled": "pit_end"})
    stm = stm.merge(fl[["year", "round", "driver", "lap_end", "pit_end", "sc_end"]],
                    on=["year", "round", "driver", "lap_end"], how="left")
    stm["sc_short"] = (stm["pit_end"].fillna(False) & (~stm["is_last"])
                       & stm["sc_end"].fillna(False))
    ml = ml.merge(stm[["year", "round", "driver", "stint", "length",
                       "is_last", "sc_short"]],
                  on=["year", "round", "driver", "stint"], how="left")
    return ml, stm, strat, t020


def stint_slopes(ml: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uid, g in ml.groupby("stint_uid"):
        if len(g) < MIN_STINT_LAPS:
            continue
        r = g.iloc[0]
        g20 = g[g["tyre_life"] <= MAX_SLOPE_AGE].sort_values("tyre_life")
        g8 = g.sort_values("tyre_life").head(8)

        def fit(sub):
            if len(sub) < 5:
                return np.nan, np.nan, len(sub)
            age = sub["tyre_life"].to_numpy(float)
            y = sub["t_corr"].to_numpy(float)
            if np.var(age) == 0:
                return np.nan, np.nan, len(sub)
            slope, se = ols_slope(age, y)
            return slope, se, len(sub)

        slope, se, n20 = fit(g20)
        slope8, se8, n8 = fit(g8)
        rows.append({
            "stint_uid": uid, "year": int(r["year"]), "round": int(r["round"]),
            "circuit_id": r["circuit_id"], "driver": r["driver"], "team": r["team"],
            "compound": r["compound"], "stint": int(r["stint"]),
            "length": float(r["length"]), "is_last": bool(r["is_last"]),
            "sc_short": bool(r["sc_short"]),
            "n_laps": int(len(g)), "n20": int(n20),
            "slope": slope, "se": se, "slope8": slope8, "se8": se8, "n8": int(n8),
        })
    tab = pd.DataFrame(rows)
    if len(tab):
        tab["race_key"] = tab["year"].astype(str) + "|" + tab["round"].astype(str)
        tab["cell"] = tab["circuit_id"].astype(str) + "|" + tab["compound"].astype(str)
    return tab


# ---------------------------------------------------------------- 方差分解
def decompose(slopes: np.ndarray, ses: np.ndarray, cells: np.ndarray) -> dict:
    """总方差 = 站×配方之间真实方差 + 站×配方之内真实方差 + 估计误差方差。

    估计：
      - 格均值方差（跨格子，ddof=1）减去格均值的抽样方差，得到“之间”真实方差；
      - 格内方差的加权平均（权重 n_c−1）减去格内平均抽样方差，得到“之内”真实方差；
      - 估计误差方差 = 每段 SE² 的均值（全样本口径，用于误差占比）。
    """
    slopes = np.asarray(slopes, float)
    ses = np.asarray(ses, float)
    cells = np.asarray(cells, str)
    n = len(slopes)
    if n < 3:
        return {}
    df = pd.DataFrame({"s": slopes, "se2": ses ** 2, "cell": cells})
    grp = df.groupby("cell")
    cnt = grp.size().to_numpy(float)
    mean = grp["s"].mean().to_numpy(float)
    within_var = grp["s"].var(ddof=1).to_numpy(float)
    mse = grp["se2"].mean().to_numpy(float)
    C = len(cnt)
    # 格均值抽样方差 ≈ 格内平均 SE² / n_c
    samp_var_mean = mse / cnt
    if C >= 2:
        v_between_true = max(float(np.var(mean, ddof=1) - np.mean(samp_var_mean)), 0.0)
    else:
        v_between_true = 0.0
    w = np.where(cnt > 1, cnt - 1, 0.0)
    if w.sum() > 0:
        v_within_obs = float(np.sum(w * np.nan_to_num(within_var)) / w.sum())
        v_within_err = float(np.sum(w * mse) / w.sum())
        v_within_true = max(v_within_obs - v_within_err, 0.0)
    else:
        v_within_true = 0.0
    v_true = v_between_true + v_within_true
    v_obs = float(np.var(slopes, ddof=1))
    v_err = float(np.mean(ses ** 2))
    return {
        "n_stints": n,
        "v_obs": v_obs, "sd_obs": float(np.sqrt(v_obs)),
        "v_err": v_err, "sd_err": float(np.sqrt(v_err)),
        "v_true": v_true, "sd_true": float(np.sqrt(v_true)),
        "v_between_true": v_between_true,
        "sd_between_true": float(np.sqrt(v_between_true)),
        "v_within_true": v_within_true,
        "sd_within_true": float(np.sqrt(v_within_true)),
        "err_share": v_err / v_obs if v_obs > 0 else np.nan,
    }


def variance_table(tab: pd.DataFrame, circuits: list) -> tuple[pd.DataFrame, dict]:
    rows = []
    boot = {}
    rng = np.random.default_rng(SEED)
    subsets = [("全部", tab)] + [(c, tab[tab["compound"] == c]) for c in COMPOUNDS]
    for label, sub in subsets:
        d = decompose(sub["slope"], sub["se"], sub["cell"])
        if not d:
            continue
        # 按赛道分块自助
        circuits_here = sorted(sub["circuit_id"].unique())
        boots = []
        for _ in range(BOOT_REPS):
            pick = rng.choice(circuits_here, size=len(circuits_here), replace=True)
            parts = []
            for i, c in enumerate(pick):
                s = sub[sub["circuit_id"] == c].copy()
                if len(s) == 0:
                    continue
                s["stint_uid"] = s["stint_uid"].astype(str) + f"__{i}"
                s["cell"] = s["cell"].astype(str) + f"__{i}"
                parts.append(s)
            if not parts:
                continue
            bs = pd.concat(parts, ignore_index=True)
            boots.append(decompose(bs["slope"], bs["se"], bs["cell"]))
        bdf = pd.DataFrame(boots)
        row = {"分组": label, **d}
        for key in ["sd_true", "sd_between_true", "sd_within_true", "sd_err"]:
            row[f"{key}_lo"] = float(bdf[key].quantile(0.025))
            row[f"{key}_hi"] = float(bdf[key].quantile(0.975))
        rows.append(row)
        boot[label] = bdf
    return pd.DataFrame(rows), boot


# ---------------------------------------------------------------- 3a / 3b / 3c
def first8_vs_length(tab: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    s = tab[(~tab["is_last"].fillna(False)) & (~tab["sc_short"].fillna(False))
            & tab["slope8"].notna()].copy()
    pearson = float(s["slope8"].corr(s["length"]))
    spearman = float(s["slope8"].corr(s["length"], method="spearman"))
    s["q"] = pd.qcut(s["slope8"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    rows = []
    for q, g in s.groupby("q", observed=True):
        rows.append({"前8圈斜率分位": str(q), "n": int(len(g)),
                     "斜率均值": float(g["slope8"].mean()),
                     "段长均值": float(g["length"].mean()),
                     "段长中位数": float(g["length"].median())})
    # 按段长分箱看前 8 圈斜率
    bins = [-np.inf, 15, 20, 25, 30, 40, np.inf]
    labels = ["≤15", "16–20", "21–25", "26–30", "31–40", ">40"]
    s["bin"] = pd.cut(s["length"], bins, labels=labels)
    rows2 = []
    for b in labels:
        g = s[s["bin"] == b]
        if len(g) == 0:
            continue
        rows2.append({"段长分箱": b, "n": int(len(g)),
                      "前8圈斜率均值": float(g["slope8"].mean()),
                      "前8圈斜率中位数": float(g["slope8"].median())})
    return pd.DataFrame(rows), {"pearson": pearson, "spearman": spearman,
                                "n": int(len(s)), "by_length": pd.DataFrame(rows2)}


def pitstop_highlow(tab: pd.DataFrame, strat: pd.DataFrame, t020: pd.DataFrame):
    keep = set(map(tuple, t020[t020["usable"] & t020["dry_for_analysis"]
                                & (~t020["has_red_flag"])][["year", "round"]].values))
    st = strat[["year", "round", "driver", "n_pit_stops", "start_compound",
                "finisher", "finish_position"]].copy()
    ts = tab.merge(st, on=["year", "round", "driver"], how="left")
    ts = ts.sort_values(["year", "round", "driver", "stint"]).reset_index(drop=True)
    ts["min_stint"] = ts.groupby(["year", "round", "driver"])["stint"].transform("min")
    ts = ts[ts["stint"] == ts["min_stint"]].copy()
    ts = ts[ts["finisher"].fillna(False) & ts["start_compound"].isin(COMPOUNDS)
            & ts["slope8"].notna()]
    ts = ts[[(int(y), int(r)) in keep for y, r in zip(ts["year"], ts["round"])]]
    ts["group"] = ts["year"].astype(str) + "|" + ts["round"].astype(str) + "|" + ts["start_compound"]
    counts = ts["group"].value_counts()
    ts = ts[ts["group"].isin(counts[counts >= 4].index)]
    ts["median8"] = ts.groupby("group")["slope8"].transform("median")
    ts["high"] = ts["slope8"] > ts["median8"]
    ts["ns_w"] = ts["n_pit_stops"] - ts.groupby("group")["n_pit_stops"].transform("mean")
    high = ts[ts["high"]]
    low = ts[~ts["high"]]
    y = ts["ns_w"].to_numpy(float)
    X = np.column_stack([np.ones(len(ts)), ts["high"].astype(float).to_numpy()])
    fit = ols_cluster(y, X, ts["race_key"].to_numpy())
    rows = [
        {"组": "高斜率（>同场同配方中位数）", "n": int(len(high)),
         "进站次数均值": float(high["n_pit_stops"].mean()),
         "进站次数中位数": float(high["n_pit_stops"].median())},
        {"组": "低斜率（≤中位数）", "n": int(len(low)),
         "进站次数均值": float(low["n_pit_stops"].mean()),
         "进站次数中位数": float(low["n_pit_stops"].median())},
    ]
    summary = {
        "within_diff": float(fit["beta"][1]),
        "within_se": float(fit["se"][1]),
        "p": float(2 * (1 - stats.t.cdf(abs(fit["beta"][1] / fit["se"][1]), fit["df"]))),
        "n_races": int(ts["race_key"].nunique()),
        "n_cells": int(ts["group"].nunique()),
        "n_cars": int(len(ts)),
        "high_stops_mean": float(high["n_pit_stops"].mean()),
        "low_stops_mean": float(low["n_pit_stops"].mean()),
    }
    return pd.DataFrame(rows), summary, ts


def extra_stop(tab: pd.DataFrame, strat: pd.DataFrame, t020: pd.DataFrame):
    ts = pitstop_highlow(tab, strat, t020)[2].copy()
    top10 = strat[strat["finish_position"].notna() & (strat["finish_position"] <= 10)]
    mainstream = (top10.groupby(["year", "round"])["n_pit_stops"]
                  .agg(lambda x: x.mode().iloc[0]).rename("mainstream").reset_index())
    ts = ts.merge(mainstream, on=["year", "round"], how="left")
    ts = ts[ts["mainstream"].notna()].copy()

    def grp(row):
        if row["n_pit_stops"] == row["mainstream"]:
            return "同主流"
        if row["n_pit_stops"] == row["mainstream"] + 1:
            return "+1 次"
        return "其他"

    ts["group2"] = ts.apply(grp, axis=1)
    sub = ts[ts["group2"].isin(["同主流", "+1 次"])].copy()
    y = sub["slope8"].to_numpy(float)
    X = np.column_stack([np.ones(len(sub)),
                         (sub["group2"] == "+1 次").astype(float).to_numpy()])
    fit = ols_cluster(y, X, sub["race_key"].to_numpy())
    rows = []
    for g in ["同主流", "+1 次", "其他"]:
        s = ts[ts["group2"] == g]
        rows.append({"组": g, "n": int(len(s)),
                     "前8圈斜率均值": float(s["slope8"].mean()),
                     "前8圈斜率中位数": float(s["slope8"].median())})
    summary = {
        "diff": float(fit["beta"][1]),
        "se": float(fit["se"][1]),
        "p": float(2 * (1 - stats.t.cdf(abs(fit["beta"][1] / fit["se"][1]), fit["df"]))),
        "n_plus": int((ts["group2"] == "+1 次").sum()),
        "n_same": int((ts["group2"] == "同主流").sum()),
        "n_races": int(ts["race_key"].nunique()),
        "mainstream_dist": mainstream["mainstream"].value_counts().sort_index().to_dict(),
    }
    return pd.DataFrame(rows), summary, ts


# ---------------------------------------------------------------- 4 车手/车队
def driver_team_variance(tab: pd.DataFrame):
    s = tab[tab["slope"].notna()].copy()
    s["season"] = s["year"]
    s["dr"] = s["year"].astype(str) + "|" + s["driver"].astype(str)
    s["dr_c"] = s["dr"] + "|" + s["compound"].astype(str)
    s["tm"] = s["year"].astype(str) + "|" + s["team"].astype(str)

    def loo_means(key):
        g = s.groupby(key)["slope"]
        ssum = g.transform("sum")
        cnt = g.transform("size")
        return (ssum - s["slope"]) / (cnt - 1)

    s["dr_loo"] = loo_means("dr")
    s["dr_c_loo"] = loo_means("dr_c")
    s["tm_loo"] = loo_means("tm")
    s2 = s.dropna(subset=["dr_loo", "dr_c_loo", "tm_loo"]).copy()
    if len(s2) == 0:
        return pd.DataFrame(columns=["项目", "R2", "ΔR2", "n_stints"])
    y = s2["slope"].to_numpy(float)
    base = pd.get_dummies(s2["circuit_id"].astype(str) + "|" + s2["compound"].astype(str),
                          drop_first=True, dtype=float).to_numpy(float)
    r0 = r2_ols(y, base)
    rows = [{"项目": "站×配方固定效应（基准）", "R2": r0, "ΔR2": np.nan,
             "n_stints": int(len(s2))}]
    for name, col in [("+ 车手同一赛季平均（留一）", "dr_loo"),
                      ("+ 车手同一赛季×配方平均（留一）", "dr_c_loo"),
                      ("+ 车队同一赛季平均（留一）", "tm_loo")]:
        r = r2_ols(y, np.column_stack([base, s2[col].to_numpy(float)]))
        rows.append({"项目": name, "R2": r, "ΔR2": r - r0, "n_stints": int(len(s2))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 图
def make_figures(tab: pd.DataFrame, var_tab: pd.DataFrame, first8: dict,
                 ts_highlow: pd.DataFrame, ts_extra: pd.DataFrame):
    paths = []
    colors = {"HARD": "#4c72b0", "MEDIUM": "#dd8452", "SOFT": "#55a868"}

    # fig1: 每段斜率分布
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharex=True)
    for ax, c in zip(axes, COMPOUNDS):
        s = tab[tab["compound"] == c]["slope"].dropna()
        ax.hist(s, bins=40, color=colors[c], alpha=0.75, edgecolor="black", linewidth=0.3)
        row = var_tab[var_tab["分组"] == c].iloc[0]
        ax.axvline(float(s.mean()), color="black", linewidth=1, linestyle="--")
        ax.axvline(float(s.mean()) - float(row["sd_true"]), color="black", linewidth=0.8, linestyle=":")
        ax.axvline(float(s.mean()) + float(row["sd_true"]), color="black", linewidth=0.8, linestyle=":")
        ax.set_title(f"{c} (n={len(s)})\ntrue SD={row['sd_true']:.3f} s/lap")
        ax.set_xlabel("Per-stint degradation slope (s/lap)")
        ax.set_ylabel("stints")
        ax.grid(alpha=0.25)
    fig.suptitle("Per-stint degradation slope (tyre age <=20, fuel-corrected)")
    fig.tight_layout()
    p = FIG / "fig1_slope_distributions.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # fig2: 方差分解
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    labels = ["全部"] + COMPOUNDS
    display = ["ALL"] + COMPOUNDS
    x = np.arange(len(labels))
    between = []
    within = []
    err = []
    for lab in labels:
        row = var_tab[var_tab["分组"] == lab].iloc[0]
        between.append(float(row["v_between_true"]))
        within.append(float(row["v_within_true"]))
        err.append(float(row["v_err"]))
    ax.bar(x, between, color="#4c72b0", label="between circuit×compound (true)")
    ax.bar(x, within, bottom=between, color="#dd8452",
           label="within circuit×compound (true)")
    ax.bar(x, err, bottom=np.array(between) + np.array(within), color="#999999",
           label="estimation error")
    for i, lab in enumerate(labels):
        row = var_tab[var_tab["分组"] == lab].iloc[0]
        ax.text(i, between[i] + within[i] + err[i] + 0.0003,
                f"SDtrue={row['sd_true']:.3f}\n({row['err_share']:.0%} error)",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(display)
    ax.set_ylabel("Variance of per-stint slope (s/lap)$^2$")
    ax.set_title("Variance decomposition with estimation error removed")
    ax.set_ylim(0, max(np.array(between) + np.array(within) + np.array(err)) * 1.22)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = FIG / "fig2_variance_decomposition.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # fig3: 前 8 圈斜率 vs 段长 + 进站组比较
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    ax = axes[0]
    qtab = first8["qtab"]
    ax.plot(qtab["斜率均值"], qtab["段长均值"], marker="o", color="#4c72b0", linewidth=1.5)
    for r in qtab.itertuples(index=False):
        ax.annotate(f"{r.前8圈斜率分位}\n(n={r.n})", (r.斜率均值, r.段长均值),
                    textcoords="offset points", xytext=(6, 5), fontsize=7.5)
    ax.set_xlabel("First-8-lap slope (s/lap)")
    ax.set_ylabel("Final stint length (laps)")
    ax.set_title(f"First-8 slope vs final length (r={first8['pearson']:.2f})")
    ax.grid(alpha=0.25)

    ax = axes[1]
    sub = ts_extra[ts_extra["group2"].isin(["同主流", "+1 次"])].copy()
    data = [sub[sub["group2"] == "同主流"]["slope8"].dropna(),
            sub[sub["group2"] == "+1 次"]["slope8"].dropna()]
    ax.boxplot(data, tick_labels=["same as mainstream", "one extra stop"], widths=0.5,
               showfliers=False)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i + 1) + np.random.default_rng(SEED).normal(0, 0.04, len(d)),
                   d, s=6, alpha=0.35, color="#4c72b0")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("First-8-lap slope (s/lap)")
    ax.set_title("First-8 slope by pit-stop group")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = FIG / "fig3_first8_pit.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


# ---------------------------------------------------------------- 报告
def build_report(sample_info, dist_tab, var_tab, first8, pit_highlow, pit_summary,
                 extra_tab, extra_summary, dt_tab, figs, hashes) -> str:
    lines = []
    lines.append("# T026　段与段之间的衰减差异，以及它和进站决定的关系")
    lines.append("")
    lines.append("由 `analysis/stint_heterogeneity/run_stint_heterogeneity.py` 离线生成；"
                 "只做描述统计，不评价策略、也不写建模建议。唯一随机过程是按赛道分块自助抽样，"
                 "种子固定 SEED=20261117；两次运行输出一致。")
    lines.append("")
    lines.append("## 口径")
    lines.append("")
    lines.append("- 数据：`model_laps.parquet`（81 场干地可用比赛、77,652 个干净圈、3,957 段 stint）；"
                 f"燃油修正 `t_corr = lap_time_s − {FUEL_PER_LAP:.2f} × laps_remaining`。")
    lines.append(f"- **每段主斜率**：只取该段胎龄 ≤{MAX_SLOPE_AGE} 的干净圈做 OLS（`t_corr ~ 胎龄`），"
                 f"要求该段至少 {MIN_STINT_LAPS} 圈干净圈、斜率窗口至少 5 圈；斜率单位 s/圈（正 = 越跑越慢）。"
                 "标准误为 OLS 斜率标准误。")
    lines.append("- **前 8 圈斜率**：该段最早的 8 个干净圈（按胎龄排序）做同样的 OLS；"
                 "用于第 3 节与最终段长/进站次数的关系（样本更大、噪声也更大）。")
    lines.append("- **段长**：OpenF1 stint 原始起止圈 `lap_end − lap_start + 1`；"
                 "第 3 节主样本剔除比赛最后一段与因 SC/VSC 提前进站的段（与 T024 同口径）。")
    lines.append("- **进站次数**：T020 `strategies.csv` 的 `n_pit_stops`；"
                 "第 3b/3c 节只用干地、无红旗、可用比赛里的完赛车（`finisher=True`），"
                 "且起步配方为 SOFT/MEDIUM/HARD。")
    lines.append(f"- **方差分解**：总观测方差 = 站×配方之间真实方差 + 站×配方之内真实方差 + 估计误差方差；"
                 f"估计误差用每段斜率 SE² 的均值；按赛道分块自助 {BOOT_REPS} 次给 95% 区间。")
    lines.append("- **车手/车队成分**：以站×配方为固定效应，加入车手/车队同一赛季平均斜率（留一法，"
                 "避免自己解释自己），看 R² 增量。")
    lines.append("")

    lines.append("## 1. 每段斜率：段数与筛选")
    lines.append("")
    lines.append(md_table(pd.DataFrame(sample_info["rows"])))
    lines.append("")
    lines.append("按配方：")
    lines.append("")
    lines.append(md_table(sample_info["by_compound"]))
    lines.append("")
    lines.append("每段斜率的分布（s/圈，正 = 越跑越慢）：")
    lines.append("")
    dist_show = dist_tab[["分组", "n", "mean", "median", "sd_obs", "sd_err", "sd_true",
                         "q05", "q95"]].rename(columns={
        "分组": "分组", "n": "段数", "mean": "均值", "median": "中位数",
        "sd_obs": "观测 SD", "sd_err": "估计误差 SD", "sd_true": "扣除误差后 SD",
        "q05": "5% 分位", "q95": "95% 分位"})
    lines.append(md_table(dist_show))
    lines.append("")
    lines.append("![斜率分布](figures/fig1_slope_distributions.png)")
    lines.append("")
    lines.append("**图 1**：三种配方的每段斜率分布；虚线 = 均值，点线 = 均值 ± 扣除估计误差后的真实 SD。")
    lines.append("")

    lines.append("## 2. 差异有多大：方差分解")
    lines.append("")
    vshow = var_tab[["分组", "n_stints", "v_obs", "v_err", "v_true",
                      "sd_true", "sd_true_lo", "sd_true_hi",
                      "sd_between_true", "sd_between_true_lo", "sd_between_true_hi",
                      "sd_within_true", "sd_within_true_lo", "sd_within_true_hi",
                      "err_share"]].rename(columns={
        "分组": "分组", "n_stints": "段数", "v_obs": "观测方差",
        "v_err": "估计误差方差", "v_true": "真实方差",
        "sd_true": "真实 SD", "sd_true_lo": "真实 SD 下", "sd_true_hi": "真实 SD 上",
        "sd_between_true": "站×配方之间 SD", "sd_between_true_lo": "之间 SD 下",
        "sd_between_true_hi": "之间 SD 上",
        "sd_within_true": "站×配方之内 SD", "sd_within_true_lo": "之内 SD 下",
        "sd_within_true_hi": "之内 SD 上", "err_share": "误差占比"})
    lines.append(md_table(vshow))
    lines.append("")
    lines.append("![方差分解](figures/fig2_variance_decomposition.png)")
    lines.append("")
    lines.append("**图 2**：每段斜率方差的分解（站×配方之间真实方差、站×配方之内真实方差、估计误差方差），"
                 "标注扣除误差后的真实 SD 与估计误差占比。")
    lines.append("")

    lines.append("## 3. 和进站的关系")
    lines.append("")
    lines.append("### 3a. 前 8 圈斜率 vs 该段最终长度（主样本：排除最后一段、SC 提前进站段）")
    lines.append("")
    lines.append(f"- Pearson 相关 **{first8['pearson']:+.3f}**，Spearman **{first8['spearman']:+.3f}**"
                 f"（n={first8['n']} 段）。")
    lines.append("")
    lines.append(md_table(first8["qtab"].rename(columns={
        "前8圈斜率分位": "前 8 圈斜率分位", "n": "段数", "斜率均值": "斜率均值",
        "段长均值": "最终段长均值", "段长中位数": "最终段长中位数"})))
    lines.append("")
    lines.append("按最终段长分箱看前 8 圈斜率：")
    lines.append("")
    lines.append(md_table(first8["by_length"].rename(columns={
        "段长分箱": "段长分箱", "n": "段数", "前8圈斜率均值": "前 8 圈斜率均值",
        "前8圈斜率中位数": "前 8 圈斜率中位数"})))
    lines.append("")
    lines.append("### 3b. 同场同配方：前 8 圈斜率高于中位数的车，总进站次数更多吗？")
    lines.append("")
    lines.append(md_table(pit_highlow.rename(columns={
        "组": "组", "n": "车数", "进站次数均值": "进站次数均值",
        "进站次数中位数": "进站次数中位数"})))
    lines.append("")
    lines.append(f"- 同一场比赛×起步配方内（要求格子 ≥4 车）比较，高斜率组比低斜率组的进站次数"
                 f"平均 **{pit_summary['within_diff']:+.3f}** 次（按比赛聚类 SE {pit_summary['within_se']:.3f}，"
                 f"p={fmt_p(pit_summary['p'])}；{pit_summary['n_cars']} 车 / {pit_summary['n_races']} 场 / "
                 f"{pit_summary['n_cells']} 个格子）。原始均值：高 {pit_summary['high_stops_mean']:.2f} 次、"
                 f"低 {pit_summary['low_stops_mean']:.2f} 次。")
    lines.append("")
    lines.append("### 3c. 计划外多停一次的车，起步段前 8 圈斜率更高吗？")
    lines.append("")
    lines.append(md_table(extra_tab.rename(columns={
        "组": "组", "n": "车数", "前8圈斜率均值": "前 8 圈斜率均值",
        "前8圈斜率中位数": "前 8 圈斜率中位数"})))
    lines.append("")
    lines.append(f"- 以该场前十名主流进站次数为参照：比主流多停 1 次的车（n={extra_summary['n_plus']}）"
                 f"与同主流车（n={extra_summary['n_same']}）的前 8 圈斜率差为 "
                 f"**{extra_summary['diff']:+.3f} s/圈**（按比赛聚类 SE {extra_summary['se']:.3f}，"
                 f"p={fmt_p(extra_summary['p'])}）。主流进站次数分布："
                 + "、".join(f"{int(k)} 停：{int(v)} 场"
                            for k, v in extra_summary["mainstream_dist"].items()) + "。")
    lines.append("")
    lines.append("![前8圈与进站](figures/fig3_first8_pit.png)")
    lines.append("")
    lines.append("**图 3**：左 = 前 8 圈斜率四分位对应的最终段长均值；右 = “同主流”和“多停一次”两组的"
                 "起步段前 8 圈斜率分布（箱线图 + 抖动点）。")
    lines.append("")

    lines.append("## 4. 车手 / 车队成分")
    lines.append("")
    lines.append(md_table(dt_tab.rename(columns={
        "项目": "模型", "R2": "R²", "ΔR2": "ΔR²", "n_stints": "段数"})))
    lines.append("")
    lines.append("（留一法：计算某车手/车队同一赛季平均斜率时排除当前这一段；站×配方固定效应作为基准。）")
    lines.append("")

    # ---------- 观察
    allrow = var_tab[var_tab["分组"] == "全部"].iloc[0]
    driver_delta = dt_tab[dt_tab["项目"].str.contains("车手同一赛季平均")]["ΔR2"]
    team_delta = dt_tab[dt_tab["项目"].str.contains("车队同一赛季平均")]["ΔR2"]
    lines.append("## 观察到的现象")
    lines.append("")
    lines.append(f"1. **每段衰减斜率差别很大。** 扣除估计误差后，段间斜率真实 SD ≈ "
                 f"**{allrow['sd_true']:.3f} s/圈**（95% CI {allrow['sd_true_lo']:.3f}–{allrow['sd_true_hi']:.3f}），"
                 f"均值/中位数约 {dist_tab.loc[dist_tab['分组'] == '全部', 'mean'].iloc[0]:.3f}/"
                 f"{dist_tab.loc[dist_tab['分组'] == '全部', 'median'].iloc[0]:.3f} s/圈；"
                 f"5%–95% 分位 {dist_tab.loc[dist_tab['分组'] == '全部', 'q05'].iloc[0]:+.3f} ~ "
                 f"{dist_tab.loc[dist_tab['分组'] == '全部', 'q95'].iloc[0]:+.3f}。"
                 f"也就是说，有的段几乎不掉速、有的段每圈掉 0.1–0.2 秒，差异不是估计噪声。")
    lines.append(f"2. **方差主要来自站×配方之内，而不是站与站之间。** 站×配方之间真实 SD "
                 f"{allrow['sd_between_true']:.3f}、站×配方之内真实 SD {allrow['sd_within_true']:.3f}；"
                 f"估计误差只占观测方差的 {allrow['err_share']:.0%}。")
    lines.append("3. **衰减快的段确实更短：前 8 圈斜率与最终段长负相关 "
                 f"（Pearson {first8['pearson']:+.3f}、Spearman {first8['spearman']:+.3f}）。** "
                 f"前 8 圈斜率最低四分位的段平均 {first8['qtab'].iloc[0]['段长均值']:.1f} 圈、"
                 f"最高四分位 {first8['qtab'].iloc[-1]['段长均值']:.1f} 圈。")
    lines.append(f"4. **但“斜率高的车多停一次”在车层面看不到显著关系。** 同场同配方里，高斜率组的进站次数"
                 f"反而略低（差 {pit_summary['within_diff']:+.3f} 次，p={fmt_p(pit_summary['p'])}）；"
                 f"比前十名主流多停一次的车，起步段前 8 圈斜率略高但同样不显著（差 "
                 f"{extra_summary['diff']:+.3f} s/圈，p={fmt_p(extra_summary['p'])}）。"
                 "两个方向不一致、且都在零附近，不能支持“起步衰减快直接导致多停一次”。")
    lines.append(f"5. **车手/车队同一赛季平均水平几乎解释不了段斜率。** 在站×配方固定效应之上，"
                 f"加车手同一赛季平均（留一）的 ΔR² = "
                 f"{driver_delta.iloc[0]:.4f}，加车队同一赛季平均的 ΔR² = {team_delta.iloc[0]:.4f}。"
                 "段斜率更像是“这一段自己的状态/条件”，而不是车手或车队的稳定特征。")
    lines.append("")

    lines.append("## 这些数据不能说明什么")
    lines.append("")
    lines.append("1. **段斜率不是“纯轮胎衰减”。** 它包含交通（跟车/被堵）、车手主动管胎、赛道演化、"
                 "燃油修正误差、车损/蓝旗等；T023 已说明跟车会显著改变圈时，因此“斜率大”不等于“轮胎本身衰减快”。")
    lines.append("2. **短段的斜率噪声大。** 虽然主斜率要求 ≥10 圈并扣掉了每段 OLS 的估计误差，"
                 "但剩余的真实差异仍可能来自少数几个异常圈；前 8 圈斜率更短、噪声更大。")
    lines.append("3. **反向因果不能从本单区分。** 第 3a 节的“斜率大→段短”既可能是“衰减差被提前结束”，"
                 "也可能是“计划短段的车手/轮胎状态本来就是另一种跑法”；第 3b/3c 节在车层面也看不到"
                 "斜率→多停的正向关系，所以不能把本单当作“边跑边决定”的因果证据。")
    lines.append("4. **进站次数比较受策略/位置/SC 混杂。** 同场同配方只做了粗控制（起步配方、同场中位数），"
                 "没有控制发车位、车队计划、SC 时机、轮胎新旧；多停一次组只有一百多辆车，p 值不稳健。")
    lines.append("5. **车手/车队 R² 小也可能是指标噪声造成的衰减。** 同一车手同一赛季的段数不多，"
                 "留一平均本身噪声大；本单只能说明“用这个样本、这个指标看不到稳定的车手/车队成分”。")
    lines.append("6. **样本范围**：81 场干地可用比赛；湿胎/湿地的段、以及每段开头/结尾被 clean 规则剔除的圈"
                 "不在斜率里；最后一段与 SC 提前进站段只在第 3 节排除，第 1/2 节包含它们（第 1 节没有排除）。")
    lines.append("7. **本单只做描述，不能直接当模拟器参数。** 也不回答“模拟器该不该加段级随机斜率”"
                 "（本单不给建模建议）。")
    lines.append("")
    lines.append("## 输入哈希")
    lines.append("")
    lines.append(md_table(hashes))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def main() -> None:
    np.random.seed(SEED)
    ml, stm, strat, t020 = load_and_prepare()
    tab = stint_slopes(ml)
    usable = tab[tab["slope"].notna()].copy()

    # 1. 段数与筛选
    n_total = ml["stint_uid"].nunique()
    n_ge10 = int((ml.groupby("stint_uid").size() >= MIN_STINT_LAPS).sum())
    n_usable = int(usable["stint_uid"].nunique())
    by_compound = []
    for c in COMPOUNDS:
        s = usable[usable["compound"] == c]
        by_compound.append({"配方": c, "段数": int(len(s)),
                            "圈数": int(s["n_laps"].sum()),
                            "斜率均值": float(s["slope"].mean()),
                            "斜率中位数": float(s["slope"].median())})
    by_compound = pd.DataFrame(by_compound)
    sample_info = {
        "rows": [
            {"项目": "全部 stint 数", "值": n_total},
            {"项目": f"至少 {MIN_STINT_LAPS} 圈干净圈的段数", "值": n_ge10},
            {"项目": "斜率可估计的段数", "值": n_usable},
            {"项目": f"剔除：不足 {MIN_STINT_LAPS} 圈干净圈", "值": n_total - n_ge10},
            {"项目": "剔除：斜率窗口 <5 圈或胎龄无变化", "值": n_ge10 - n_usable},
            {"项目": f"斜率窗口（胎龄≤{MAX_SLOPE_AGE}）圈数合计", "值": int(usable["n20"].sum())},
            {"项目": "前 8 圈斜率可估计的段数", "值": int(usable["slope8"].notna().sum())},
        ],
        "by_compound": by_compound,
    }

    # 2. 分布与方差分解
    dist_rows = []
    for label, sub in [("全部", usable)] + [(c, usable[usable["compound"] == c]) for c in COMPOUNDS]:
        d = decompose(sub["slope"], sub["se"], sub["cell"])
        if not d:
            continue
        q = sub["slope"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
        dist_rows.append({"分组": label, "n": d["n_stints"], **d,
                          "mean": float(sub["slope"].mean()),
                          "median": float(sub["slope"].median()),
                          "q05": float(q[0.05]), "q25": float(q[0.25]),
                          "q50": float(q[0.5]), "q75": float(q[0.75]),
                          "q95": float(q[0.95])})
    dist_tab = pd.DataFrame(dist_rows)
    var_tab, boot = variance_table(usable, sorted(usable["circuit_id"].unique()))

    # 3. 与进站的关系
    qtab, first8 = first8_vs_length(usable)
    first8["qtab"] = qtab
    pit_highlow, pit_summary, ts_high = pitstop_highlow(usable, strat, t020)
    extra_tab, extra_summary, ts_extra = extra_stop(usable, strat, t020)

    # 4. 车手/车队
    dt_tab = driver_team_variance(usable)

    hashes = pd.DataFrame([
        {"文件": str(MODEL.relative_to(ROOT)), "sha256": sha256(MODEL),
         "字节": MODEL.stat().st_size},
        {"文件": str(STINTS.relative_to(ROOT)), "sha256": sha256(STINTS),
         "字节": STINTS.stat().st_size},
        {"文件": str(STRATEGIES.relative_to(ROOT)), "sha256": sha256(STRATEGIES),
         "字节": STRATEGIES.stat().st_size},
        {"文件": str(T020_RACE.relative_to(ROOT)), "sha256": sha256(T020_RACE),
         "字节": T020_RACE.stat().st_size},
        {"文件": str(LAPS.relative_to(ROOT)), "sha256": sha256(LAPS),
         "字节": LAPS.stat().st_size},
    ])

    figs = make_figures(usable, var_tab, {"pearson": first8["pearson"],
                                          "qtab": first8["qtab"]},
                        ts_high, ts_extra)

    tab.to_csv(OUT / "stint_slopes.csv", index=False)
    dist_tab.to_csv(OUT / "slope_distribution.csv", index=False)
    var_tab.to_csv(OUT / "variance_decomposition.csv", index=False)
    first8["qtab"].to_csv(OUT / "first8_vs_length.csv", index=False)
    first8["by_length"].to_csv(OUT / "first8_by_length.csv", index=False)
    pit_highlow.to_csv(OUT / "pitstop_highlow.csv", index=False)
    extra_tab.to_csv(OUT / "extra_stop.csv", index=False)
    dt_tab.to_csv(OUT / "driver_team_variance.csv", index=False)
    hashes.to_csv(OUT / "input_sha256.csv", index=False)

    report = build_report(sample_info, dist_tab, var_tab, first8, pit_highlow,
                          pit_summary, extra_tab, extra_summary, dt_tab, figs, hashes)
    REPORT.write_text(report, encoding="utf-8")

    print("==" * 30)
    print(sample_info["rows"])
    print("\nby compound:")
    print(by_compound.to_string(index=False))
    print("\ndistribution:")
    print(dist_tab[["分组", "n", "mean", "median", "sd_obs", "sd_err", "sd_true",
                    "q05", "q95"]].to_string(index=False))
    print("\nvariance decomposition:")
    print(var_tab[["分组", "n_stints", "v_obs", "v_err", "v_true", "sd_true",
                   "sd_true_lo", "sd_true_hi", "sd_between_true", "sd_within_true"]]
          .to_string(index=False))
    print("\nfirst8 vs length:", {k: first8[k] for k in ["pearson", "spearman", "n"]})
    print(pit_highlow.to_string(index=False))
    print("pit summary:", pit_summary)
    print(extra_tab.to_string(index=False))
    print("extra summary:", extra_summary)
    print("\ndriver/team:")
    print(dt_tab.to_string(index=False))
    print("\noutputs:")
    for p in sorted(OUT.glob("*.csv")):
        print(" ", p.relative_to(HERE))
    for p in figs:
        print(" ", p.relative_to(HERE))
    print(" ", REPORT.relative_to(HERE))
    print("\nDONE")


if __name__ == "__main__":
    main()
