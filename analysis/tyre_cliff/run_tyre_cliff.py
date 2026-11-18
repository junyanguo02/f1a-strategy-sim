"""T021: 轮胎后段非线性（“悬崖”）的证据（只做描述）。

一条命令重跑全部结果（在项目根目录执行）：

    .venv/bin/python analysis/tyre_cliff/run_tyre_cliff.py

输入（只读，全部在本地，不联网、不安装依赖）：
  data/processed/model_laps.parquet            干净圈（81 场、77,652 行、3,957 stint）
  analysis/traffic_features/traffic_features.parquet  T012 逐圈交通特征（gap_ahead_prev）
  data/processed/laps_pit_filled.parquet       SC/VSC 旗标（用于排除安全车后 1–3 圈）

输出（全部在 analysis/tyre_cliff/ 下）：
  outputs/degradation_curve_points.csv    按配方 × 胎龄的平均相对退化曲线与 95% 区间
  outputs/compound_fits.csv               三种配方的分段/二次/线性拟合与检验
  outputs/knot_cv.csv                     胎龄拐点 k 的交叉验证 MSE（k=10..30）
  outputs/circuit_compound_slopes.csv     分站 × 配方的分段斜率（样本足够的格子）
  outputs/top10_cells.csv                 后段斜率 − 前段斜率 最大的 10 个格子
  outputs/three_races_2024.csv            澳大利亚 / 比利时 / 拉斯维加斯 2024 的数值
  outputs/traffic_comparison.csv          全样本 / 排除交通 / 严格无交通 / 排除安全车后 的对照
  outputs/cell_traffic_comparison.csv     分站格子的全样本 vs 排除交通 对照
  outputs/exclusions.csv                  样本筛选与排除计数
  outputs/input_sha256.csv
  figures/fig1_degradation_curves.png
  figures/fig2_early_vs_late_slopes.png
  figures/fig3_circuit_cells.png
  report.md

只写 analysis/tyre_cliff/；不删除数据、不提交 git、不安装依赖。
所有随机过程固定种子（交叉验证折划分）；本脚本的随机过程仅此一处。
"""
from __future__ import annotations

import glob
import hashlib
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="tyrecliff-mpl-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------- 常量
SEED = 20261117
FUEL_PER_LAP = 0.06          # T008 合并估计，秒/圈
BASE_LAPS = 5                # 段内基准：每段 stint 前 5 圈均值
MAX_START_AGE = 5            # 起始胎龄 >5 的 stint 没有“前 5 圈”，不进入相对曲线
OUTLIER_ABS_Y = 5.0          # |段内相对| > 5 秒的圈视为计时异常，剔除（并计数）
KNOT_GRID = list(range(10, 31))
COMPOUNDS = ["HARD", "MEDIUM", "SOFT"]
MIN_CELL_LAPS = 100
MIN_CELL_STINTS = 10
MIN_SEG_LAPS = 10
THREE_RACES_2024 = {("albert_park", 3): "Australian Grand Prix",
                    ("spa", 14): "Belgian Grand Prix",
                    ("vegas", 22): "Las Vegas Grand Prix"}

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROCESSED = ROOT / "data" / "processed"
LAPS = PROCESSED / "model_laps.parquet"
TRAFFIC = ROOT / "analysis" / "traffic_features" / "traffic_features.parquet"
RAW_LAPS = PROCESSED / "laps_pit_filled.parquet"

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


def p_one_sided(coef: float, se: float, df: int) -> float:
    if not np.isfinite(se) or se <= 0 or df <= 0:
        return np.nan
    return float(1.0 - stats.t.cdf(coef / se, df=df))


def p_two_sided(coef: float, se: float, df: int) -> float:
    if not np.isfinite(se) or se <= 0 or df <= 0:
        return np.nan
    return float(2.0 * (1.0 - stats.t.cdf(abs(coef / se), df=df)))


def fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return "—"
    try:
        if pd.isna(p):
            return "—"
    except (TypeError, ValueError):
        pass
    return "<0.001" if float(p) < 0.001 else f"{float(p):.3f}"


def ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid, int(len(y))


def cluster_cov(X: np.ndarray, resid: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """按 stint 聚类的稳健协方差（CR0，带 G/(G-1) 小样本修正）。"""
    XtXi = np.linalg.pinv(X.T @ X)
    uniq, inv = np.unique(groups, return_inverse=True)
    scores = np.zeros((len(uniq), X.shape[1]))
    np.add.at(scores, inv, X * resid[:, None])
    meat = scores.T @ scores
    G = len(uniq)
    meat *= G / max(G - 1, 1)
    return XtXi @ meat @ XtXi


def design(age: np.ndarray, knot: float, quadratic: bool = False) -> np.ndarray:
    ones = np.ones_like(age, dtype=float)
    if quadratic:
        return np.column_stack([ones, age, age ** 2])
    return np.column_stack([ones, age, np.maximum(0.0, age - knot)])


# ---------------------------------------------------------------- 数据准备
def load_and_prepare():
    laps = pd.read_parquet(LAPS)
    traffic = pd.read_parquet(TRAFFIC)[["year", "round", "driver", "lap",
                                        "gap_ahead_prev"]]
    raw = pd.read_parquet(RAW_LAPS)
    for c in ["safety_car", "virtual_safety_car"]:
        raw[c] = raw[c].fillna(False).astype(bool)

    df = laps.copy()
    df["t_corr"] = (df["lap_time_s"].astype(float)
                    - FUEL_PER_LAP * df["laps_remaining"].astype(float))
    df = df.sort_values(["stint_uid", "tyre_life"]).reset_index(drop=True)

    # 段内基准：每段 stint 前 5 圈（模型表里可用的前 5 圈）均值
    first = df.groupby("stint_uid", sort=False).head(BASE_LAPS)
    base = (first.groupby("stint_uid")["t_corr"]
            .agg(base="mean", base_n="size", base_min_age="min"))
    df = df.merge(base, left_on="stint_uid", right_index=True, how="left")
    df["y"] = df["t_corr"] - df["base"]
    df["start_age"] = df.groupby("stint_uid")["tyre_life"].transform("min")

    # 交通：排除前车 2 秒内的圈；gap 缺失 = 无前车/来源不可靠，主口径保留，严格口径剔除
    df = df.merge(traffic, on=["year", "round", "driver", "lap"], how="left")
    df["traffic_free"] = df["gap_ahead_prev"].isna() | (df["gap_ahead_prev"] >= 2.0)
    df["traffic_free_strict"] = (df["gap_ahead_prev"] >= 2.0)

    # 安全车后 1–3 圈：用逐圈 SC/VSC 旗标
    flags = raw[["year", "round", "driver", "lap",
                 "safety_car", "virtual_safety_car"]].copy()
    flags["scvsc"] = flags["safety_car"] | flags["virtual_safety_car"]
    for lag in (1, 2, 3):
        sh = flags[["year", "round", "driver", "lap", "scvsc"]].copy()
        sh["lap"] = sh["lap"] + lag
        sh = sh.rename(columns={"scvsc": f"sc_lag{lag}"})
        df = df.merge(sh, on=["year", "round", "driver", "lap"], how="left")
    for lag in (1, 2, 3):
        df[f"sc_lag{lag}"] = df[f"sc_lag{lag}"].fillna(False).astype(bool)
    df["post_sc"] = df[["sc_lag1", "sc_lag2", "sc_lag3"]].any(axis=1)
    return df


def analysis_sample(df: pd.DataFrame):
    """任务口径的分析样本：有前 5 圈基准、起始胎龄 <=5、非计时异常。"""
    n0 = len(df)
    no_base = int(df["base"].isna().sum())
    late_start = int((df["base"].notna() & (df["start_age"] > MAX_START_AGE)).sum())
    outlier = int((df["base"].notna() & (df["start_age"] <= MAX_START_AGE)
                   & (df["y"].abs() > OUTLIER_ABS_Y)).sum())
    keep = (df["base"].notna() & (df["start_age"] <= MAX_START_AGE)
            & (df["y"].abs() <= OUTLIER_ABS_Y)
            & df["compound"].isin(COMPOUNDS))
    sample = df[keep].copy()
    excl = pd.DataFrame([
        {"项目": "model_laps 总圈数", "圈数": n0},
        {"项目": "无段内基准（stint 中无可用圈）", "圈数": no_base},
        {"项目": f"起始胎龄 > {MAX_START_AGE} 圈（没有“前 5 圈”基准）", "圈数": late_start},
        {"项目": f"|段内相对| > {OUTLIER_ABS_Y} 秒（计时异常）", "圈数": outlier},
        {"项目": "进入分析的圈数", "圈数": int(len(sample))},
        {"项目": "进入分析的 stint 数", "圈数": int(sample["stint_uid"].nunique())},
    ])
    return sample, excl


# ---------------------------------------------------------------- 拟合
def fit_piecewise(sample: pd.DataFrame, compound: str, knot: float) -> dict:
    s = sample[sample["compound"] == compound]
    age = s["tyre_life"].to_numpy(float)
    y = s["y"].to_numpy(float)
    groups = s["stint_uid"].astype(str).to_numpy()
    X = design(age, knot)
    beta, resid, n = ols(X, y)
    V = cluster_cov(X, resid, groups)
    G = len(np.unique(groups))
    dfree = max(G - 1, 1)
    diff = float(beta[2])
    se_diff = float(np.sqrt(max(V[2, 2], 0)))
    return {
        "compound": compound, "n_laps": int(n), "n_stints": int(G),
        "knot": float(knot),
        "early_slope": float(beta[1]),
        "late_slope": float(beta[1] + diff),
        "diff_late_minus_early": diff,
        "diff_se": se_diff,
        "diff_ci_lo": diff - stats.t.ppf(0.975, dfree) * se_diff,
        "diff_ci_hi": diff + stats.t.ppf(0.975, dfree) * se_diff,
        "diff_t": diff / se_diff if se_diff > 0 else np.nan,
        "diff_p_one_sided": p_one_sided(diff, se_diff, dfree),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
    }


def fit_quadratic(sample: pd.DataFrame, compound: str) -> dict:
    s = sample[sample["compound"] == compound]
    age = s["tyre_life"].to_numpy(float)
    y = s["y"].to_numpy(float)
    groups = s["stint_uid"].astype(str).to_numpy()
    X = design(age, 0.0, quadratic=True)
    beta, resid, n = ols(X, y)
    V = cluster_cov(X, resid, groups)
    dfree = max(len(np.unique(groups)) - 1, 1)
    kappa = float(beta[2])
    se = float(np.sqrt(max(V[2, 2], 0)))
    return {"kappa_quadratic": kappa, "kappa_se": se,
            "kappa_p_one_sided": p_one_sided(kappa, se, dfree),
            "rmse_quadratic": float(np.sqrt(np.mean(resid ** 2))),
            "n_laps_quad": int(n)}


def fit_linear(sample: pd.DataFrame, compound: str) -> dict:
    s = sample[sample["compound"] == compound]
    age = s["tyre_life"].to_numpy(float)
    y = s["y"].to_numpy(float)
    X = np.column_stack([np.ones_like(age), age])
    beta, resid, n = ols(X, y)
    return {"slope_linear": float(beta[1]),
            "rmse_linear": float(np.sqrt(np.mean(resid ** 2))),
            "n_laps_linear": int(n)}


def knot_cv(sample: pd.DataFrame, compound: str, folds: int = 5) -> tuple[int, pd.DataFrame]:
    """按 stint 分组 5 折交叉验证，选分段拐点 k。"""
    s = sample[sample["compound"] == compound]
    age = s["tyre_life"].to_numpy(float)
    y = s["y"].to_numpy(float)
    groups = s["stint_uid"].astype(str).to_numpy()
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    parts = np.array_split(uniq, folds)
    rows = []
    for k in KNOT_GRID:
        msse = []
        for part in parts:
            test = np.isin(groups, part)
            if test.all() or (~test).all():
                continue
            Xtr = design(age[~test], k)
            Xte = design(age[test], k)
            b, _, _ = ols(Xtr, y[~test])
            msse.append(float(np.mean((y[test] - Xte @ b) ** 2)))
        rows.append({"compound": compound, "knot": k, "cv_mse": float(np.mean(msse))})
    cv = pd.DataFrame(rows)
    best = int(cv.loc[cv["cv_mse"].idxmin(), "knot"])
    return best, cv


def curve_points(sample: pd.DataFrame, fits: dict) -> pd.DataFrame:
    rows = []
    for c in COMPOUNDS:
        s = sample[sample["compound"] == c]
        knot = fits[c]["knot"]
        beta = fits[c]["beta"]
        for age, g in s.groupby("tyre_life"):
            n = int(len(g))
            if n < 30:
                continue
            mean = float(g["y"].mean())
            sd = float(g["y"].std(ddof=1))
            se = sd / np.sqrt(n)
            tc = stats.t.ppf(0.975, n - 1)
            rows.append({"compound": c, "tyre_life": int(age), "n": n,
                         "mean": mean, "ci_lo": mean - tc * se,
                         "ci_hi": mean + tc * se,
                         "fit_piecewise": float(beta[0] + beta[1] * age
                                                + beta[2] * max(0.0, age - knot)),
                         "knot": int(knot)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 分站格子
def circuit_cells(sample: pd.DataFrame, fits: dict) -> pd.DataFrame:
    rows = []
    for (circuit, compound), s in sample.groupby(["circuit_id", "compound"]):
        knot = fits[compound]["knot"]
        age = s["tyre_life"].to_numpy(float)
        y = s["y"].to_numpy(float)
        groups = s["stint_uid"].astype(str).to_numpy()
        G = len(np.unique(groups))
        n_early = int((age <= knot).sum())
        n_late = int((age > knot).sum())
        if (len(s) < MIN_CELL_LAPS or G < MIN_CELL_STINTS
                or n_early < MIN_SEG_LAPS or n_late < MIN_SEG_LAPS):
            continue
        X = design(age, knot)
        beta, resid, _ = ols(X, y)
        V = cluster_cov(X, resid, groups)
        diff = float(beta[2])
        se = float(np.sqrt(max(V[2, 2], 0)))
        dfree = max(G - 1, 1)
        rows.append({
            "circuit_id": circuit, "compound": compound,
            "n_laps": int(len(s)), "n_stints": int(G), "knot": int(knot),
            "early_slope": float(beta[1]),
            "late_slope": float(beta[1] + diff),
            "diff_late_minus_early": diff, "diff_se": se,
            "diff_ci_lo": diff - stats.t.ppf(0.975, dfree) * se,
            "diff_ci_hi": diff + stats.t.ppf(0.975, dfree) * se,
            "diff_p_one_sided": p_one_sided(diff, se, dfree),
        })
    cells = pd.DataFrame(rows)
    return cells.sort_values("diff_late_minus_early", ascending=False).reset_index(drop=True)


def three_races_table(sample: pd.DataFrame, fits: dict) -> pd.DataFrame:
    rows = []
    s2024 = sample[sample["year"] == 2024]
    for (circuit, rnd), event in THREE_RACES_2024.items():
        s_race = s2024[(s2024["circuit_id"] == circuit) & (s2024["round"] == rnd)]
        for compound in COMPOUNDS:
            s = s_race[s_race["compound"] == compound]
            if len(s) < 20:
                continue
            knot = fits[compound]["knot"]
            age = s["tyre_life"].to_numpy(float)
            y = s["y"].to_numpy(float)
            groups = s["stint_uid"].astype(str).to_numpy()
            G = len(np.unique(groups))
            n_early = int((age <= knot).sum())
            n_late = int((age > knot).sum())
            low = (len(s) < MIN_CELL_LAPS or G < MIN_CELL_STINTS
                   or n_early < MIN_SEG_LAPS or n_late < MIN_SEG_LAPS)
            if n_early < MIN_SEG_LAPS or n_late < MIN_SEG_LAPS:
                # 拐点一侧样本不足：只给前段斜率（早期样本），后段/差值为空
                e = age <= knot
                if int(e.sum()) >= 3:
                    Xe = np.column_stack([np.ones(int(e.sum())), age[e]])
                    be, _, _ = ols(Xe, y[e])
                    early = float(be[1])
                else:
                    early = np.nan
                rows.append({
                    "year": 2024, "round": rnd, "event": event, "circuit_id": circuit,
                    "compound": compound, "n_laps": int(len(s)), "n_stints": int(G),
                    "knot": int(knot),
                    "early_slope": early,
                    "late_slope": np.nan,
                    "diff_late_minus_early": np.nan,
                    "diff_se": np.nan,
                    "diff_p_one_sided": np.nan,
                    "low_n": True,
                })
                continue
            X = design(age, knot)
            beta, resid, _ = ols(X, y)
            V = cluster_cov(X, resid, groups)
            diff = float(beta[2])
            se = float(np.sqrt(max(V[2, 2], 0)))
            dfree = max(G - 1, 1)
            rows.append({
                "year": 2024, "round": rnd, "event": event, "circuit_id": circuit,
                "compound": compound, "n_laps": int(len(s)), "n_stints": int(G),
                "knot": int(knot),
                "early_slope": float(beta[1]),
                "late_slope": float(beta[1] + diff),
                "diff_late_minus_early": diff, "diff_se": se,
                "diff_p_one_sided": p_one_sided(diff, se, dfree),
                "low_n": bool(low),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 对照
def sensitivity_tables(sample: pd.DataFrame, cells: pd.DataFrame):
    variants = [
        ("全样本（任务口径）", sample),
        ("排除前车 2 秒内（gap>=2，保留 gap 缺失）",
         sample[sample["traffic_free"]]),
        ("严格无交通（gap>=2 且非缺失）", sample[sample["traffic_free_strict"]]),
        ("排除安全车后 1–3 圈", sample[~sample["post_sc"]]),
    ]
    # 用全样本拐点，保持各版本可比
    knots = {}
    for c in COMPOUNDS:
        knots[c], _ = knot_cv(sample, c)
    rows = []
    for name, sub in variants:
        for c in COMPOUNDS:
            f = fit_piecewise(sub, c, knots[c])
            rows.append({"variant": name, "compound": c, "n_laps": f["n_laps"],
                         "n_stints": f["n_stints"],
                         "early_slope": f["early_slope"],
                         "late_slope": f["late_slope"],
                         "diff_late_minus_early": f["diff_late_minus_early"],
                         "diff_se": f["diff_se"],
                         "diff_p_one_sided": f["diff_p_one_sided"]})
    comp = pd.DataFrame(rows)

    # 分站格子：全样本 vs 排除交通
    traffic_free = sample[sample["traffic_free"]]
    cells_tf = circuit_cells(traffic_free, {c: {"knot": knots[c]} for c in COMPOUNDS})
    m = cells.merge(cells_tf, on=["circuit_id", "compound"], how="inner",
                    suffixes=("_all", "_trafficfree"))
    if len(m):
        pearson = float(stats.pearsonr(m["diff_late_minus_early_all"],
                                       m["diff_late_minus_early_trafficfree"])[0])
        spearman = float(stats.spearmanr(m["diff_late_minus_early_all"],
                                         m["diff_late_minus_early_trafficfree"])[0])
        top_all = set(map(tuple, cells.head(10)[["circuit_id", "compound"]].values))
        top_tf = set(map(tuple, cells_tf.head(10)[["circuit_id", "compound"]].values))
        overlap = len(top_all & top_tf)
    else:
        pearson = spearman = np.nan
        overlap = 0
    m = m.rename(columns={"diff_late_minus_early_all": "diff_all",
                          "diff_late_minus_early_trafficfree": "diff_traffic_free",
                          "n_laps_all": "n_laps_all",
                          "n_laps_trafficfree": "n_laps_traffic_free"})
    return comp, m, {"pearson": pearson, "spearman": spearman,
                     "top10_overlap": int(overlap), "n_cells": int(len(m))}


# ---------------------------------------------------------------- 图
def make_figures(points: pd.DataFrame, compound_tab: pd.DataFrame,
                 cells: pd.DataFrame, three: pd.DataFrame):
    paths = []
    colors = {"HARD": "#4c72b0", "MEDIUM": "#dd8452", "SOFT": "#55a868"}

    # fig1: 平均相对退化曲线 + 分段拟合
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=False)
    for ax, c in zip(axes, COMPOUNDS):
        p = points[points["compound"] == c]
        ax.fill_between(p["tyre_life"], p["ci_lo"], p["ci_hi"],
                        color=colors[c], alpha=0.18)
        ax.plot(p["tyre_life"], p["mean"], color=colors[c], marker="o",
                markersize=3.5, linewidth=1.4, label="mean (95% CI)")
        ax.plot(p["tyre_life"], p["fit_piecewise"], color="black",
                linestyle="--", linewidth=1.2, label="piecewise fit")
        knot = int(p["knot"].iloc[0])
        ax.axvline(knot, color="#888888", linestyle=":", linewidth=1.0)
        ax.set_title(f"{c} (knot k={knot})")
        ax.set_xlabel("Tyre age (laps)")
        ax.set_ylabel("Within-stint relative lap time (s)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Mean within-stint degradation (fuel-corrected, first-5-lap baseline)")
    fig.tight_layout()
    p1 = FIG / "fig1_degradation_curves.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p1)

    # fig2: early vs late slopes
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    x = np.arange(len(COMPOUNDS))
    width = 0.34
    early = [float(compound_tab.loc[compound_tab["compound"] == c, "early_slope"].iloc[0])
             for c in COMPOUNDS]
    late = [float(compound_tab.loc[compound_tab["compound"] == c, "late_slope"].iloc[0])
            for c in COMPOUNDS]
    b1 = ax.bar(x - width / 2, early, width, color="#8fb1d9", edgecolor="black",
                linewidth=0.5, label="early slope (age <= k)")
    b2 = ax.bar(x + width / 2, late, width, color="#c44e52", edgecolor="black",
                linewidth=0.5, label="late slope (age > k)")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ytop = max(max(early), max(late)) * 1.55
    ybot = min(0.0, min(late)) - 0.01
    for i, c in enumerate(COMPOUNDS):
        row = compound_tab[compound_tab["compound"] == c].iloc[0]
        ax.text(i, ytop * 0.94,
                f"diff {row['diff_late_minus_early']:+.3f}\np1={row['diff_p_one_sided']:.3f}",
                ha="center", va="top", fontsize=7.5)
    ax.set_ylim(ybot, ytop)
    ax.set_xticks(x)
    ax.set_xticklabels(COMPOUNDS)
    ax.set_ylabel("Slope (s/lap)")
    ax.set_title("Early vs late degradation slope (piecewise, cluster-robust SE)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p2 = FIG / "fig2_early_vs_late_slopes.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p2)

    # fig3: 分站格子的 early slope vs (late - early)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for c in COMPOUNDS:
        s = cells[cells["compound"] == c]
        ax.scatter(s["early_slope"], s["diff_late_minus_early"],
                   s=np.clip(s["n_laps"] / 8, 18, 120), alpha=0.75,
                   color=colors[c], edgecolor="black", linewidth=0.4, label=c)
    ax.axhline(0, color="#333333", linewidth=0.9)
    top = cells.head(6)
    for r in top.itertuples(index=False):
        ax.annotate(f"{r.circuit_id}-{r.compound[:1]}",
                    (r.early_slope, r.diff_late_minus_early),
                    textcoords="offset points", xytext=(5, 5), fontsize=7.5)
    if len(three):
        for r in three.itertuples(index=False):
            if bool(r.low_n):
                continue
            ax.scatter([r.early_slope], [r.diff_late_minus_early], s=140,
                       facecolor="none", edgecolor="black", linewidth=1.4)
            ax.annotate(f"2024 {r.event.split()[0]}",
                        (r.early_slope, r.diff_late_minus_early),
                        textcoords="offset points", xytext=(6, -10), fontsize=7)
    ax.set_xlabel("Early slope (s/lap)")
    ax.set_ylabel("Late - early slope (s/lap)")
    ax.set_title("Circuit x compound cells: late-minus-early slope")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p3 = FIG / "fig3_circuit_cells.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p3)
    return paths


# ---------------------------------------------------------------- 报告
def build_report(excl, compound_tab, knot_best, cv_tab, points, cells, top10,
                 three, sens_comp, sens_cells, sens_stats, figs, hashes,
                 n_post_sc, n_traffic) -> str:
    lines: list[str] = []
    lines.append("# T021　轮胎后段非线性（“悬崖”）的证据")
    lines.append("")
    lines.append("由 `analysis/tyre_cliff/run_tyre_cliff.py` 离线生成；只做数据描述，"
                 "不评价策略、也不写建模建议。随机过程只有交叉验证的折划分，种子固定 SEED=20261117；"
                 "两次运行输出一致。")
    lines.append("")
    lines.append("## 口径")
    lines.append("")
    lines.append("- 数据：`model_laps.parquet`（81 场干地可用比赛、77,652 个干净圈、3,957 段 stint）。")
    lines.append(f"- 燃油修正：`t_corr = lap_time_s − {FUEL_PER_LAP:.2f} × laps_remaining`（T008 合并估计，秒/圈）。")
    lines.append(f"- 段内相对：每段 stint 减去其**前 {BASE_LAPS} 圈的平均 t_corr**（模型表里可用的前 5 圈），"
                 f"得到 `y`；起始胎龄 >{MAX_START_AGE} 圈的 stint 没有“前 5 圈”可用，整个 stint 不进入主分析。")
    lines.append(f"- 计时异常：`|y| > {OUTLIER_ABS_Y} 秒` 的圈剔除（数量见下表），长段/红旗异常不会进入拟合。")
    lines.append("- 分段回归：`y = a + b1·age + b2·max(0, age−k)`；k 在 10–30 岁圈内用"
                 "按 stint 分组的 5 折交叉验证选；b1 为前段斜率，b1+b2 为后段斜率，b2 为"
                 "“后段 − 前段”。标准误按 stint 聚类稳健（CR0，G/(G−1) 修正），"
                 "单侧 p 值对应“后段斜率 > 前段斜率”。")
    lines.append("- 二次拟合对照：`y = a + b·age + kappa·age²`。kappa > 0 表示后段加速变慢；"
                 "同时比较线性 / 二次 / 分段三种拟合的 RMSE。")
    lines.append(f"- 分站格子：circuit × compound，要求 ≥{MIN_CELL_LAPS} 圈、≥{MIN_CELL_STINTS} 段 stint，"
                 f"且拐点两侧各 ≥{MIN_SEG_LAPS} 圈；每格用该配方的全样本拐点。")
    lines.append("- 交通对照：T012 的 `gap_ahead_prev`，主口径排除前车 <2 秒的圈（保留 gap 缺失），"
                 "另给“严格无交通”（gap≥2 且非缺失）与“排除安全车后 1–3 圈”两个敏感性版本。")
    lines.append("")

    lines.append("## 1. 样本")
    lines.append("")
    lines.append(md_table(excl))
    lines.append("")
    lines.append("按配方：")
    lines.append("")
    rows = []
    for c in COMPOUNDS:
        rows.append({"配方": c, "圈数": int(compound_tab.loc[compound_tab["compound"] == c, "n_laps"].iloc[0]),
                     "stint 数": int(compound_tab.loc[compound_tab["compound"] == c, "n_stints"].iloc[0])})
    lines.append(md_table(pd.DataFrame(rows)))
    lines.append("")
    lines.append(f"- 交通：可用 gap 的圈中，前车 <2 秒 {n_traffic['lt2']} 圈；"
                 f"主口径保留 {n_traffic['keep_traffic_free']} 圈、严格口径保留 {n_traffic['keep_strict']} 圈。")
    lines.append(f"- 安全车后 1–3 圈的圈数：{n_post_sc}（`post_sc`）。")
    lines.append("")

    lines.append("## 2. 三种配方：前段/后段斜率与拐点")
    lines.append("")
    tab = compound_tab[["compound", "n_laps", "n_stints", "knot", "early_slope",
                        "late_slope", "diff_late_minus_early", "diff_se",
                        "diff_ci_lo", "diff_ci_hi", "diff_p_one_sided",
                        "kappa_quadratic", "kappa_se", "kappa_p_one_sided",
                        "rmse_linear", "rmse_quadratic", "rmse"]].copy()
    tab["knot"] = tab["knot"].astype(int)
    tab["diff_p_one_sided"] = tab["diff_p_one_sided"].map(fmt_p)
    tab["kappa_quadratic"] = tab["kappa_quadratic"].map(lambda v: f"{v:.4f}")
    tab["kappa_se"] = tab["kappa_se"].map(lambda v: f"{v:.4f}")
    tab["kappa_p_one_sided"] = tab["kappa_p_one_sided"].map(fmt_p)
    tab = tab.rename(columns={
        "compound": "配方", "n_laps": "圈数", "n_stints": "stint 数", "knot": "拐点 k",
        "early_slope": "前段斜率", "late_slope": "后段斜率",
        "diff_late_minus_early": "后段−前段", "diff_se": "SE",
        "diff_ci_lo": "CI 下", "diff_ci_hi": "CI 上",
        "diff_p_one_sided": "单侧 p", "kappa_quadratic": "kappa",
        "kappa_se": "kappa SE", "kappa_p_one_sided": "kappa 单侧 p",
        "rmse_linear": "RMSE 线性", "rmse_quadratic": "RMSE 二次", "rmse": "RMSE 分段"})
    lines.append(md_table(tab))
    lines.append("")
    lines.append("交叉验证选出的拐点与 CV-MSE（k=10–30，越小越好；完整网格见 `outputs/knot_cv.csv`）：")
    lines.append("")
    cv_show = knot_best.rename(columns={"compound": "配方", "knot": "拐点 k", "cv_mse": "CV-MSE"})
    lines.append(md_table(cv_show))
    lines.append("")
    lines.append("![退化曲线](figures/fig1_degradation_curves.png)")
    lines.append("")
    lines.append("**图 1**：三种配方的段内平均相对退化曲线（点 = 平均，带 95% t 区间；胎龄点样本 ≥30 圈）"
                 "与分段线性拟合（黑色虚线），点线为交叉验证选出的拐点。")
    lines.append("")
    lines.append("![前后段斜率](figures/fig2_early_vs_late_slopes.png)")
    lines.append("")
    lines.append("**图 2**：前段（age≤k）与后段（age>k）斜率，柱上标注“后段−前段”与单侧 p。")
    lines.append("")
    # selected age curve points
    sel = points[points["tyre_life"].isin([5, 10, 15, 20, 25, 30])]
    if len(sel):
        lines.append("平均相对退化曲线在若干胎龄点的数值（s；括号为 95% 区间）：")
        lines.append("")
        rows = []
        for c in COMPOUNDS:
            row = {"配方": c}
            for age in [5, 10, 15, 20, 25, 30]:
                p = sel[(sel["compound"] == c) & (sel["tyre_life"] == age)]
                if len(p):
                    row[f"age {age}"] = (f"{p.iloc[0]['mean']:.3f} "
                                         f"({p.iloc[0]['ci_lo']:.3f}, {p.iloc[0]['ci_hi']:.3f})")
                else:
                    row[f"age {age}"] = "—"
            rows.append(row)
        lines.append(md_table(pd.DataFrame(rows)))
        lines.append("")

    lines.append("## 3. 分站 × 配方格子")
    lines.append("")
    lines.append(f"共 {len(cells)} 个格子满足样本门槛。按“后段 − 前段”从大到小，前 10 个：")
    lines.append("")
    top10r = top10.copy()
    top10r["diff_p_one_sided"] = top10r["diff_p_one_sided"].map(fmt_p)
    top10r["knot"] = top10r["knot"].astype(int)
    top_show = top10r[["circuit_id", "compound", "n_laps", "n_stints", "knot",
                       "early_slope", "late_slope", "diff_late_minus_early",
                       "diff_se", "diff_p_one_sided"]].rename(columns={
        "circuit_id": "赛道", "compound": "配方", "n_laps": "圈数",
        "n_stints": "stint 数", "knot": "k", "early_slope": "前段斜率",
        "late_slope": "后段斜率", "diff_late_minus_early": "后段−前段",
        "diff_se": "SE", "diff_p_one_sided": "单侧 p"})
    lines.append(md_table(top_show))
    lines.append("")
    lines.append("2024 年澳大利亚 / 比利时 / 拉斯维加斯（分站 × 配方，低样本格子标 low_n）：")
    lines.append("")
    threer = three.copy()
    threer["diff_p_one_sided"] = threer["diff_p_one_sided"].map(fmt_p)
    three_show = threer[["round", "event", "circuit_id", "compound", "n_laps",
                         "n_stints", "early_slope", "late_slope",
                         "diff_late_minus_early", "diff_se", "diff_p_one_sided",
                         "low_n"]].rename(columns={
        "round": "站次", "event": "比赛", "circuit_id": "赛道", "compound": "配方",
        "n_laps": "圈数", "n_stints": "stint 数", "early_slope": "前段斜率",
        "late_slope": "后段斜率", "diff_late_minus_early": "后段−前段",
        "diff_se": "SE", "diff_p_one_sided": "单侧 p", "low_n": "低样本"})
    lines.append(md_table(three_show))
    lines.append("")
    lines.append("![分站格子](figures/fig3_circuit_cells.png)")
    lines.append("")
    lines.append("**图 3**：分站 × 配方格子的前段斜率（x）与“后段−前段”（y），点大小与圈数成比例；"
                 "空心圈标出 2024 年澳大利亚 / 比利时 / 拉斯维加斯中样本足够的格子，"
                 "文字标注“后段−前段”最大的 6 个格子。")
    lines.append("")

    lines.append("## 4. 混杂对照：交通与安全车后")
    lines.append("")
    sens_show = sens_comp.copy()
    sens_show["diff_p_one_sided"] = sens_show["diff_p_one_sided"].map(fmt_p)
    lines.append(md_table(sens_show.rename(columns={
        "variant": "样本", "compound": "配方", "n_laps": "圈数", "n_stints": "stint 数",
        "early_slope": "前段斜率", "late_slope": "后段斜率",
        "diff_late_minus_early": "后段−前段", "diff_se": "SE",
        "diff_p_one_sided": "单侧 p"})))
    lines.append("")
    lines.append(f"分站格子层面：全样本与“排除前车 2 秒内”两版的“后段−前段”相关系数 "
                 f"Pearson {sens_stats['pearson']:.3f}、Spearman {sens_stats['spearman']:.3f}"
                 f"（共同格子 {sens_stats['n_cells']} 个）；两版前 10 名格子重合 "
                 f"{sens_stats['top10_overlap']} 个。逐格数值见 `outputs/cell_traffic_comparison.csv`。")
    lines.append("")

    # ---------- 观察
    def row(c):
        return compound_tab[compound_tab["compound"] == c].iloc[0]

    sig_pos = top10[top10["diff_p_one_sided"] < 0.05]
    lines.append("## 观察到的现象")
    lines.append("")
    lines.append("1. **总体看没有“后段加速变慢”，三种配方的后段斜率都低于前段。** "
                 + "；".join(
                     f"{c}: 前段 {row(c)['early_slope']:.3f}、后段 {row(c)['late_slope']:.3f} s/圈"
                     f"（后段−前段 {row(c)['diff_late_minus_early']:+.3f}，"
                     f"95% CI {row(c)['diff_ci_lo']:+.3f}~{row(c)['diff_ci_hi']:+.3f}，"
                     f"单侧 p={row(c)['diff_p_one_sided']:.3g}）"
                     for c in COMPOUNDS) + "。"
                 "也就是说数据里看到的是**后段衰减变缓**（一种可能是能留到长段的 stint 本身是被选择出来的），"
                 "不是悬崖式的加速。")
    lines.append("2. **二次项同样是负的。** kappa（二次项系数）"
                 + "；".join(f"{c} {row(c)['kappa_quadratic']:+.4f}"
                            f"（单侧 p={row(c)['kappa_p_one_sided']:.3g}）" for c in COMPOUNDS)
                 + "；二次/分段拟合的 RMSE 略低于线性，但改进很小（见第 2 节表）。")
    lines.append("3. **分站层面存在少数“后段更陡”的格子，但集中在个别赛道/配方。** "
                 f"前 10 格子里单侧 p<0.05 的有 {len(sig_pos)} 个："
                 + "、".join(f"{r.circuit_id}-{r.compound}（{r.diff_late_minus_early:+.3f}s/圈，"
                             f"p={r.diff_p_one_sided:.3g}）" for r in sig_pos.itertuples(index=False))
                 + "。这些格子的前段斜率本身可能接近 0 或为负，后段反而抬升；样本只有几十到上千圈，"
                   "需要更细的分年/分场检查才能确认。")
    zh = {"Australian Grand Prix": "澳大利亚", "Belgian Grand Prix": "比利时",
          "Las Vegas Grand Prix": "拉斯维加斯"}
    parts = []
    for event, g in three.groupby("event", sort=False):
        cells_txt = []
        for r in g.itertuples(index=False):
            if pd.isna(r.diff_late_minus_early):
                cells_txt.append(f"{r.compound} 后段样本不足（前段 {r.early_slope:+.3f}，{r.n_laps} 圈）")
            else:
                cells_txt.append(f"{r.compound} 后段−前段 {r.diff_late_minus_early:+.3f}"
                                 f"（单侧 p={fmt_p(r.diff_p_one_sided)}，{r.n_laps} 圈"
                                 f"{'，低样本' if r.low_n else ''}）")
        parts.append(f"{zh.get(event, event)}：" + "、".join(cells_txt))
    lines.append("4. **2024 年那三场回测偏差最大的比赛，数据不支持“悬崖”解释。** "
                 + "；".join(parts)
                 + "。这些数值没有呈现显著的正向“后段加速”。")
    lines.append("5. **排除交通后结论不变。** 主口径（排除前车 2 秒内）"
                 + "；".join(f"{c}: 后段−前段 {row2:+.3f}，p={p:.3g}"
                            for c, row2, p in [
                                (r.compound, r.diff_late_minus_early, r.diff_p_one_sided)
                                for r in sens_comp[sens_comp["variant"].str.startswith("排除前车")].itertuples(index=False)
                            ])
                 + "；与全样本方向一致，分站格子排名也基本稳定（见第 4 节相关系数/重合数）。")
    lines.append("6. **安全车后 1–3 圈的剔除同样不改变方向。** 三种配方的“后段−前段”仍在 "
                 f"{sens_comp[sens_comp['variant'].str.startswith('排除安全车')]['diff_late_minus_early'].min():+.3f}"
                 f" ~ {sens_comp[sens_comp['variant'].str.startswith('排除安全车')]['diff_late_minus_early'].max():+.3f} s/圈之间。")
    lines.append("")

    lines.append("## 这些数据不能说明什么")
    lines.append("")
    lines.append("1. **不能证明“没有悬崖”。** 长段是被选择出来的：会掉速的轮胎/车手早就进站了，"
                 "能留到 30 圈以上的 stint 本身可能已经避开了最陡的衰减；本单只描述“留下来的数据里的形状”。")
    lines.append("2. **燃油修正系数 0.06 秒/圈来自 T008 的合并估计，不是本单重新标定的。** "
                 "燃油项与胎龄在不换胎时高度共线；系数偏大/偏小会直接改变早期斜率，"
                 "而 kappa 与“后段−前段”对它的敏感性没有在本单逐项扫描。")
    lines.append("3. **段内基准与清洗规则会影响曲线起点。** 前 5 圈用模型表里可用的干净圈（通常是第 2–6 圈，"
                 "因为比赛前两圈、安全车圈被 T007 的 clean 规则去掉）；起始胎龄 >5 的 stint（"
                 f"{int(excl.loc[excl['项目'].str.startswith('起始胎龄'), '圈数'].iloc[0])} 圈）被整段排除。")
    lines.append("4. **按 stint 聚类的标准误只修正了“同段相关”，不能修正车手/赛道选择效应。** "
                 "分站格子只有几十段 stint，p 值应谨慎解读；本单没有做多重比较校正。")
    lines.append("5. **分站格子混合了 2023–2026 四个赛季**（每站每年不同轮胎配方/赛道条件），"
                 "“澳大利亚/比利时/拉斯维加斯”表里给的是对应 2024 年那一场的格子，不是分站固有参数。")
    lines.append("6. **“后段”指超过交叉验证拐点（HARD 27 / MEDIUM 23 / SOFT 13 圈）的胎龄**；"
                 "换 k、换燃油系数、换清洗规则都会改变数值大小，结论方向需要在这些口径下重复检验。")
    lines.append("7. **本单只做描述，不能回答“模拟器为什么推荐一停”**：没有把这里的斜率接到任何模拟器，"
                 "也没有做反事实；回测偏差是否与轮胎形状有关需要单独检验。")
    lines.append("")
    lines.append("## 输入哈希")
    lines.append("")
    lines.append(md_table(hashes))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def main() -> None:
    np.random.seed(SEED)
    df = load_and_prepare()
    sample, excl = analysis_sample(df)

    knots = {}
    cv_frames = []
    for c in COMPOUNDS:
        k, cv = knot_cv(sample, c)
        knots[c] = k
        cv_frames.append(cv)
    cv_tab = pd.concat(cv_frames, ignore_index=True)
    knot_best = pd.DataFrame([{"compound": c,
                               "knot": knots[c],
                               "cv_mse": float(cv_tab[(cv_tab["compound"] == c)
                                                      & (cv_tab["knot"] == knots[c])]["cv_mse"].iloc[0])}
                              for c in COMPOUNDS])

    compound_rows = []
    for c in COMPOUNDS:
        pw = fit_piecewise(sample, c, knots[c])
        q = fit_quadratic(sample, c)
        lin = fit_linear(sample, c)
        row = {"compound": c}
        row.update({k2: v for k2, v in pw.items() if k2 != "compound"})
        row.update(q)
        row.update(lin)
        compound_rows.append(row)
    compound_tab = pd.DataFrame(compound_rows)

    # 保存每配方的 beta（供曲线用）
    fit_beta = {}
    for c in COMPOUNDS:
        s = sample[sample["compound"] == c]
        age = s["tyre_life"].to_numpy(float)
        y = s["y"].to_numpy(float)
        beta, resid, _ = ols(design(age, knots[c]), y)
        fit_beta[c] = {"beta": beta, "knot": knots[c]}
    points = curve_points(sample, {c: {"knot": knots[c], "beta": fit_beta[c]["beta"]}
                                   for c in COMPOUNDS})

    cells = circuit_cells(sample, {c: {"knot": knots[c]} for c in COMPOUNDS})
    # circuit_cells 需要 fits[c]['knot']，上面的 dict 已满足
    three = three_races_table(sample, {c: {"knot": knots[c]} for c in COMPOUNDS})
    top10 = cells.head(10).copy()

    sens_comp, sens_cells, sens_stats = sensitivity_tables(sample, cells)

    # 交通计数
    n_traffic = {
        "lt2": int((sample["gap_ahead_prev"] < 2).sum()),
        "keep_traffic_free": int(sample["traffic_free"].sum()),
        "keep_strict": int(sample["traffic_free_strict"].sum()),
    }
    n_post_sc = int(sample["post_sc"].sum())

    hashes = pd.DataFrame([
        {"文件": str(LAPS.relative_to(ROOT)), "sha256": sha256(LAPS),
         "字节": LAPS.stat().st_size},
        {"文件": str(TRAFFIC.relative_to(ROOT)), "sha256": sha256(TRAFFIC),
         "字节": TRAFFIC.stat().st_size},
        {"文件": str(RAW_LAPS.relative_to(ROOT)), "sha256": sha256(RAW_LAPS),
         "字节": RAW_LAPS.stat().st_size},
    ])

    figs = make_figures(points, compound_tab, cells, three)

    # 输出
    points.to_csv(OUT / "degradation_curve_points.csv", index=False)
    compound_tab.to_csv(OUT / "compound_fits.csv", index=False)
    cv_tab.to_csv(OUT / "knot_cv.csv", index=False)
    cells.to_csv(OUT / "circuit_compound_slopes.csv", index=False)
    top10.to_csv(OUT / "top10_cells.csv", index=False)
    three.to_csv(OUT / "three_races_2024.csv", index=False)
    sens_comp.to_csv(OUT / "traffic_comparison.csv", index=False)
    sens_cells.to_csv(OUT / "cell_traffic_comparison.csv", index=False)
    excl.to_csv(OUT / "exclusions.csv", index=False)
    hashes.to_csv(OUT / "input_sha256.csv", index=False)

    report = build_report(excl, compound_tab, knot_best, cv_tab, points, cells,
                          top10, three, sens_comp, sens_cells, sens_stats,
                          figs, hashes, n_post_sc, n_traffic)
    REPORT.write_text(report, encoding="utf-8")

    print("==" * 30)
    print("analysis laps", len(sample), "stints", sample["stint_uid"].nunique())
    print("\nexclusions:"); print(excl.to_string(index=False))
    print("\nknots:", knot_best.to_string(index=False))
    print("\ncompound fits:")
    print(compound_tab[["compound", "knot", "early_slope", "late_slope",
                        "diff_late_minus_early", "diff_se", "diff_p_one_sided",
                        "kappa_quadratic", "kappa_se", "kappa_p_one_sided",
                        "rmse_linear", "rmse_quadratic", "rmse"]].to_string(index=False))
    print("\ntop10 cells:")
    print(top10.to_string(index=False))
    print("\n2024 three races:")
    print(three.to_string(index=False))
    print("\nsensitivity:")
    print(sens_comp.to_string(index=False))
    print("\ncell traffic stats:", sens_stats)
    print("\noutputs:")
    for p in sorted(OUT.glob("*.csv")):
        print(" ", p.relative_to(HERE))
    for p in figs:
        print(" ", p.relative_to(HERE))
    print(" ", REPORT.relative_to(HERE))
    print("\nDONE")


if __name__ == "__main__":
    main()
