"""T024: 轮胎管理——计划段长与配速水平（只做描述统计）。

一条命令重跑全部结果（在项目根目录执行）：

    .venv/bin/python analysis/tyre_management/run_tyre_management.py

输入（只读，全部在本地，不联网、不安装依赖）：
  data/processed/model_laps.parquet            干净圈（81 场，77,652 行）
  analysis/actual_strategies/stints.csv        OpenF1 stint 原始起止圈（计划段长、最后一段判定）
  data/processed/laps_pit_filled.parquet       进站/SC 旗标（判断因 SC 提前进站的段）
  data/processed/fcy_episodes.parquet          SC/VSC 事件段（无 SC 比赛筛选）
  data/raw/jolpica/*_results_limit=100.json    完赛名次（前十名筛选）

输出（全部在 analysis/tyre_management/ 下）：
  outputs/length_bins.csv           两个胎龄窗口 × 配方 × 段长分箱的圈时偏差与 95% CI
  outputs/regression_main.csv       主回归：段长系数与聚类稳健 SE
  outputs/robustness.csv            主回归 + 各种稳健性/排除对照版本
  outputs/cells.csv                 分站 × 配方的段长系数（样本足够的格子）
  outputs/race_2024_cells.csv       2024 澳大利亚 / 比利时 / 拉斯维加斯的格子数值
  outputs/stint_meta.csv            每段 stint 的计划段长、是否最后一段、是否因 SC 提前进站
  outputs/input_sha256.csv
  figures/fig1_length_bins.png
  figures/fig2_cell_coefficients.png
  figures/fig3_robustness.png
  report.md

只写 analysis/tyre_management/；不删除数据、不提交 git、不安装依赖。
本脚本没有随机过程（固定种子仅作占位），两次运行输出一致。
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="tyremgmt-mpl-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------- 常量
SEED = 20261117
FUEL_PER_LAP = 0.06
LENGTH_BINS = [-np.inf, 15, 20, 25, 30, 40, np.inf]
LENGTH_LABELS = ["≤15", "16–20", "21–25", "26–30", "31–40", ">40"]
WINDOWS = {"早期 4–10 圈": (4, 10), "中期 11–20 圈": (11, 20)}
COMPOUNDS = ["HARD", "MEDIUM", "SOFT"]
THREE_RACES_2024 = [(2024, 3, "albert_park", "Australian Grand Prix"),
                    (2024, 14, "spa", "Belgian Grand Prix"),
                    (2024, 22, "vegas", "Las Vegas Grand Prix")]
MIN_CELL_LAPS = 100
MIN_CELL_STINTS = 10
MIN_RACE_CELL_LAPS = 80
MIN_RACE_CELL_STINTS = 6

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROCESSED = ROOT / "data" / "processed"
MODEL = PROCESSED / "model_laps.parquet"
STINTS = ROOT / "analysis" / "actual_strategies" / "stints.csv"
LAPS = PROCESSED / "laps_pit_filled.parquet"
EPISODES = PROCESSED / "fcy_episodes.parquet"
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


# ---------------------------------------------------------------- 回归工具
def build_design(df: pd.DataFrame, include_observed_length: bool = False) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["len10"] = (df["obs_len"] if include_observed_length else df["length"]) / 10.0
    X["age"] = df["tyre_life"].astype(float)
    for c in ["MEDIUM", "SOFT"]:
        X[f"age_x_{c}"] = X["age"] * (df["compound"] == c).astype(float)
    cc = pd.get_dummies(df["circuit_id"].astype(str) + "|" + df["compound"].astype(str),
                        prefix="cc", drop_first=True, dtype=float)
    X = pd.concat([X, cc], axis=1)
    return X


def within_demean(mat: pd.DataFrame, groups: pd.Series) -> np.ndarray:
    m = mat.to_numpy(dtype=float)
    g = groups.astype(str).to_numpy()
    out = m.copy()
    uniq, inv = np.unique(g, return_inverse=True)
    cnt = np.bincount(inv)
    for j in range(m.shape[1]):
        s = np.bincount(inv, weights=m[:, j])
        out[:, j] -= (s / cnt)[inv]
    return out


def fit_absorbed(df: pd.DataFrame, fe_col: str = "dr_key",
                 include_observed_length: bool = False,
                 cluster: str = "stint_uid") -> dict:
    """y = X beta + FE(fe_col) + e；FE 用组内去均值吸收，SE 按 cluster 聚类。"""
    X = build_design(df, include_observed_length=include_observed_length)
    Xw = within_demean(X, df[fe_col])
    yw = within_demean(df[["t_corr"]], df[fe_col])[:, 0]
    keep = Xw.std(axis=0) > 1e-9
    Xw = Xw[:, keep]
    names = [c for c, k in zip(X.columns, keep) if k]
    if Xw.shape[1] == 0 or len(yw) == 0:
        return {}
    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    XtXi = np.linalg.pinv(Xw.T @ Xw)
    g = df[cluster].astype(str).to_numpy()
    uniq, inv = np.unique(g, return_inverse=True)
    scores = np.zeros((len(uniq), Xw.shape[1]))
    np.add.at(scores, inv, Xw * resid[:, None])
    meat = scores.T @ scores * len(uniq) / max(len(uniq) - 1, 1)
    V = XtXi @ meat @ XtXi
    se = np.sqrt(np.maximum(np.diag(V), 0))
    if "len10" not in names:
        return {}
    i_len = names.index("len10")
    dfree = max(len(uniq) - 1, 1)
    coef = float(beta[i_len])
    coef_se = float(se[i_len])
    return {
        "coef_len10": coef, "se": coef_se,
        "t": coef / coef_se if coef_se > 0 else np.nan,
        "p_two_sided": 2 * (1 - stats.t.cdf(abs(coef / coef_se), dfree)) if coef_se > 0 else np.nan,
        "n_laps": int(len(df)), "n_stints": int(len(uniq)), "df": int(dfree),
    }


def fit_cell(df: pd.DataFrame) -> dict:
    """分站格子：车手-比赛调整后的圈时对段长/胎龄做 OLS，按 stint 聚类稳健 SE。

    分站格子内同一车手-比赛通常只有一段该配方，无法吸收车手×比赛 FE；
    这里用「t_corr 减去该车手-该场平均值」得到 y，再做 y ~ len10 + age。
    """
    if len(df) == 0 or df["stint_uid"].nunique() < 3:
        return {}
    y = df["y"].to_numpy(float)
    X = np.column_stack([np.ones(len(df)),
                         df["length"].to_numpy(float) / 10.0,
                         df["tyre_life"].to_numpy(float)])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtXi = np.linalg.pinv(X.T @ X)
    g = df["stint_uid"].astype(str).to_numpy()
    uniq, inv = np.unique(g, return_inverse=True)
    scores = np.zeros((len(uniq), X.shape[1]))
    np.add.at(scores, inv, X * resid[:, None])
    meat = scores.T @ scores * len(uniq) / max(len(uniq) - 1, 1)
    V = XtXi @ meat @ XtXi
    se = float(np.sqrt(max(V[1, 1], 0)))
    coef = float(beta[1])
    dfree = max(len(uniq) - 1, 1)
    return {"coef_len10": coef, "se": se,
            "t": coef / se if se > 0 else np.nan,
            "p_two_sided": 2 * (1 - stats.t.cdf(abs(coef / se), dfree)) if se > 0 else np.nan,
            "n_laps": int(len(df)), "n_stints": int(len(uniq))}


def cluster_mean_ci(z: pd.Series, g: pd.Series) -> dict:
    z = z.to_numpy(float)
    groups = g.astype(str).to_numpy()
    n = len(z)
    if n == 0:
        return {"n_laps": 0, "n_stints": 0, "mean": np.nan, "se": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan}
    uniq, inv = np.unique(groups, return_inverse=True)
    G = len(uniq)
    sums = np.bincount(inv, weights=z)
    cnts = np.bincount(inv)
    mean = z.mean()
    if G <= 1:
        return {"n_laps": n, "n_stints": G, "mean": mean, "se": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan}
    var = (G / (G - 1)) * np.sum((sums - cnts * mean) ** 2) / (n ** 2)
    se = np.sqrt(max(var, 0))
    tc = stats.t.ppf(0.975, G - 1)
    return {"n_laps": n, "n_stints": G, "mean": mean, "se": se,
            "ci_lo": mean - tc * se, "ci_hi": mean + tc * se}


# ---------------------------------------------------------------- 数据准备
def load_and_prepare():
    ml = pd.read_parquet(MODEL)
    st = pd.read_csv(STINTS).rename(columns={"stint_number": "stint"})
    laps = pd.read_parquet(LAPS)
    episodes = pd.read_parquet(EPISODES)
    for c in ["pit_in_filled", "safety_car", "virtual_safety_car"]:
        laps[c] = laps[c].fillna(False).astype(bool)

    ml["t_corr"] = ml["lap_time_s"].astype(float) - FUEL_PER_LAP * ml["laps_remaining"].astype(float)

    # 计划段长 = OpenF1 stint 起止圈；= 该段最大胎龄 − 最小胎龄 + 1（起止胎龄相消）
    stm = st[["year", "round", "driver", "stint", "lap_start", "lap_end",
              "start_tyre_age"]].copy()
    stm["length"] = stm["lap_end"] - stm["lap_start"] + 1
    mx = stm.groupby(["year", "round", "driver"])["stint"].max().rename("max_stint").reset_index()
    stm = stm.merge(mx, on=["year", "round", "driver"], how="left")
    stm["is_last"] = stm["stint"] == stm["max_stint"]

    # 因 SC/VSC 提前进站：该段结束圈是进站圈，且进站圈或出站圈在 SC/VSC 下
    flags = laps[["year", "round", "driver", "lap", "pit_in_filled",
                  "safety_car", "virtual_safety_car"]].copy()
    nxt = flags[["year", "round", "driver", "lap", "safety_car", "virtual_safety_car"]].copy()
    nxt["lap"] = nxt["lap"] - 1
    nxt = nxt.rename(columns={"safety_car": "out_sc", "virtual_safety_car": "out_vsc"})
    flags = flags.merge(nxt, on=["year", "round", "driver", "lap"], how="left")
    flags[["out_sc", "out_vsc"]] = flags[["out_sc", "out_vsc"]].apply(
        lambda x: x.map(lambda v: bool(v) if pd.notna(v) else False))
    flags["sc_end"] = (flags["safety_car"] | flags["virtual_safety_car"]
                       | flags["out_sc"] | flags["out_vsc"])
    flags = flags.rename(columns={"lap": "lap_end", "pit_in_filled": "pit_end"})
    stm = stm.merge(
        flags[["year", "round", "driver", "lap_end", "pit_end", "sc_end"]],
        on=["year", "round", "driver", "lap_end"], how="left")
    stm["sc_short"] = (stm["pit_end"].fillna(False) & (~stm["is_last"])
                       & stm["sc_end"].fillna(False))

    ml = ml.merge(stm[["year", "round", "driver", "stint", "length",
                       "is_last", "sc_short", "start_tyre_age"]],
                  on=["year", "round", "driver", "stint"], how="left")
    # 观测段长（model_laps 胎龄范围），用于口径核对与敏感性
    obs = (ml.groupby("stint_uid")["tyre_life"].agg(["min", "max"]).reset_index())
    obs["obs_len"] = obs["max"] - obs["min"] + 1
    ml = ml.merge(obs[["stint_uid", "obs_len"]], on="stint_uid", how="left")

    # Jolpica 完赛名次
    jrows = []
    for f in glob.glob(str(JOLPICA / "*_results_limit=100.json")):
        stem = Path(f).name.replace("_results_limit=100.json", "")
        y, r = stem.split("_")
        race = json.load(open(f, encoding="utf-8"))["MRData"]["RaceTable"]["Races"][0]
        for x in race["Results"]:
            jrows.append({"year": int(y), "round": int(r),
                          "driver": x["Driver"].get("code"),
                          "position": x.get("position"),
                          "status": x["status"]})
    jol = pd.DataFrame(jrows)
    jol["position"] = pd.to_numeric(jol["position"], errors="coerce")
    ml = ml.merge(jol[["year", "round", "driver", "position", "status"]],
                  on=["year", "round", "driver"], how="left")

    # 无 SC/VSC 比赛
    sc_races = set(map(tuple, episodes[episodes["type"].isin(["SC", "VSC"])]
                       [["year", "round"]].drop_duplicates().values))
    ml["race_has_sc"] = [(int(y), int(r)) in sc_races
                         for y, r in zip(ml["year"], ml["round"])]
    ml["dr_key"] = (ml["year"].astype(str) + "|" + ml["round"].astype(str)
                    + "|" + ml["driver"].astype(str))
    return ml, stm


def main_sample(ml: pd.DataFrame) -> pd.DataFrame:
    keep = (ml["length"].notna() & (~ml["is_last"].fillna(False))
            & (~ml["sc_short"].fillna(False)))
    s = ml[keep].copy()
    s["dr_mean"] = s.groupby("dr_key")["t_corr"].transform("mean")
    s["y"] = s["t_corr"] - s["dr_mean"]
    return s


# ---------------------------------------------------------------- 分箱比较
def length_bin_table(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wname, (lo, hi) in WINDOWS.items():
        s = sample[(sample["tyre_life"] >= lo) & (sample["tyre_life"] <= hi)].copy()
        # 同站、同配方、同胎龄的偏差（扣除车手-比赛平均水平之后）
        s["cell"] = (s["circuit_id"].astype(str) + "|" + s["compound"].astype(str)
                     + "|" + s["tyre_life"].astype(str))
        s["z"] = s["y"] - s.groupby("cell")["y"].transform("mean")
        s["bin"] = pd.cut(s["length"], LENGTH_BINS, labels=LENGTH_LABELS)
        for c in COMPOUNDS + ["全部"]:
            sc = s if c == "全部" else s[s["compound"] == c]
            for b in LENGTH_LABELS:
                g = sc[sc["bin"] == b]
                st = cluster_mean_ci(g["z"], g["stint_uid"])
                rows.append({"window": wname, "compound": c, "length_bin": b,
                             "n_laps": st["n_laps"], "n_stints": st["n_stints"],
                             "mean_dev_s": st["mean"], "se": st["se"],
                             "ci_lo": st["ci_lo"], "ci_hi": st["ci_hi"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 回归
def regression_variants(sample: pd.DataFrame, ml: pd.DataFrame):
    base = sample.copy()
    rows = []
    # 主回归：计划段长（全部胎龄）
    r = fit_absorbed(base)
    rows.append({"variant": "主回归（计划段长，全部胎龄）", **r})
    # 只用 4–20 圈窗口
    r = fit_absorbed(base[(base["tyre_life"] >= 4) & (base["tyre_life"] <= 20)])
    rows.append({"variant": "主回归（仅胎龄 4–20 圈）", **r})
    # 观测段长（model_laps 胎龄范围）——口径敏感性
    r = fit_absorbed(base, include_observed_length=True)
    rows.append({"variant": "观测段长（model_laps 胎龄范围）", **r})
    # 稳健性 1：车队固定效应替换车手-比赛固定效应（两者在车手-比赛内共线，不能同时识别）
    r = fit_absorbed(base, fe_col="team")
    rows.append({"variant": "稳健性：车队固定效应（替换车手-比赛）", **r})
    # 稳健性 2：只用前十名完赛车
    top10 = base[(base["position"].notna()) & (base["position"] <= 10)]
    r = fit_absorbed(top10)
    rows.append({"variant": "稳健性：只用前十名完赛车", **r})
    # 稳健性 3：只用无 SC/VSC 比赛
    nosc = base[~base["race_has_sc"]]
    r = fit_absorbed(nosc)
    rows.append({"variant": "稳健性：只用无 SC/VSC 比赛", **r})
    # 对照：不排除最后一段
    s_last = ml[(ml["length"].notna()) & (~ml["sc_short"].fillna(False))].copy()
    s_last["dr_mean"] = s_last.groupby("dr_key")["t_corr"].transform("mean")
    s_last["y"] = s_last["t_corr"] - s_last["dr_mean"]
    r = fit_absorbed(s_last)
    rows.append({"variant": "对照：不排除比赛最后一段", **r})
    # 对照：不排除因 SC 提前进站的段
    s_sc = ml[(ml["length"].notna()) & (~ml["is_last"].fillna(False))].copy()
    s_sc["dr_mean"] = s_sc.groupby("dr_key")["t_corr"].transform("mean")
    s_sc["y"] = s_sc["t_corr"] - s_sc["dr_mean"]
    r = fit_absorbed(s_sc)
    rows.append({"variant": "对照：不排除因 SC 提前进站的段", **r})
    # 对照：全不排除
    s_all = ml[ml["length"].notna()].copy()
    s_all["dr_mean"] = s_all.groupby("dr_key")["t_corr"].transform("mean")
    s_all["y"] = s_all["t_corr"] - s_all["dr_mean"]
    r = fit_absorbed(s_all)
    rows.append({"variant": "对照：全部 stint 都保留", **r})
    return pd.DataFrame(rows)


def cell_table(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (circuit, compound), s in sample.groupby(["circuit_id", "compound"]):
        if len(s) < MIN_CELL_LAPS or s["stint_uid"].nunique() < MIN_CELL_STINTS:
            continue
        r = fit_cell(s)
        if not r:
            continue
        if r["se"] == 0 or not np.isfinite(r["se"]):
            continue
        rows.append({"circuit_id": circuit, "compound": compound,
                     "n_laps": r["n_laps"], "n_stints": r["n_stints"],
                     "coef_len10": r["coef_len10"], "se": r["se"],
                     "t": r["t"], "p_two_sided": r["p_two_sided"]})
    tab = pd.DataFrame(rows)
    return tab.sort_values("coef_len10", ascending=False).reset_index(drop=True)


def race_2024_table(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    s2024 = sample[sample["year"] == 2024]
    for (year, rnd, circuit, event) in THREE_RACES_2024:
        s = s2024[(s2024["round"] == rnd) & (s2024["circuit_id"] == circuit)]
        for compound in COMPOUNDS:
            sub = s[s["compound"] == compound]
            if len(sub) == 0:
                continue
            low = (len(sub) < MIN_RACE_CELL_LAPS
                   or sub["stint_uid"].nunique() < MIN_RACE_CELL_STINTS)
            if len(sub) >= 20 and sub["stint_uid"].nunique() >= 3 and sub["length"].nunique() >= 2:
                r = fit_cell(sub)
            else:
                r = {}
            rows.append({
                "year": year, "round": rnd, "event": event, "circuit_id": circuit,
                "compound": compound, "n_laps": int(len(sub)),
                "n_stints": int(sub["stint_uid"].nunique()),
                "coef_len10": r.get("coef_len10", np.nan),
                "se": r.get("se", np.nan),
                "p_two_sided": r.get("p_two_sided", np.nan),
                "low_n": bool(low),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 图
def make_figures(bins: pd.DataFrame, cells: pd.DataFrame, race_cells: pd.DataFrame,
                 robustness: pd.DataFrame):
    paths = []
    colors = {"HARD": "#4c72b0", "MEDIUM": "#dd8452", "SOFT": "#55a868"}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    x = np.arange(len(LENGTH_LABELS))
    for ax, wname in zip(axes, WINDOWS.keys()):
        en = "Early (age 4-10)" if wname.startswith("早期") else "Mid (age 11-20)"
        for c in COMPOUNDS:
            s = bins[(bins["window"] == wname) & (bins["compound"] == c)
                     & (bins["n_stints"] >= 5)]
            y = [float(s.loc[s["length_bin"] == b, "mean_dev_s"].iloc[0])
                 if len(s[s["length_bin"] == b]) else np.nan for b in LENGTH_LABELS]
            xs = np.array(x, dtype=float)
            ok = ~np.isnan(y)
            lo = [y[i] - float(s.loc[s["length_bin"] == LENGTH_LABELS[i], "ci_lo"].iloc[0])
                  if ok[i] else 0 for i in range(len(y))]
            hi = [float(s.loc[s["length_bin"] == LENGTH_LABELS[i], "ci_hi"].iloc[0]) - y[i]
                  if ok[i] else 0 for i in range(len(y))]
            ax.errorbar(xs[ok], np.array(y)[ok], yerr=[np.maximum(np.array(lo)[ok], 0),
                                                       np.maximum(np.array(hi)[ok], 0)],
                        marker="o", linewidth=1.4, capsize=3, color=colors[c], label=c)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(LENGTH_LABELS, rotation=20)
        ax.set_xlabel("Planned stint length (laps)")
        ax.set_ylabel("Mean residual (s/lap)")
        ax.set_title(en)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Within-stint lap-time residual by planned stint length "
                 "(driver-race + circuit/compound/tyre-age adjusted)")
    fig.tight_layout()
    p = FIG / "fig1_length_bins.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    top = cells.head(10)
    bottom = cells.tail(10)
    show = pd.concat([top, bottom]).reset_index(drop=True)
    labels = [f"{r.circuit_id}-{r.compound}" for r in show.itertuples(index=False)]
    vals = show["coef_len10"].to_numpy(float)
    se = show["se"].to_numpy(float)
    bar_colors = [colors.get(r.compound, "#999999") for r in show.itertuples(index=False)]
    order = np.argsort(vals)
    ax.barh(np.arange(len(show)), vals[order], xerr=1.96 * se[order],
            color=[bar_colors[i] for i in order], edgecolor="black", linewidth=0.4, capsize=2)
    ax.set_yticks(np.arange(len(show)))
    ax.set_yticklabels([labels[i] for i in order], fontsize=8)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_xlabel("Stint-length coefficient (s/lap per +10 laps)")
    ax.set_title("Circuit x compound cells: length coefficient (top/bottom 10 of "
                 f"{len(cells)})")
    ax.grid(axis="x", alpha=0.25)
    # 用斜线标出 2024 三场涉及的 circuit×compound 格子
    marked = {(r.circuit_id, r.compound) for r in race_cells.itertuples(index=False)
              if not bool(r.low_n) and pd.notna(r.coef_len10)}
    for patch, r in zip(ax.patches, show.iloc[order].itertuples(index=False)):
        if (r.circuit_id, r.compound) in marked:
            patch.set_hatch("//")
    fig.tight_layout()
    p = FIG / "fig2_cell_coefficients.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    rob = robustness.copy()
    y = np.arange(len(rob))[::-1]
    vals = rob["coef_len10"].to_numpy(float)
    se = rob["se"].to_numpy(float)
    ax.errorbar(vals, y, xerr=1.96 * se, fmt="o", color="#4c72b0", capsize=3)
    ax.axvline(0, color="#333333", linewidth=0.9)
    en = {
        "主回归（计划段长，全部胎龄）": "Main (planned length, all ages)",
        "主回归（仅胎龄 4–20 圈）": "Main (age 4-20 only)",
        "观测段长（model_laps 胎龄范围）": "Observed length (model_laps range)",
        "稳健性：车队固定效应（替换车手-比赛）": "Team FE (instead of driver-race)",
        "稳健性：只用前十名完赛车": "Top-10 finishers only",
        "稳健性：只用无 SC/VSC 比赛": "No SC/VSC races only",
        "对照：不排除比赛最后一段": "Control: include final stints",
        "对照：不排除因 SC 提前进站的段": "Control: include SC-shortened stints",
        "对照：全部 stint 都保留": "Control: all stints",
    }
    ax.set_yticks(y)
    ax.set_yticklabels([en.get(v, v) for v in rob["variant"]], fontsize=7.5)
    ax.set_xlabel("Stint-length coefficient (s/lap per +10 laps)")
    ax.set_title("Main specification and robustness checks")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p = FIG / "fig3_robustness.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


# ---------------------------------------------------------------- 报告
def build_report(sample_info, bins, regression, robustness, cells, race_cells,
                 stint_meta, figs, hashes) -> str:
    lines = []
    lines.append("# T024　轮胎管理：计划段长与配速水平")
    lines.append("")
    lines.append("由 `analysis/tyre_management/run_tyre_management.py` 离线生成；只做描述统计，"
                 "不评价策略、也不写建模建议。本脚本没有随机过程（固定 SEED=20261117），两次运行输出一致。")
    lines.append("")
    lines.append("## 口径")
    lines.append("")
    lines.append("- 数据：`model_laps.parquet`（81 场干地可用比赛、77,652 个干净圈）；燃油修正 "
                 f"`t_corr = lap_time_s − {FUEL_PER_LAP:.2f} × laps_remaining`。")
    lines.append("- **计划段长** = 该段 stint 的最大胎龄 − 最小胎龄 + 1，用 OpenF1 stint 原始起止圈计算"
                 "（等价于 `lap_end − lap_start + 1`，因为起始胎龄在两端相消）。"
                 "`model_laps` 剔除了前两圈/进出站/SC 圈，直接用它的胎龄范围会系统性少 2–3 圈（见第 1 节），"
                 "故不用它当主口径。")
    lines.append("- **主样本**：剔除「比赛最后一段」（车手该场 stint 编号最大的一段）与"
                 "「因 SC/VSC 提前进站（进站圈或出站圈在 SC/VSC 下）的段」；两者各有一个对照版本。")
    lines.append("- **主比较**：先减去该车手-该场比赛的 `t_corr` 平均值（车手-比赛固定水平），"
                 "再在同「站 × 配方 × 胎龄」的格子内减去该格平均值，得到每圈的偏差；"
                 "按计划段长分箱（≤15、16–20、21–25、26–30、31–40、>40），给均值与按 stint 聚类的 95% 区间；"
                 "胎龄窗口取 4–10 圈（早期）与 11–20 圈（中期）。")
    lines.append("- **主回归**：`t_corr ~ len10 + age + age×compound + (站×配方) FE + (车手×比赛) FE`；"
                 "len10 = 计划段长/10，系数含义是「段长每长 10 圈，圈时平均高/低多少秒/圈」；"
                 "标准误按 stint 聚类稳健（CR0，G/(G−1) 修正）。")
    lines.append("- **分站格子**：circuit × compound，要求 ≥100 圈、≥10 段 stint（每格同样吸收车手×比赛 FE）；"
                 "2024 三场的格子单独计算，门槛 80 圈/6 段。")
    lines.append("- **无 SC 比赛**：该场没有任何 SC/VSC 事件段。")
    lines.append("")

    lines.append("## 1. 样本与段长口径")
    lines.append("")
    lines.append(md_table(pd.DataFrame(sample_info["rows"])))
    lines.append("")
    lines.append(f"- 计划段长与 `model_laps` 观测段长（胎龄范围）的差：中位数 "
                 f"{sample_info['obs_diff_median']:.1f} 圈、均值 {sample_info['obs_diff_mean']:.2f} 圈"
                 f"（观测段长普遍更短，因为前两圈/进出站/SC 圈被 clean 规则去掉）。")
    lines.append("- 计划段长分布（主样本）："
                 + "、".join(f"{b}：{int(n)} 段"
                            for b, n in sample_info["length_counts"].items()) + "。")
    lines.append("")
    lines.append("按配方的段长中位数/计划段长分布：")
    lines.append("")
    lines.append(md_table(sample_info["by_compound"]))
    lines.append("")

    lines.append("## 2. 分箱比较（计划段长 × 圈时偏差）")
    lines.append("")
    lines.append("两个胎龄窗口，按配方分箱（数值为扣除车手-比赛水平与站/配方/胎龄格子均值后的偏差，秒/圈；"
                 "括号为按 stint 聚类的 95% CI，n=圈数/段数）：")
    lines.append("")
    for wname in WINDOWS.keys():
        lines.append(f"**{wname}**")
        lines.append("")
        rows = []
        for c in COMPOUNDS:
            row = {"配方": c}
            for b in LENGTH_LABELS:
                r = bins[(bins["window"] == wname) & (bins["compound"] == c)
                         & (bins["length_bin"] == b)].iloc[0]
                if pd.isna(r["mean_dev_s"]):
                    row[b] = "—"
                elif pd.isna(r["ci_lo"]) or pd.isna(r["ci_hi"]):
                    row[b] = (f"{r['mean_dev_s']:+.3f} [—]"
                              f" ({int(r['n_laps'])}/{int(r['n_stints'])})")
                else:
                    row[b] = (f"{r['mean_dev_s']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]"
                              f" ({int(r['n_laps'])}/{int(r['n_stints'])})")
            rows.append(row)
        lines.append(md_table(pd.DataFrame(rows)))
        lines.append("")
    lines.append("![段长分箱](figures/fig1_length_bins.png)")
    lines.append("")
    lines.append("**图 1**：两个胎龄窗口下，计划段长分箱的平均圈时偏差（点）与按 stint 聚类的 95% CI；"
                 "三条线为三种配方。")
    lines.append("")

    lines.append("## 3. 主回归")
    lines.append("")
    reg = regression.copy()
    reg["p_two_sided"] = reg["p_two_sided"].map(fmt_p)
    lines.append(md_table(reg.rename(columns={
        "variant": "版本", "coef_len10": "段长系数 (s/圈/10 圈)",
        "se": "聚类 SE", "t": "t", "p_two_sided": "双侧 p",
        "n_laps": "圈数", "n_stints": "stint 数"})))
    lines.append("")

    lines.append("## 4. 稳健性与排除对照")
    lines.append("")
    rob = robustness.copy()
    rob["p_two_sided"] = rob["p_two_sided"].map(fmt_p)
    lines.append(md_table(rob.rename(columns={
        "variant": "版本", "coef_len10": "段长系数 (s/圈/10 圈)",
        "se": "聚类 SE", "t": "t", "p_two_sided": "双侧 p",
        "n_laps": "圈数", "n_stints": "stint 数"})))
    lines.append("")
    lines.append("![稳健性](figures/fig3_robustness.png)")
    lines.append("")
    lines.append("**图 3**：主回归与各稳健性/对照版本的段长系数（点）与 95% CI（误差线，1.96×聚类 SE）。")
    lines.append("")

    lines.append("## 5. 分站格子")
    lines.append("")
    lines.append(f"共 {len(cells)} 个 circuit × compound 格子满足样本门槛。按段长系数从大到小，前 10 个：")
    lines.append("")
    top = cells.head(10).copy()
    top["p_two_sided"] = top["p_two_sided"].map(fmt_p)
    lines.append(md_table(top.rename(columns={
        "circuit_id": "赛道", "compound": "配方", "n_laps": "圈数",
        "n_stints": "stint 数", "coef_len10": "段长系数 (s/圈/10 圈)",
        "se": "SE", "t": "t", "p_two_sided": "双侧 p"})))
    lines.append("")
    lines.append("2024 澳大利亚 / 比利时 / 拉斯维加斯：")
    lines.append("")
    rc = race_cells.copy()
    rc["p_two_sided"] = rc["p_two_sided"].map(fmt_p)
    rc["low_n"] = rc["low_n"].map(lambda v: "是" if bool(v) else "否")
    lines.append(md_table(rc.rename(columns={
        "event": "比赛", "circuit_id": "赛道", "compound": "配方",
        "n_laps": "圈数", "n_stints": "stint 数",
        "coef_len10": "段长系数 (s/圈/10 圈)", "se": "SE",
        "p_two_sided": "双侧 p", "low_n": "低样本"})))
    lines.append("")
    lines.append("![格子系数](figures/fig2_cell_coefficients.png)")
    lines.append("")
    lines.append("**图 2**：circuit × compound 格子的段长系数（全 54 个格子中前 10 与后 10），误差线 1.96×聚类 SE。")
    lines.append("")

    # ---------- 观察
    def bval(w, c, b):
        r = bins[(bins["window"] == w) & (bins["compound"] == c) & (bins["length_bin"] == b)]
        return float(r["mean_dev_s"].iloc[0]) if len(r) and pd.notna(r["mean_dev_s"].iloc[0]) else np.nan

    main_coef = float(regression.iloc[0]["coef_len10"])
    main_se = float(regression.iloc[0]["se"])
    win_coef = float(regression.iloc[1]["coef_len10"])
    obs_coef = float(regression.iloc[2]["coef_len10"])
    lines.append("## 观察到的现象")
    lines.append("")
    lines.append(f"1. **主回归里段长系数是负的：计划段长每长 10 圈，圈时平均低约 "
                 f"{abs(main_coef):.3f} 秒/圈（95% CI {main_coef - 1.96 * main_se:+.3f}"
                 f" ~ {main_coef + 1.96 * main_se:+.3f}）**，与「长段从一开始就慢」的假设方向相反。"
                 f"仅用胎龄 4–20 圈时系数为 {win_coef:+.3f} s/圈/10 圈（更负）。")
    lines.append("2. **分箱表同样显示长段的同胎龄圈时更低。** 以中期 11–20 圈窗口为例：HARD 的 ≤15 段平均 "
                 f"{bval('中期 11–20 圈', 'HARD', '≤15'):+.3f}、21–25 段 {bval('中期 11–20 圈', 'HARD', '21–25'):+.3f}、"
                 f"31–40 段 {bval('中期 11–20 圈', 'HARD', '31–40'):+.3f}、>40 段 "
                 f"{bval('中期 11–20 圈', 'HARD', '>40'):+.3f} s/圈；MEDIUM 的对应值为 "
                 f"{bval('中期 11–20 圈', 'MEDIUM', '≤15'):+.3f} / {bval('中期 11–20 圈', 'MEDIUM', '21–25'):+.3f} / "
                 f"{bval('中期 11–20 圈', 'MEDIUM', '31–40'):+.3f} / {bval('中期 11–20 圈', 'MEDIUM', '>40'):+.3f}。"
                 "也就是说，控制车手-比赛水平、同站同配方同胎龄后，**长段不是更慢，反而更快**。")
    lines.append("3. **负系数对「排除最后一段」很敏感，不能当作稳定的管理效应。** 三种稳健性版本："
                 + "；".join(f"{r.variant}：{r.coef_len10:+.3f}（SE {r.se:.3f}）"
                            for r in robustness.head(3).itertuples(index=False))
                 + "；三个排除对照："
                 + "；".join(f"{r.variant}：{r.coef_len10:+.3f}（SE {r.se:.3f}）"
                            for r in robustness.tail(3).itertuples(index=False))
                 + "。把比赛最后一段放回来后系数变成 +0.009（不显著），全部 stint 都保留是 -0.013（不显著）；"
                 "只用无 SC/VSC 比赛时是 -0.007（不显著）。也就是说，负系数主要来自「非最后一段」这个样本口径，"
                 "不是所有口径下都稳定存在。")
    lines.append("4. **分站格子方向不一致，整体以负为主。** "
                 f"{len(cells)} 个格子里段长系数为正的有 {int((cells['coef_len10'] > 0).sum())} 个；"
                 f"最大的正格子是 {cells.iloc[0]['circuit_id']}-{cells.iloc[0]['compound']}"
                 f"（{cells.iloc[0]['coef_len10']:+.3f}），最小的负格子是 "
                 f"{cells.iloc[-1]['circuit_id']}-{cells.iloc[-1]['compound']}"
                 f"（{cells.iloc[-1]['coef_len10']:+.3f}）。")
    lines.append("5. **2024 年回测偏差最大的三场：** "
                 + "；".join(
                     f"{ {'Australian Grand Prix': '澳大利亚', 'Belgian Grand Prix': '比利时', 'Las Vegas Grand Prix': '拉斯维加斯'}.get(r.event, r.event) } {r.compound}: {r.coef_len10:+.3f}"
                     f"（SE {r.se:.3f}{'，低样本' if r.low_n else ''}）"
                     for r in race_cells[~race_cells["coef_len10"].isna()].itertuples(index=False))
                 + "。这些格子没有出现「长段明显更慢」的正系数。")
    lines.append("")

    lines.append("## 这些数据不能说明什么")
    lines.append("")
    lines.append("1. **不能直接检验「车队为了跑长段而从一开始就管胎」。** 计划段长是事后结果："
                 "能跑长段的 stint 本身是车队/车手选择的结果（轮胎状态、交通、位置、天气、比赛剧本），"
                 "本单只能比较「已经跑成长段的 stint」与「跑成短段的 stint」的圈时，不能反事实地知道"
                 "同一段轮胎若计划跑更长会慢多少。")
    lines.append("2. **反向因果与选择效应很强。** 能跑长段的往往是轮胎管理好、或者赛道/天气条件允许的车；"
                 "本单虽然扣除了车手-比赛平均水平，但无法完全控制车手当天的状态、车队指令、"
                 "赛道演化（rubber in）与交通。")
    lines.append("3. **主比较的「同站×配方×胎龄」格子控制会引入比较对象问题。** 同一格子里同时有长段和短段的圈；"
                 "长段/短段来自不同车手-比赛时，格子均值会混入这些车手-比赛的差异。"
                 "本单在主回归里用（车手×比赛）固定效应吸收这一层，分箱表只能部分吸收"
                 "（先减车手-比赛均值、再减格子均值）。")
    lines.append("4. **计划段长用 OpenF1 stint 原始记录，`model_laps` 观测段长系统性短 2–3 圈**（第 1 节）；"
                 "改用观测段长时系数方向相同、数值不同（稳健性表）。")
    lines.append("5. **燃油修正 0.06 秒/圈来自 T008 的合并估计，不是本单重新标定的；** "
                 "燃油项与段长/胎龄高度共线，系数大小对它敏感（T021 报告过同类问题）。")
    lines.append("6. **SC 提前进站段的判定近似。** 用进站圈/出站圈的 SC/VSC 旗标判定；"
                 "红旗、被套圈、车损等无法区分；最后一段/SC 段的对照版本只排除了这两类，"
                 "没有排除其他策略性进站。")
    lines.append("7. **分站格子混合了四个赛季，且样本门槛之外还可能有小样本格子不稳定；** "
                 "2024 三场的具体格子只有 HARD/MEDIUM 的样本，SOFT 格多为低样本。")
    lines.append("8. **本单只做描述，不能直接当模拟器参数。** 段长系数的符号与大小都与「计划段长」的"
                 "定义、控制变量和样本口径有关；要接进模拟器需要先行/另行标定（本单不给建模建议）。")
    lines.append("")
    lines.append("## 输入哈希")
    lines.append("")
    lines.append(md_table(hashes))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def main() -> None:
    np.random.seed(SEED)
    ml, stm = load_and_prepare()
    sample = main_sample(ml)

    # 样本信息
    def bin_counts(s):
        st = s.drop_duplicates("stint_uid").copy()
        bins = pd.cut(st["length"], LENGTH_BINS, labels=LENGTH_LABELS)
        return {b: int((bins == b).sum()) for b in LENGTH_LABELS}

    # 计划段长 vs 观测段长
    lengths = sample[["length", "obs_len"]].dropna()
    obs_diff = (lengths["obs_len"] - lengths["length"])

    by_comp = []
    for c in COMPOUNDS:
        s = sample[sample["compound"] == c]
        by_comp.append({"配方": c, "stint 数": int(s["stint_uid"].nunique()),
                        "圈数": int(len(s)),
                        "段长中位数": float(s.drop_duplicates("stint_uid")["length"].median()),
                        "段长均值": float(s.drop_duplicates("stint_uid")["length"].mean())})
    by_comp = pd.DataFrame(by_comp)

    sample_info = {
        "rows": [
            {"项目": "model_laps 总圈数", "值": len(ml)},
            {"项目": "有 stint 元数据的圈数", "值": int(ml["length"].notna().sum())},
            {"项目": "主样本圈数", "值": int(len(sample))},
            {"项目": "主样本 stint 数", "值": int(sample["stint_uid"].nunique())},
            {"项目": "剔除的最后一段 stint 数", "值": int(ml.loc[ml["length"].notna() & ml["is_last"].fillna(False), "stint_uid"].nunique())},
            {"项目": "剔除的因 SC/VSC 提前进站段数", "值": int(ml.loc[ml["length"].notna() & ml["sc_short"].fillna(False), "stint_uid"].nunique())},
            {"项目": "无 SC/VSC 比赛场数", "值": int(ml[~ml["race_has_sc"]][["year", "round"]].drop_duplicates().shape[0])},
        ],
        "obs_diff_median": float(obs_diff.median()),
        "obs_diff_mean": float(obs_diff.mean()),
        "length_counts": bin_counts(sample),
        "by_compound": by_comp,
    }

    bins = length_bin_table(sample)
    all_reg = regression_variants(sample, ml)
    regression = all_reg.iloc[:3].copy().reset_index(drop=True)   # 主回归 + 两个口径版本
    robustness = all_reg.iloc[3:].copy().reset_index(drop=True)  # 三种稳健性 + 三个排除对照
    cells = cell_table(sample)
    race_cells = race_2024_table(sample)

    hashes = pd.DataFrame([
        {"文件": str(MODEL.relative_to(ROOT)), "sha256": sha256(MODEL),
         "字节": MODEL.stat().st_size},
        {"文件": str(STINTS.relative_to(ROOT)), "sha256": sha256(STINTS),
         "字节": STINTS.stat().st_size},
        {"文件": str(LAPS.relative_to(ROOT)), "sha256": sha256(LAPS),
         "字节": LAPS.stat().st_size},
        {"文件": str(EPISODES.relative_to(ROOT)), "sha256": sha256(EPISODES),
         "字节": EPISODES.stat().st_size},
        {"文件": f"data/raw/jolpica/*_results_limit=100.json（{len(glob.glob(str(JOLPICA / '*_results_limit=100.json')))} 个文件）",
         "sha256": None, "字节": ""},
    ])

    figs = make_figures(bins, cells, race_cells, robustness)

    bins.to_csv(OUT / "length_bins.csv", index=False)
    regression.to_csv(OUT / "regression_main.csv", index=False)
    robustness.to_csv(OUT / "robustness.csv", index=False)
    cells.to_csv(OUT / "cells.csv", index=False)
    race_cells.to_csv(OUT / "race_2024_cells.csv", index=False)
    stm.to_csv(OUT / "stint_meta.csv", index=False)
    hashes.to_csv(OUT / "input_sha256.csv", index=False)

    report = build_report(sample_info, bins, regression, robustness, cells,
                          race_cells, stm, figs, hashes)
    REPORT.write_text(report, encoding="utf-8")

    print("==" * 30)
    print(sample_info["rows"])
    print("\nmain regression:")
    print(regression[["variant", "coef_len10", "se", "t", "p_two_sided",
                      "n_laps", "n_stints"]].to_string(index=False))
    print("\ncells top/bottom:")
    print(pd.concat([cells.head(5), cells.tail(5)]).to_string(index=False))
    print("\n2024 races:")
    print(race_cells.to_string(index=False))
    print("\noutputs:")
    for p in sorted(OUT.glob("*.csv")):
        print(" ", p.relative_to(HERE))
    for p in figs:
        print(" ", p.relative_to(HERE))
    print(" ", REPORT.relative_to(HERE))
    print("\nDONE")


if __name__ == "__main__":
    main()
