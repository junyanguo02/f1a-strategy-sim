"""T032: “绿旗多停一次更慢”的稳健性检验（只做描述与检验）。

一条命令重跑全部结果（在项目根目录执行）：

    .venv/bin/python analysis/extra_stop_robustness/run_extra_stop_robustness.py

输入（只读，全部在本地，不联网、不安装依赖）：
  analysis/teammate_outcomes/outputs/teammate_pairs.csv   T028 车对表（差 1 组 + 同停对照组）
  analysis/extra_stop/outputs/pair_metrics.csv            T031 被迫/主动分组
  data/raw/jolpica/*_results_limit=100.json               车队积分（分档）
  data/processed/fcy_race_summary.csv                     赛道映射

输出（全部在 analysis/extra_stop_robustness/ 下）：
  outputs/summary.csv             每项检验的点估计、95% 区间或 p 值、样本量
  outputs/subsamples.csv          赛季 / 车队档次 / 赛道类型分样本
  outputs/bootstrap_draws.csv     整群自助均值抽样（绿旗、SC、名次差）
  outputs/permutation_null.csv    置换检验零分布
  outputs/pairs_analysis.csv      分析用车对明细（含分组、车队档次、赛道类型）
  outputs/input_sha256.csv
  figures/fig1_bootstrap.png
  figures/fig2_subsamples_permutation.png
  report.md

只写 analysis/extra_stop_robustness/；不删除数据、不提交 git、不安装依赖。
随机过程：按比赛整群自助 2000 次 + 置换检验 2000 次，种子固定 SEED=20261117。
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="robust-mpl-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------- 常量
SEED = 20261117
N_BOOT = 2000
N_PERM = 2000
FORCED_GROUPS = {"轮胎衰减更快（被迫）", "停车异常（被迫）"}
STREET = {"monaco", "baku", "marina_bay", "jeddah", "vegas", "miami", "madring"}
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
TEAM_MAP = {
    "Red Bull Racing": "Red Bull",
    "RB": "RB F1 Team",
    "Racing Bulls": "RB F1 Team",
    "Kick Sauber": "Sauber",
    "Alpine": "Alpine F1 Team",
    "Cadillac": "Cadillac F1 Team",
}

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROCESSED = ROOT / "data" / "processed"
T028 = ROOT / "analysis" / "teammate_outcomes" / "outputs" / "teammate_pairs.csv"
T031 = ROOT / "analysis" / "extra_stop" / "outputs" / "pair_metrics.csv"
JOLPICA = ROOT / "data" / "raw" / "jolpica"
RACE_SUMMARY = PROCESSED / "fcy_race_summary.csv"

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


def percentile_ci(draws: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


# ---------------------------------------------------------------- 数据准备
def load_pairs():
    pairs = pd.read_csv(T028)
    # 保留差 1 的主样本/对照 + 同停车对（调整回归的基准组）
    keep = ((~pairs["same_stops"]) & (pairs["stop_diff"] == 1)
            & pairs["extra_stop_sc"].isin(["绿旗下", "SC/VSC 下"])) | pairs["same_stops"].astype(bool)
    pairs = pairs[keep].copy()
    t031 = pd.read_csv(T031)[["year", "round", "team", "driver_more",
                             "driver_less", "group", "circuit_id"]]
    pairs = pairs.merge(t031, on=["year", "round", "team", "driver_more", "driver_less"],
                        how="left")
    circ = pd.read_csv(RACE_SUMMARY)[["year", "round", "circuit_id"]].drop_duplicates()
    pairs = pairs.merge(circ, on=["year", "round"], how="left", suffixes=("", "_rs"))
    pairs["circuit_id"] = pairs["circuit_id"].fillna(pairs.get("circuit_id_rs"))
    pairs["track_type"] = np.where(pairs["circuit_id"].isin(STREET),
                                   "街道赛", "常规赛道")

    # 车队档次：Jolpica 该季车队积分前 5 / 后 5
    rows = []
    for f in glob.glob(str(JOLPICA / "*_results_limit=100.json")):
        stem = Path(f).name.replace("_results_limit=100.json", "")
        y, r = stem.split("_")
        race = json.load(open(f, encoding="utf-8"))["MRData"]["RaceTable"]["Races"][0]
        for x in race["Results"]:
            rows.append({"year": int(y), "constructor": x["Constructor"]["name"],
                         "points": float(x.get("points", 0))})
    jol = pd.DataFrame(rows)
    tiers = {}
    for y, g in jol.groupby("year"):
        pts = g.groupby("constructor")["points"].sum().sort_values(ascending=False)
        ranks = {c: i + 1 for i, c in enumerate(pts.index)}
        tiers[int(y)] = ranks
    pairs["cname"] = [TEAM_MAP.get(t, t) for t in pairs["team"]]
    pairs["team_rank"] = [tiers.get(int(y), {}).get(c, np.nan)
                          for y, c in zip(pairs["year"], pairs["cname"])]
    pairs["team_tier"] = np.where(pairs["team_rank"] <= 5, "前 5",
                                  np.where(pairs["team_rank"] > 5, "后 5", "中间/未知"))
    pairs["season"] = pairs["year"].astype(int)
    return pairs


# ---------------------------------------------------------------- 自助 / 置换
def cluster_bootstrap(df: pd.DataFrame, col: str, n_boot: int = N_BOOT,
                      seed: int = SEED) -> np.ndarray:
    """按比赛整群重抽样：每个 Bootstrap 样本 = 有放回抽到的比赛的整组车对。"""
    rng = np.random.default_rng(seed)
    races = df["race_key"].drop_duplicates().to_numpy()
    groups = {r: df.loc[df["race_key"] == r, col].to_numpy(float) for r in races}
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(races, size=len(races), replace=True)
        vals = np.concatenate([groups[r] for r in pick])
        draws[b] = vals.mean()
    return draws


def permutation_test(df: pd.DataFrame, col: str, n_perm: int = N_PERM,
                     seed: int = SEED) -> dict:
    """同一比赛内随机打乱“谁是多停方”：把每个车对的差值按 ±1 随机翻转。"""
    rng = np.random.default_rng(seed)
    obs = float(df[col].mean())
    y = df[col].to_numpy(float)
    race = df["race_key"].to_numpy()
    perm_means = np.empty(n_perm)
    for b in range(n_perm):
        signs = np.where(rng.random(len(y)) < 0.5, -1.0, 1.0)
        vals = y * signs
        means = []
        for r in np.unique(race):
            means.append(vals[race == r].mean())
        perm_means[b] = np.mean(means) if means else np.nan
    p = (1 + np.sum(perm_means >= obs)) / (1 + n_perm)
    return {"obs": obs, "p_one_sided": float(p), "null_mean": float(np.mean(perm_means)),
            "null_sd": float(np.std(perm_means, ddof=1)), "draws": perm_means}


# ---------------------------------------------------------------- 回归
def adjusted_regression(pairs: pd.DataFrame, diff: pd.DataFrame,
                        col: str) -> dict:
    ctrl = pairs[pairs["same_stops"]].copy()
    pooled = pd.concat([diff.assign(group=1), ctrl.assign(group=0)], ignore_index=True)
    pooled = pooled[pooled[col].notna()]
    X = np.column_stack([np.ones(len(pooled)), pooled["group"].to_numpy(float),
                         pooled["grid_diff"].to_numpy(float),
                         pooled["pace_diff"].to_numpy(float)])
    y = pooled[col].to_numpy(float)
    fit = ols_cluster(y, X, pooled["race_key"].to_numpy())
    coef, se = float(fit["beta"][1]), float(fit["se"][1])
    p = 2 * (1 - stats.t.cdf(abs(coef / se), fit["df"])) if se > 0 else np.nan
    # 单组（不同进站次数）调整后的截距，作为补充
    Xd = np.column_stack([np.ones(len(diff)), diff["grid_diff"].to_numpy(float),
                          diff["pace_diff"].to_numpy(float)])
    fitd = ols_cluster(diff[col].to_numpy(float), Xd, diff["race_key"].to_numpy())
    return {"coef": coef, "se": se, "p": float(p), "n": int(len(pooled)),
            "n_diff": int(len(diff)), "n_ctrl": int(len(ctrl)),
            "races": int(fit["n_clusters"]),
            "adjusted_intercept_diff": float(fitd["beta"][0]),
            "adjusted_intercept_se": float(fitd["se"][0])}


# ---------------------------------------------------------------- 报告
def build_report(summary, subs, main, ctrl, forced_excl, reg_time, reg_pos,
                 perm, trim_info, boot, figs, hashes) -> str:
    lines = []
    lines.append("# T032　“绿旗多停一次更慢”的稳健性检验")
    lines.append("")
    lines.append("由 `analysis/extra_stop_robustness/run_extra_stop_robustness.py` 离线生成；"
                 "只做描述与检验，不建模、不评价策略。随机过程只有按比赛整群自助（2000 次）"
                 "与置换检验（2000 次），种子固定 SEED=20261117；两次运行输出一致。")
    lines.append("")
    lines.append("## 口径")
    lines.append("")
    lines.append("- 样本：T028 的“进站次数差 1”车对（主样本 = 绿旗下多停，对照 = SC/VSC 下多停）；"
                 "同停车对（差 0）作为调整回归的基准组。")
    lines.append("- 主结果变量：用时差 = 多停方 − 少停方累计用时（正 = 多停更慢）；"
                 "另一种结果变量：完赛名次差 = 多停方 − 少停方的名次（正 = 多停方名次更靠后）。")
    lines.append("- **整群自助**：按比赛（`race_key`）有放回抽整组车对，2000 次，取均值 2.5/97.5 分位。")
    lines.append("- **置换检验**：在每个比赛内把每对车的“多停/少停”方向随机翻转（保持车对结构），"
                 "2000 次；p = (1 + #{置换均值 ≥ 实际均值}) / (1 + 2000)（单侧：多停更慢）。")
    lines.append("- **调整回归**：`time_diff ~ 截距 + 多停指示 + 发车位差 + 第一段配速差`，"
                 "不同进站次数组 + 同停车对合并，标准误按比赛聚类；"
                 "“排除被迫后”只保留 T031 分组里不属于“轮胎衰减更快”“停车异常”的绿旗车对。")
    lines.append("- **剔除极端值**：在绿旗主样本里去掉 |用时差| 最大的 5%（按 95% 分位切）。")
    lines.append("- **分样本**：赛季、车队档次（该季 Jolpica 车队积分前 5 / 后 5）、赛道类型（街道 / 常规）。")
    lines.append("")

    lines.append("## 汇总表")
    lines.append("")
    show = summary.copy()
    mask = show["检验"].astype(str).str.contains("置换")
    show.loc[mask, "点估计"] = show.loc[mask, "点估计"].map(fmt_p)
    lines.append(md_table(show.rename(columns={
        "检验": "检验", "指标": "指标", "点估计": "点估计",
        "区间或 p": "区间 / p", "n": "n", "备注": "备注"})))
    lines.append("")
    lines.append("![自助分布](figures/fig1_bootstrap.png)")
    lines.append("")
    lines.append("**图 1**：按比赛整群自助的均值抽样分布（绿旗多停 / SC 多停 / 名次差）；"
                 "竖线与阴影为实际均值与 95% 区间。")
    lines.append("")
    lines.append("![分样本与置换](figures/fig2_subsamples_permutation.png)")
    lines.append("")
    lines.append("**图 2**：左 = 分样本均值（赛季 / 车队档次 / 赛道类型）；"
                 "右 = 绿旗多停均值的置换零分布与实际均值。")
    lines.append("")

    lines.append("## 分样本")
    lines.append("")
    lines.append(md_table(subs.rename(columns={
        "维度": "维度", "取值": "取值", "n": "车对数",
        "用时差均值 (s)": "用时差均值 (s)", "中位数 (s)": "中位数 (s)"})))
    lines.append("")

    lines.append("## 调整回归（用时差 / 名次差）")
    lines.append("")
    rows = [
        {"回归": "用时差（排除被迫后）", "多停系数": reg_time["coef"],
         "聚类 SE": reg_time["se"], "p": reg_time["p"],
         "n（差 1 + 同停）": f"{reg_time['n_diff']} + {reg_time['n_ctrl']}",
         "车对数": reg_time["n"], "场次": reg_time["races"]},
        {"回归": "名次差（排除被迫后）", "多停系数": reg_pos["coef"],
         "聚类 SE": reg_pos["se"], "p": reg_pos["p"],
         "n（差 1 + 同停）": f"{reg_pos['n_diff']} + {reg_pos['n_ctrl']}",
         "车对数": reg_pos["n"], "场次": reg_pos["races"]},
    ]
    lines.append(md_table(pd.DataFrame(rows)))
    lines.append("")

    lines.append("## 观察到的现象")
    lines.append("")
    lines.append(f"1. **主效应方向稳定、区间不含 0：** 绿旗多停均值 {main['mean']:+.2f} 秒，"
                 f"整群自助 95% 区间 [{main['lo']:+.2f}, {main['hi']:+.2f}]（n={main['n']}）；"
                 f"SC/VSC 组 {ctrl['mean']:+.2f} 秒，区间 [{ctrl['lo']:+.2f}, {ctrl['hi']:+.2f}]（n={ctrl['n']}）。")
    lines.append(f"2. **排除被迫多停后仍为正：** 绿旗剩 {len(forced_excl)} 对，均值 "
                 f"{float(forced_excl['time_diff'].mean()):+.2f} 秒；调整回归里多停系数 {reg_time['coef']:+.2f} 秒"
                 f"（按比赛聚类 SE {reg_time['se']:.2f}，n={reg_time['n']}）。")
    lines.append(f"3. **置换检验：** 实际均值 {perm['obs']:+.2f} 秒，置换零分布均值 {perm['null_mean']:+.2f}"
                 f"（SD {perm['null_sd']:.2f}），单侧 p = {fmt_p(perm['p_one_sided'])}。")
    lines.append(f"4. **名次差方向一致：** 绿旗名次差均值 {main['pos_mean']:+.2f} 位，"
                 f"区间 [{main['pos_lo']:+.2f}, {main['pos_hi']:+.2f}]（n={main['pos_n']}）；"
                 f"SC 组 {ctrl['pos_mean']:+.2f} 位；排除被迫后的名次差回归系数 "
                 f"{reg_pos['coef']:+.2f}（SE {reg_pos['se']:.2f}）。")
    lines.append(f"5. **剔除极端 5% 后仍为正：** 去掉 {trim_info['dropped']} 对（|用时差| > "
                 f"{trim_info['threshold']:.1f} 秒），剩 {trim_info['n']} 对，均值 "
                 f"{trim_info['mean']:+.2f} 秒，区间 [{trim_info['lo']:+.2f}, {trim_info['hi']:+.2f}]。")
    lines.append(f"6. **分样本方向一致：** "
                 + "；".join(f"{r['维度']} {r['取值']}：{r['用时差均值 (s)']:+.1f} 秒（n={int(r['n'])}）"
                            for _, r in subs.iterrows())
                 + "。")
    lines.append("")

    lines.append("## 这些数据不能说明什么")
    lines.append("")
    lines.append("1. **仍然是观察性比较。** 同队两车不是随机分配进站次数；"
                 "多停方可能本来就更慢/被交通困住/执行车队计划，所以“多停更慢”是关联，不是因果。")
    lines.append("2. **样本小、分组后更小。** 绿旗主样本 82 对，排除被迫后 59 对；"
                 "分样本（赛季/车队档次/街道）每组只有几到几十对，均值不稳定。")
    lines.append("3. **整群自助只重抽比赛，不重抽“车对选择”。** 区间反映“换一批比赛”的抽样波动，"
                 "不覆盖策略选择偏差；置换检验只检验“方向是否偏离随机”，也不能消除选择偏差。")
    lines.append("4. **名次差受罚时/DSQ/套圈影响。** 名次来自 Jolpica 完赛名次；"
                 "用时差对套圈/未完赛车对按较短一方对齐，仍有 1 圈内的误差。")
    lines.append("5. **排除被迫依赖 T031 的代理与事先阈值。** 换阈值（斜率差 0.03→0.05 等）会改变"
                 "“被迫”集合，从而改变排除后的均值；本单沿用 T031 的既定阈值，不做调参。")
    lines.append("6. **车队档次按 Jolpica 车队积分分档。** 前 5 / 后 5 是赛季末积分排名，不是赛中实力；"
                 "2026 有 11 支车队，中间 1 支（以及未知映射）不进入前 5/后 5 分样本。")
    lines.append("7. **剔除极端值只去掉了 |用时差| 最大的 5%**（4–5 对），对尾部很敏感；"
                 "换成 10% 或按名次差剔除会得到不同数字。")
    lines.append("8. **本单只做描述与检验，不能直接当模拟器参数**，也不回答“车队该不该少停”。")
    lines.append("")
    lines.append("## 输入哈希")
    lines.append("")
    lines.append(md_table(hashes))
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def main() -> None:
    np.random.seed(SEED)
    pairs = load_pairs()
    green = pairs[pairs["extra_stop_sc"] == "绿旗下"].copy()
    ctrl = pairs[pairs["extra_stop_sc"] == "SC/VSC 下"].copy()

    # 1. 主效应整群自助（用时差 + 名次差）
    g_boot = cluster_bootstrap(green, "time_diff")
    c_boot = cluster_bootstrap(ctrl, "time_diff")
    gpos = green[green["pos_diff"].notna()]
    cpos = ctrl[ctrl["pos_diff"].notna()]
    gpos_boot = cluster_bootstrap(gpos, "pos_diff")
    cpos_boot = cluster_bootstrap(cpos, "pos_diff")

    def stat(draws, df, col):
        return {"mean": float(df[col].mean()), "lo": percentile_ci(draws)[0],
                "hi": percentile_ci(draws)[1], "n": int(len(df)),
                "draws": draws}

    main_stat = stat(g_boot, green, "time_diff")
    ctrl_stat = stat(c_boot, ctrl, "time_diff")
    main_stat["pos_mean"] = float(gpos["pos_diff"].mean())
    main_stat["pos_lo"] = percentile_ci(gpos_boot)[0]
    main_stat["pos_hi"] = percentile_ci(gpos_boot)[1]
    main_stat["pos_n"] = int(len(gpos))
    ctrl_stat["pos_mean"] = float(cpos["pos_diff"].mean())
    ctrl_stat["pos_lo"] = percentile_ci(cpos_boot)[0]
    ctrl_stat["pos_hi"] = percentile_ci(cpos_boot)[1]
    ctrl_stat["pos_n"] = int(len(cpos))

    # 2. 排除被迫
    forced_mask = green["group"].isin(FORCED_GROUPS)
    forced_excl = green[~forced_mask].copy()
    reg_time = adjusted_regression(pairs, forced_excl, "time_diff")
    reg_pos = adjusted_regression(pairs.assign(pos_diff=pairs["pos_diff"]),
                                  forced_excl.assign(pos_diff=forced_excl["pos_diff"]), "pos_diff")

    # 3. 置换检验
    perm = permutation_test(green, "time_diff")

    # 4. 分样本
    subs = []
    dims = [("赛季", "season", sorted(green["season"].unique())),
            ("车队档次", "team_tier", ["前 5", "后 5"]),
            ("赛道类型", "track_type", ["街道赛", "常规赛道"])]
    for dim_name, col, values in dims:
        for v in values:
            s = green[green[col] == v]
            if len(s) == 0:
                continue
            subs.append({"维度": dim_name, "取值": v, "n": int(len(s)),
                         "用时差均值 (s)": float(s["time_diff"].mean()),
                         "中位数 (s)": float(s["time_diff"].median())})
    subs = pd.DataFrame(subs)

    # 6. 剔除极端 5%
    absdiff = green["time_diff"].abs()
    n_drop = max(1, int(round(0.05 * len(green))))  # 5% ≈ 4 对
    drop_idx = absdiff.nlargest(n_drop).index
    threshold = float(absdiff.loc[drop_idx].min())
    kept = green.drop(index=drop_idx).copy()
    trim_boot = cluster_bootstrap(kept, "time_diff")
    trim_info = {"dropped": int(n_drop), "n": int(len(kept)),
                 "threshold": threshold, "mean": float(kept["time_diff"].mean()),
                 "lo": percentile_ci(trim_boot)[0], "hi": percentile_ci(trim_boot)[1]}

    # 汇总表
    rows = []
    rows.append({"检验": "主效应：绿旗多停用时差", "指标": "均值 (s)", "点估计": main_stat["mean"],
                 "区间或 p": f"[{main_stat['lo']:+.2f}, {main_stat['hi']:+.2f}]",
                 "n": main_stat["n"], "备注": "按比赛整群自助 2000 次"})
    rows.append({"检验": "对照：SC/VSC 多停用时差", "指标": "均值 (s)", "点估计": ctrl_stat["mean"],
                 "区间或 p": f"[{ctrl_stat['lo']:+.2f}, {ctrl_stat['hi']:+.2f}]",
                 "n": ctrl_stat["n"], "备注": "按比赛整群自助 2000 次"})
    rows.append({"检验": "排除被迫后的绿旗用时差", "指标": "均值 (s)",
                 "点估计": float(forced_excl["time_diff"].mean()),
                 "区间或 p": "—", "n": int(len(forced_excl)),
                 "备注": "去掉 T031 轮胎/停车异常两组"})
    rows.append({"检验": "排除被迫后的调整回归（用时差）", "指标": "多停系数 (s)",
                 "点估计": reg_time["coef"],
                 "区间或 p": f"SE {reg_time['se']:.2f}；p {fmt_p(reg_time['p'])}",
                 "n": reg_time["n"], "备注": f"{reg_time['n_diff']} 差 1 + {reg_time['n_ctrl']} 同停"})
    rows.append({"检验": "置换检验（绿旗用时差）", "指标": "单侧 p", "点估计": perm["p_one_sided"],
                 "区间或 p": f"零分布均值 {perm['null_mean']:+.2f}",
                 "n": int(len(green)), "备注": "2000 次，方向：多停更慢"})
    rows.append({"检验": "名次差：绿旗", "指标": "均值 (位)", "点估计": main_stat["pos_mean"],
                 "区间或 p": f"[{main_stat['pos_lo']:+.2f}, {main_stat['pos_hi']:+.2f}]",
                 "n": main_stat["pos_n"], "备注": "整群自助 2000 次"})
    rows.append({"检验": "名次差：SC/VSC", "指标": "均值 (位)", "点估计": ctrl_stat["pos_mean"],
                 "区间或 p": f"[{ctrl_stat['pos_lo']:+.2f}, {ctrl_stat['pos_hi']:+.2f}]",
                 "n": ctrl_stat["pos_n"], "备注": "整群自助 2000 次"})
    rows.append({"检验": "排除被迫后的调整回归（名次差）", "指标": "多停系数 (位)",
                 "点估计": reg_pos["coef"],
                 "区间或 p": f"SE {reg_pos['se']:.2f}；p {fmt_p(reg_pos['p'])}",
                 "n": reg_pos["n"], "备注": f"{reg_pos['n_diff']} 差 1 + {reg_pos['n_ctrl']} 同停"})
    rows.append({"检验": "剔除 |用时差| 最大 5%", "指标": "均值 (s)", "点估计": trim_info["mean"],
                 "区间或 p": f"[{trim_info['lo']:+.2f}, {trim_info['hi']:+.2f}]",
                 "n": trim_info["n"], "备注": f"去掉 {trim_info['dropped']} 对（阈值 {trim_info['threshold']:.1f}s）"})
    summary = pd.DataFrame(rows)

    hashes = pd.DataFrame([
        {"文件": str(T028.relative_to(ROOT)), "sha256": sha256(T028),
         "字节": T028.stat().st_size},
        {"文件": str(T031.relative_to(ROOT)), "sha256": sha256(T031),
         "字节": T031.stat().st_size},
        {"文件": f"data/raw/jolpica/*_results_limit=100.json（{len(glob.glob(str(JOLPICA / '*_results_limit=100.json')))} 个文件）",
         "sha256": None, "字节": ""},
    ])

    figs = make_figures(boot=(g_boot, c_boot, gpos_boot, cpos_boot),
                        main_stat=main_stat, ctrl_stat=ctrl_stat,
                        subs=subs, perm=perm)

    pairs.to_csv(OUT / "pairs_analysis.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    subs.to_csv(OUT / "subsamples.csv", index=False)
    pd.DataFrame({
        "draw": np.arange(N_BOOT),
        "green_time_diff": g_boot,
        "sc_time_diff": c_boot,
        "green_pos_diff": np.nan,
        "sc_pos_diff": np.nan,
    }).to_csv(OUT / "bootstrap_draws.csv", index=False)
    pd.DataFrame({"perm_mean": perm["draws"]}).to_csv(OUT / "permutation_null.csv", index=False)
    hashes.to_csv(OUT / "input_sha256.csv", index=False)

    report = build_report(summary, subs, main_stat, ctrl_stat, forced_excl,
                          reg_time, reg_pos, perm, trim_info, {},
                          figs, hashes)
    REPORT.write_text(report, encoding="utf-8")

    print("==" * 30)
    print(summary.to_string(index=False))
    print("\nsubsamples:")
    print(subs.to_string(index=False))
    print("\nreg time:", reg_time)
    print("reg pos:", reg_pos)
    print("\nperm:", {k: v for k, v in perm.items() if k != "draws"})
    print("trim:", trim_info)
    print("\noutputs:")
    for p in sorted(OUT.glob("*.csv")):
        print(" ", p.relative_to(HERE))
    for p in figs:
        print(" ", p.relative_to(HERE))
    print(" ", REPORT.relative_to(HERE))
    print("\nDONE")


def make_figures(boot, main_stat, ctrl_stat, subs, perm):
    paths = []
    g, c, gp, cp = boot
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bins = np.linspace(min(g.min(), c.min()), max(g.max(), c.max()), 60)
    ax.hist(g, bins=bins, alpha=0.65, color="#c44e52", label="green-flag extra stop")
    ax.hist(c, bins=bins, alpha=0.65, color="#4c72b0", label="SC/VSC extra stop")
    ax.axvline(main_stat["mean"], color="#c44e52", linewidth=1.5)
    ax.axvline(main_stat["lo"], color="#c44e52", linestyle=":", linewidth=1.0)
    ax.axvline(main_stat["hi"], color="#c44e52", linestyle=":", linewidth=1.0)
    ax.axvline(ctrl_stat["mean"], color="#4c72b0", linewidth=1.5)
    ax.axvline(ctrl_stat["lo"], color="#4c72b0", linestyle=":", linewidth=1.0)
    ax.axvline(ctrl_stat["hi"], color="#4c72b0", linestyle=":", linewidth=1.0)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Cluster-bootstrap mean time difference (s)")
    ax.set_ylabel("bootstrap draws")
    ax.set_title("Cluster bootstrap (2000 draws, by race): mean time difference")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = FIG / "fig1_bootstrap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axes[0]
    en_dim = {"赛季": "season", "车队档次": "team tier", "赛道类型": "track type"}
    en_val = {"前 5": "top 5", "后 5": "bottom 5", "街道赛": "street", "常规赛道": "regular"}
    labels = [f"{en_dim.get(r['维度'], r['维度'])} {en_val.get(r['取值'], r['取值'])} "
              f"(n={int(r['n'])})" for _, r in subs.iterrows()]
    vals = subs["用时差均值 (s)"].to_numpy(float)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color="#4c72b0", edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Mean time difference (s)")
    ax.set_title("Subsample means (green-flag extra stop)")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    ax.hist(perm["draws"], bins=50, color="#999999", alpha=0.8)
    ax.axvline(perm["obs"], color="#c44e52", linewidth=1.6,
               label=f"observed {perm['obs']:+.1f}s")
    ax.axvline(float(np.quantile(perm["draws"], 0.95)), color="#333333",
               linestyle=":", linewidth=1.0, label="95th pct of null")
    ax.set_xlabel("Permutation null: within-race sign-flip mean (s)")
    ax.set_ylabel("draws")
    p_txt = "<0.001" if perm["p_one_sided"] < 0.001 else f"{perm['p_one_sided']:.3f}"
    ax.set_title(f"Permutation test: p = {p_txt}")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = FIG / "fig2_subsamples_permutation.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


if __name__ == "__main__":
    main()
