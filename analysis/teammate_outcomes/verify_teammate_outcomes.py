"""T028 独立读回校验。

在项目根目录执行：

    .venv/bin/python analysis/teammate_outcomes/verify_teammate_outcomes.py

检查：
  1. 输入哈希、输出文件、两次运行哈希一致；
  2. 车对数量与筛选计数；
  3. 实际用时差、发车位差、第一段配速差可从原始数据重算（抽两个车对）；
  4. 回归系数与 SE、SC 分组、模型对照统计可重算；
  5. 报告里关键数字存在。

只读，不写任何结果文件（模型对照会调用 evaluate()，但不改代码）。
"""
from __future__ import annotations

import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1a.sim import RaceConfig, Strategy, TyreParams, evaluate  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
STRATEGIES = ROOT / "analysis" / "actual_strategies" / "strategies.parquet"
T020_RACE = ROOT / "analysis" / "actual_strategies" / "outputs" / "race_summary.csv"
LAPS = PROCESSED / "laps_pit_filled.parquet"
MODEL = PROCESSED / "model_laps.parquet"
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


def ols_cluster(y, X, groups):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtXi = np.linalg.pinv(X.T @ X)
    uniq, inv = np.unique(groups.astype(str), return_inverse=True)
    G = len(uniq)
    scores = np.zeros((G, X.shape[1]))
    np.add.at(scores, inv, X * resid[:, None])
    V = XtXi @ (scores.T @ scores * G / max(G - 1, 1)) @ XtXi
    return beta, np.sqrt(np.maximum(np.diag(V), 0)), G


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


def strategy_from_row(row) -> Strategy | None:
    try:
        n = int(row["n_stints"])
        comps = [row[f"stint{k}_compound"] for k in range(1, n + 1)]
        if any(c not in ["SOFT", "MEDIUM", "HARD"] for c in comps):
            return None
        stops = [(int(row[f"stint{k}_end"]), comps[k]) for k in range(1, n)]
        age = int(max(1, float(row["start_tyre_age"]))) if pd.notna(row["start_tyre_age"]) else 1
        st = Strategy(name=str(row["driver"]), start_compound=comps[0],
                      stops=stops, start_age=age)
        st.validate(int(row["total_laps"]))
        return st
    except Exception:
        return None


# ---------------------------------------------------------------- 文件与哈希
def check_files() -> None:
    required = ["report.md"]
    required += [f"outputs/{n}" for n in [
        "teammate_pairs.csv", "regression_pairs.csv", "sc_split.csv",
        "model_comparison.csv", "sample_flow.csv", "pair_model_detail.csv",
        "input_sha256.csv"]]
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
    for rel, path in [("analysis/actual_strategies/strategies.parquet", STRATEGIES),
                      ("analysis/actual_strategies/outputs/race_summary.csv", T020_RACE),
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


# ---------------------------------------------------------------- 样本
def check_sample() -> None:
    flow = pd.read_csv(OUT / "sample_flow.csv")
    pairs = pd.read_csv(OUT / "teammate_pairs.csv")
    diff = pairs[(~pairs["same_stops"]) & (pairs["stop_diff"] == 1)]
    ctrl = pairs[pairs["same_stops"]]
    check("pairs total = 496", len(pairs) == 496, str(len(pairs)))
    check("diff=1 pairs = 117", len(diff) == 117, str(len(diff)))
    check("same-stop pairs = 367", len(ctrl) == 367, str(len(ctrl)))
    check("diff>1 pairs = 12", int((~pairs["same_stops"] & (pairs["stop_diff"] != 1)).sum()) == 12)
    check("flow has 8 rows", len(flow) == 8, str(len(flow)))
    check("aligned pairs = 35", int(diff["aligned"].sum()) == 35, str(int(diff["aligned"].sum())))
    check("model detail pairs = 465",
          len(pd.read_csv(OUT / "pair_model_detail.csv")) == 465)


def check_pair_values() -> None:
    pairs = pd.read_csv(OUT / "teammate_pairs.csv")
    diff = pairs[(~pairs["same_stops"]) & (pairs["stop_diff"] == 1)].reset_index(drop=True)
    strat = pd.read_parquet(STRATEGIES)
    laps = pd.read_parquet(LAPS)
    cum = load_cum()
    ml = pd.read_parquet(MODEL)
    ml["t_corr"] = ml["lap_time_s"].astype(float) - 0.06 * ml["laps_remaining"].astype(float)
    first = (ml[ml["stint"] == 1].sort_values(["year", "round", "driver", "tyre_life"])
             .groupby(["year", "round", "driver"]).head(5)
             .groupby(["year", "round", "driver"])["t_corr"].mean())
    ok = True
    for r in diff.head(2).itertuples(index=False):
        cm = cum[(int(r.year), int(r.round), r.driver_more)]
        cl = cum[(int(r.year), int(r.round), r.driver_less)]
        align = min(max(cm), max(cl))
        t = cm[align] - cl[align]
        ok &= abs(t - float(r.time_diff)) < 1e-6
        pace_m = first[(r.year, r.round, r.driver_more)]
        pace_l = first[(r.year, r.round, r.driver_less)]
        ok &= abs((pace_m - pace_l) - float(r.pace_diff)) < 1e-9
    check("two pairs time_diff/pace_diff recompute", bool(ok))
    # 总体均值
    m = float(diff["time_diff"].mean())
    check("diff=1 mean time diff ≈ +9.47", abs(m - 9.466) < 5e-3, f"{m:.2f}")


def check_regression() -> None:
    pairs = pd.read_csv(OUT / "teammate_pairs.csv")
    reg = pd.read_csv(OUT / "regression_pairs.csv")
    diff = pairs[(~pairs["same_stops"]) & (pairs["stop_diff"] == 1)]
    ctrl = pairs[pairs["same_stops"]]
    pooled = pd.concat([diff.assign(group=1), ctrl.assign(group=0)])
    X = np.column_stack([np.ones(len(pooled)), pooled["group"].to_numpy(float),
                         pooled["grid_diff"].to_numpy(float),
                         pooled["pace_diff"].to_numpy(float)])
    y = pooled["time_diff"].to_numpy(float)
    beta, se, G = ols_cluster(y, X, pooled["race_key"].to_numpy())
    row = reg[(reg["组"] == "混合样本（差 1 vs 同停）")
              & (reg["变量"].astype(str).str.contains("多停一次"))].iloc[0]
    check("pooled extra-stop coefficient recompute",
          abs(float(row["系数"]) - beta[1]) < 1e-6
          and abs(float(row["聚类 SE"]) - se[1]) < 1e-6,
          f"{beta[1]:.3f} (SE {se[1]:.3f})")
    check("pooled coefficient positive & significant", beta[1] > 0 and beta[1] / se[1] > 2,
          f"t={beta[1] / se[1]:.2f}")
    # 对照组截距
    Xc = np.column_stack([np.ones(len(ctrl)), ctrl["grid_diff"].to_numpy(float),
                          ctrl["pace_diff"].to_numpy(float)])
    bc, sec, _ = ols_cluster(ctrl["time_diff"].to_numpy(float), Xc,
                             ctrl["race_key"].to_numpy())
    rowc = reg[(reg["组"] == "相同进站次数（对照）") & (reg["变量"] == "截距")].iloc[0]
    check("control intercept recompute", abs(float(rowc["系数"]) - bc[0]) < 1e-6,
          f"{bc[0]:.3f} (SE {sec[0]:.3f})")


def check_sc() -> None:
    pairs = pd.read_csv(OUT / "teammate_pairs.csv")
    sc = pd.read_csv(OUT / "sc_split.csv")
    diff = pairs[(~pairs["same_stops"]) & (pairs["stop_diff"] == 1)]
    for cls, n in [("SC/VSC 下", 26), ("绿旗下", 82), ("无法判定", 9)]:
        g = diff[diff["extra_stop_sc"] == cls]
        row = sc[sc["多停发生时机"] == cls].iloc[0]
        check(f"SC split n {cls} = {n}", len(g) == n and int(row["车对数"]) == n,
              str(len(g)))
        check(f"SC split mean {cls}", abs(float(row["用时差均值 (s)"]) - g["time_diff"].mean()) < 1e-6,
              f"{g['time_diff'].mean():.2f}")


def check_model() -> None:
    detail = pd.read_csv(OUT / "pair_model_detail.csv")
    mstats = pd.read_csv(OUT / "model_comparison.csv")
    diff = detail[(~detail["same_stops"]) & (detail["stop_diff"] == 1)]
    ctrl = detail[detail["same_stops"]]
    check("model diff=1 pairs = 108", len(diff) == 108, str(len(diff)))
    check("model control pairs = 357", len(ctrl) == 357, str(len(ctrl)))
    pear = float(stats.pearsonr(diff["actual_time_diff"], diff["model_time_diff"])[0])
    row = mstats[mstats["组"] == "不同进站次数（差 1）"].iloc[0]
    check("model Pearson recompute", abs(pear - float(row["Pearson"])) < 1e-6,
          f"{pear:.4f}")
    agree = float(np.mean(np.sign(diff["actual_time_diff"]) == np.sign(diff["model_time_diff"])))
    check("model sign agreement recompute", abs(agree - float(row["符号一致率"])) < 1e-6,
          f"{agree:.3f}")
    rate = float(np.mean((diff["model_time_diff"] > 0) & (diff["actual_time_diff"] < 0)))
    check("model fewer/actual more rate recompute",
          abs(rate - float(row["模型说少停更快、实际多停更快"])) < 1e-6, f"{rate:.3f}")
    # 独立重跑一个车对的 evaluate()
    r = diff.iloc[0]
    strat = pd.read_parquet(STRATEGIES)
    yy = int(r["race_key"].split("|")[0])
    rr = int(r["race_key"].split("|")[1])
    g = strat[(strat["year"] == yy) & (strat["round"] == rr)
              & (strat["driver"].isin([r["driver_more"], r["driver_less"]]))]
    st_m = strategy_from_row(g[g["driver"] == r["driver_more"]].iloc[0])
    st_l = strategy_from_row(g[g["driver"] == r["driver_less"]].iloc[0])
    rid = f"{int(r['race_key'].split('|')[0])}-{int(r['race_key'].split('|')[1])}"
    race = RaceConfig.from_project(ROOT, rid)
    tyre = TyreParams.from_posterior(
        ROOT / "models" / "tyre_v3" / f"outputs31_{int(rid.split('-')[0])}" / "posterior_core.npz",
        rid, [int(rid.split("-")[0])], MODEL, n_draws=200, seed=20261118)
    out = evaluate(race, tyre, [st_m, st_l], n_sims=2000, seed=20261118)
    m = out.set_index("strategy")["mean"]
    md = float(m[st_m.name] - m[st_l.name])
    check("model evaluate recompute one pair", abs(md - float(r["model_time_diff"])) < 1e-6,
          f"{md:.3f}")


def check_report() -> None:
    text = (HERE / "report.md").read_text(encoding="utf-8")
    for needle in ["117", "367", "8.2", "26", "82", "108", "25%", "0.157",
                   "20261118", "2000"]:
        check(f"report contains {needle}", needle in text)


def main() -> None:
    check_files()
    check_two_runs()
    check_sample()
    check_pair_values()
    check_regression()
    check_sc()
    check_model()
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
