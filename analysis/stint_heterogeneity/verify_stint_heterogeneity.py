"""T026 独立读回校验。

在项目根目录执行：

    .venv/bin/python analysis/stint_heterogeneity/verify_stint_heterogeneity.py

检查：
  1. 输入哈希、输出文件、两次运行哈希一致；
  2. 每段斜率的段数/筛选与两段样例可从 model_laps 独立重算；
  3. 分布与方差分解（含扣除估计误差后的段间 SD）可重算；
  4. 前 8 圈斜率 vs 段长、3b/3c 进站关系、车手/车队 R² 可重算；
  5. 报告里关键数字存在。

只读，不写任何结果文件。
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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


def ols_slope(age, y):
    X = np.column_stack([np.ones(len(age)), age])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(age) - 2
    se = np.sqrt((resid @ resid / dof) / np.sum((age - age.mean()) ** 2))
    return float(beta[1]), float(se)


def decompose(slopes, ses, cells):
    slopes = np.asarray(slopes, float)
    ses = np.asarray(ses, float)
    cells = np.asarray(cells, str)
    n = len(slopes)
    df = pd.DataFrame({"s": slopes, "se2": ses ** 2, "cell": cells})
    grp = df.groupby("cell")
    cnt = grp.size().to_numpy(float)
    mean = grp["s"].mean().to_numpy(float)
    within_var = grp["s"].var(ddof=1).to_numpy(float)
    mse = grp["se2"].mean().to_numpy(float)
    C = len(cnt)
    samp_var_mean = mse / cnt
    v_between = max(float(np.var(mean, ddof=1) - np.mean(samp_var_mean)), 0.0) if C >= 2 else 0.0
    w = np.where(cnt > 1, cnt - 1, 0.0)
    if w.sum() > 0:
        v_within_obs = float(np.sum(w * np.nan_to_num(within_var)) / w.sum())
        v_within_err = float(np.sum(w * mse) / w.sum())
        v_within = max(v_within_obs - v_within_err, 0.0)
    else:
        v_within = 0.0
    v_true = v_between + v_within
    v_obs = float(np.var(slopes, ddof=1))
    v_err = float(np.mean(ses ** 2))
    return dict(v_obs=v_obs, v_err=v_err, v_true=v_true, sd_true=np.sqrt(v_true),
                sd_between=np.sqrt(v_between), sd_within=np.sqrt(v_within),
                err_share=v_err / v_obs)


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


# ---------------------------------------------------------------- 文件与哈希
def check_files() -> None:
    required = ["report.md"]
    required += [f"outputs/{n}" for n in [
        "stint_slopes.csv", "slope_distribution.csv", "variance_decomposition.csv",
        "first8_vs_length.csv", "first8_by_length.csv", "pitstop_highlow.csv",
        "extra_stop.csv", "driver_team_variance.csv", "input_sha256.csv"]]
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
    for rel, path in [("data/processed/model_laps.parquet", MODEL),
                      ("analysis/actual_strategies/stints.csv", STINTS),
                      ("analysis/actual_strategies/strategies.csv", STRATEGIES),
                      ("analysis/actual_strategies/outputs/race_summary.csv", T020_RACE),
                      ("data/processed/laps_pit_filled.parquet", LAPS)]:
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


# ---------------------------------------------------------------- 斜率
def check_slopes() -> None:
    ml = pd.read_parquet(MODEL)
    ml["t_corr"] = ml["lap_time_s"].astype(float) - 0.06 * ml["laps_remaining"].astype(float)
    tab = pd.read_csv(OUT / "stint_slopes.csv")
    check("total stints = 3957", ml["stint_uid"].nunique() == 3957,
          str(ml["stint_uid"].nunique()))
    check("stints with >=10 clean laps = 3260",
          int((ml.groupby("stint_uid").size() >= 10).sum()) == 3260)
    check("slope table rows = 3260", len(tab) == 3260, str(len(tab)))
    check("usable slopes = 3257", int(tab["slope"].notna().sum()) == 3257,
          str(int(tab["slope"].notna().sum())))
    check("per-compound counts", tab[tab["slope"].notna()].groupby("compound").size()
          .to_dict() == {"HARD": 1483, "MEDIUM": 1280, "SOFT": 494},
          str(tab[tab["slope"].notna()].groupby("compound").size().to_dict()))
    for c, expect in [("HARD", 0.051478), ("MEDIUM", 0.059823), ("SOFT", 0.075415)]:
        got = float(tab[tab["compound"] == c]["slope"].mean())
        check(f"mean slope {c}", abs(got - expect) < 5e-4, f"{got:.4f}")
    # 独立重算两段的斜率
    for uid in ["2023|1|VER|1", "2023|1|PER|2"]:
        pass
    sample = tab.dropna(subset=["slope"]).head(2)
    ok = True
    for r in sample.itertuples(index=False):
        g = ml[ml["stint_uid"] == r.stint_uid]
        g = g[g["tyre_life"] <= 20].sort_values("tyre_life")
        if len(g) < 5:
            continue
        sl, se = ols_slope(g["tyre_life"].to_numpy(float), g["t_corr"].to_numpy(float))
        ok &= abs(sl - float(r.slope)) < 1e-8
    check("sample stint slope recompute", ok, f"{len(sample)} stints")


def check_distribution() -> None:
    tab = pd.read_csv(OUT / "stint_slopes.csv")
    s = tab.dropna(subset=["slope"])
    d = decompose(s["slope"], s["se"], s["cell"])
    dist = pd.read_csv(OUT / "slope_distribution.csv")
    row = dist[dist["分组"] == "全部"].iloc[0]
    check("overall true SD = 0.0866", abs(float(row["sd_true"]) - d["sd_true"]) < 1e-6,
          f"{d['sd_true']:.4f}")
    check("overall estimation error SD ≈ 0.0359",
          abs(float(row["sd_err"]) - np.sqrt(d["v_err"])) < 1e-6, f"{np.sqrt(d['v_err']):.4f}")
    check("observed SD ≈ 0.0887", abs(float(row["sd_obs"]) - np.sqrt(d["v_obs"])) < 1e-6)
    var = pd.read_csv(OUT / "variance_decomposition.csv")
    v = var[var["分组"] == "全部"].iloc[0]
    check("variance decomposition recompute",
          abs(float(v["sd_true"]) - d["sd_true"]) < 1e-6
          and abs(float(v["sd_between_true"]) - d["sd_between"]) < 1e-6
          and abs(float(v["sd_within_true"]) - d["sd_within"]) < 1e-6,
          f"true={d['sd_true']:.3f}, between={d['sd_between']:.3f}, "
          f"within={d['sd_within']:.3f}")
    check("bootstrap CI ordered and contains point",
          float(v["sd_true_lo"]) < float(v["sd_true"]) < float(v["sd_true_hi"]),
          f"{v['sd_true_lo']:.3f}–{v['sd_true_hi']:.3f}")
    check("error share ~16%", abs(float(v["err_share"]) - 0.164) < 0.01,
          f"{float(v['err_share']):.3f}")
    check("per-compound rows = 4", len(var) == 4, str(len(var)))


def check_first8() -> None:
    tab = pd.read_csv(OUT / "stint_slopes.csv")
    s = tab[(~tab["is_last"]) & (~tab["sc_short"]) & tab["slope8"].notna()]
    pear = float(s["slope8"].corr(s["length"]))
    spe = float(s["slope8"].corr(s["length"], method="spearman"))
    check("first8 vs length Pearson ≈ -0.206", abs(pear - (-0.2062)) < 5e-3, f"{pear:.4f}")
    check("first8 vs length Spearman ≈ -0.239", abs(spe - (-0.2389)) < 5e-3, f"{spe:.4f}")
    q = pd.read_csv(OUT / "first8_vs_length.csv")
    check("first8 quantile table 4 rows and ordered lengths",
          len(q) == 4 and q["段长均值"].is_monotonic_decreasing,
          str(list(np.round(q["段长均值"], 1))))


def check_pitstop() -> None:
    tab = pd.read_csv(OUT / "stint_slopes.csv")
    strat = pd.read_csv(STRATEGIES)
    t020 = pd.read_csv(T020_RACE)
    keep = set(map(tuple, t020[t020["usable"] & t020["dry_for_analysis"]
                                & (~t020["has_red_flag"])][["year", "round"]].values))
    st = strat[["year", "round", "driver", "n_pit_stops", "start_compound",
                "finisher", "finish_position"]]
    ts = tab.merge(st, on=["year", "round", "driver"], how="left")
    ts = ts.sort_values(["year", "round", "driver", "stint"]).reset_index(drop=True)
    ts["min_stint"] = ts.groupby(["year", "round", "driver"])["stint"].transform("min")
    ts = ts[ts["stint"] == ts["min_stint"]]
    ts = ts[(ts["finisher"] == True) & ts["start_compound"].isin(["SOFT", "MEDIUM", "HARD"])
            & ts["slope8"].notna()]
    ts = ts[[(int(y), int(r)) in keep for y, r in zip(ts["year"], ts["round"])]]
    ts["group"] = ts["year"].astype(str) + "|" + ts["round"].astype(str) + "|" + ts["start_compound"]
    cnt = ts["group"].value_counts()
    ts = ts[ts["group"].isin(cnt[cnt >= 4].index)]
    ts["median8"] = ts.groupby("group")["slope8"].transform("median")
    ts["high"] = ts["slope8"] > ts["median8"]
    ts["ns_w"] = ts["n_pit_stops"] - ts.groupby("group")["n_pit_stops"].transform("mean")
    y = ts["ns_w"].to_numpy(float)
    X = np.column_stack([np.ones(len(ts)), ts["high"].astype(float)])
    beta, se, G = ols_cluster(y, X, ts["race_key"].to_numpy())
    check("3b sample = 1173 cars / 74 races",
          len(ts) == 1173 and ts["race_key"].nunique() == 74,
          f"{len(ts)} cars, {ts['race_key'].nunique()} races")
    check("3b within diff ≈ -0.035", abs(beta[1] - (-0.03537)) < 1e-4, f"{beta[1]:.4f}")
    check("3b cluster SE ≈ 0.0265", abs(se[1] - 0.02650) < 1e-3, f"{se[1]:.4f}")
    # 3c
    top10 = strat[strat["finish_position"].notna() & (strat["finish_position"] <= 10)]
    mainstream = (top10.groupby(["year", "round"])["n_pit_stops"]
                  .agg(lambda x: x.mode().iloc[0]).rename("mainstream").reset_index())
    ts = ts.merge(mainstream, on=["year", "round"], how="left")
    ts = ts[ts["mainstream"].notna()].copy()
    ts["grp"] = np.where(ts["n_pit_stops"] == ts["mainstream"], "same",
                         np.where(ts["n_pit_stops"] == ts["mainstream"] + 1, "+1", "other"))
    sub = ts[ts["grp"].isin(["same", "+1"])]
    y = sub["slope8"].to_numpy(float)
    X = np.column_stack([np.ones(len(sub)), (sub["grp"] == "+1").astype(float)])
    beta, se, G = ols_cluster(y, X, sub["race_key"].to_numpy())
    check("3c +1 group n = 195", int((ts["grp"] == "+1").sum()) == 195,
          str(int((ts["grp"] == "+1").sum())))
    check("3c slope8 diff ≈ +0.0108", abs(beta[1] - 0.01078) < 1e-3, f"{beta[1]:.4f}")
    check("3c diff not significant", abs(beta[1] / se[1]) < 1.96, f"t={beta[1] / se[1]:.2f}")


def check_driver_team() -> None:
    tab = pd.read_csv(OUT / "stint_slopes.csv")
    s = tab[tab["slope"].notna()].copy()
    s["dr"] = s["year"].astype(str) + "|" + s["driver"]
    s["dr_c"] = s["dr"] + "|" + s["compound"]
    s["tm"] = s["year"].astype(str) + "|" + s["team"]

    def loo(key):
        g = s.groupby(key)["slope"]
        return (g.transform("sum") - s["slope"]) / (g.transform("size") - 1)

    s["dr_loo"] = loo("dr")
    s["dr_c_loo"] = loo("dr_c")
    s["tm_loo"] = loo("tm")
    s = s.dropna(subset=["dr_loo", "dr_c_loo", "tm_loo"])
    y = s["slope"].to_numpy(float)
    base = pd.get_dummies(s["circuit_id"].astype(str) + "|" + s["compound"].astype(str),
                          drop_first=True, dtype=float).to_numpy(float)

    def r2(X):
        X = np.column_stack([np.ones(len(y)), X])
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ b
        return 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)

    r0 = r2(base)
    rd = r2(np.column_stack([base, s["dr_loo"].to_numpy(float)]))
    rt = r2(np.column_stack([base, s["tm_loo"].to_numpy(float)]))
    dt = pd.read_csv(OUT / "driver_team_variance.csv")
    check("driver ΔR² tiny", rd - r0 < 0.002 and abs(rd - r0 - 0.000305) < 1e-3,
          f"Δ={rd - r0:.5f}")
    check("team ΔR² tiny", rt - r0 < 0.005 and abs(rt - r0 - 0.001292) < 1e-3,
          f"Δ={rt - r0:.5f}")
    check("driver/team table has 4 rows", len(dt) == 4, str(len(dt)))


def check_report() -> None:
    text = (HERE / "report.md").read_text(encoding="utf-8")
    for needle in ["3257", "3260", "0.087", "0.050", "0.071", "-0.206",
                   "-0.239", "1.773", "195", "0.0003", "0.0013", "16%"]:
        check(f"report contains {needle}", needle in text)


def main() -> None:
    check_files()
    check_two_runs()
    check_slopes()
    check_distribution()
    check_first8()
    check_pitstop()
    check_driver_team()
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
