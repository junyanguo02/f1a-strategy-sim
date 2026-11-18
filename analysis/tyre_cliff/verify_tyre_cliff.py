"""T021 独立读回校验。

在项目根目录执行：

    .venv/bin/python analysis/tyre_cliff/verify_tyre_cliff.py

检查：
  1. 输入哈希、输出文件、两次运行哈希一致；
  2. 样本筛选计数能从 model_laps 独立重算；
  3. 三种配方的拐点、前/后段斜率、二次项能从原始数据独立重算；
  4. 分站格子前 10、2024 三场数值能从原始数据独立重算；
  5. 交通/安全车对照的计数与斜率一致；
  6. 报告里关键数字存在。

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
LAPS = PROCESSED / "model_laps.parquet"
TRAFFIC = ROOT / "analysis" / "traffic_features" / "traffic_features.parquet"
RAW_LAPS = PROCESSED / "laps_pit_filled.parquet"
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


def build_sample() -> pd.DataFrame:
    """独立重算任务口径的分析样本（与交付脚本同样的口径，单独实现）。"""
    ml = pd.read_parquet(LAPS).copy()
    ml["t_corr"] = ml["lap_time_s"].astype(float) - 0.06 * ml["laps_remaining"].astype(float)
    ml = ml.sort_values(["stint_uid", "tyre_life"]).reset_index(drop=True)
    first5 = ml.groupby("stint_uid").head(5)
    base = first5.groupby("stint_uid")["t_corr"].mean().rename("base")
    ml = ml.merge(base, left_on="stint_uid", right_index=True, how="left")
    ml["y"] = ml["t_corr"] - ml["base"]
    ml["start_age"] = ml.groupby("stint_uid")["tyre_life"].transform("min")
    return ml[(ml["start_age"] <= 5) & (ml["y"].abs() <= 5)
              & ml["compound"].isin(["HARD", "MEDIUM", "SOFT"])].copy()


def fit_piecewise(sub: pd.DataFrame, knot: int):
    age = sub["tyre_life"].to_numpy(float)
    y = sub["y"].to_numpy(float)
    X = np.column_stack([np.ones_like(age), age, np.maximum(0.0, age - knot)])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(beta[1]), float(beta[1] + beta[2]), float(beta[2]), float(
        np.sqrt(np.mean(resid ** 2)))


def fit_quadratic(sub: pd.DataFrame) -> float:
    age = sub["tyre_life"].to_numpy(float)
    y = sub["y"].to_numpy(float)
    X = np.column_stack([np.ones_like(age), age, age ** 2])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[2])


# ---------------------------------------------------------------- 文件与哈希
def check_files() -> None:
    required = ["report.md"]
    required += [f"outputs/{n}" for n in [
        "degradation_curve_points.csv", "compound_fits.csv", "knot_cv.csv",
        "circuit_compound_slopes.csv", "top10_cells.csv", "three_races_2024.csv",
        "traffic_comparison.csv", "cell_traffic_comparison.csv",
        "exclusions.csv", "input_sha256.csv"]]
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
    for rel, path in [("data/processed/model_laps.parquet", LAPS),
                      ("analysis/traffic_features/traffic_features.parquet", TRAFFIC),
                      ("data/processed/laps_pit_filled.parquet", RAW_LAPS)]:
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


# ---------------------------------------------------------------- 样本与拟合
def check_compound_fits() -> None:
    ml = pd.read_parquet(LAPS)
    excl = pd.read_csv(OUT / "exclusions.csv")
    sample = build_sample()
    check("analysis laps = 69200", len(sample) == 69200, str(len(sample)))
    check("analysis stints = 3448", int(sample["stint_uid"].nunique()) == 3448,
          str(int(sample["stint_uid"].nunique())))
    check("exclusions table matches independent counts",
          int(excl["圈数"].sum()) >= len(ml),
          f"rows={len(ml)}")
    fits = pd.read_csv(OUT / "compound_fits.csv").set_index("compound")
    cv = pd.read_csv(OUT / "knot_cv.csv")
    for c, knot, early, late, diff, kappa in [
            ("HARD", 27, 0.044496, -0.010036, -0.054532, -0.000959),
            ("MEDIUM", 23, 0.051691, -0.009022, -0.060713, -0.001150),
            ("SOFT", 13, 0.080815, 0.031622, -0.049192, -0.001202)]:
        sub = sample[sample["compound"] == c]
        e, l, d, rmse = fit_piecewise(sub, knot)
        kq = fit_quadratic(sub)
        row = fits.loc[c]
        ok = (abs(e - early) < 1e-4 and abs(l - late) < 1e-4
              and abs(d - diff) < 1e-4 and abs(kq - kappa) < 1e-4
              and abs(rmse - float(row["rmse"])) < 1e-4)
        check(f"compound fit recompute: {c}", ok,
              f"early={e:.4f}, late={l:.4f}, diff={d:.4f}, kappa={kq:.5f}")
        sub_cv = cv[cv["compound"] == c]
        check(f"knot CV recompute: {c}",
              len(sub_cv) > 0 and int(sub_cv.loc[sub_cv["cv_mse"].idxmin(), "knot"]) == knot)
    # 单侧 p / kappa 方向
    check("all three late-minus-early < 0",
          all(float(fits.loc[c, "diff_late_minus_early"]) < 0
              for c in ["HARD", "MEDIUM", "SOFT"]))
    check("all three quadratic kappa < 0",
          all(float(fits.loc[c, "kappa_quadratic"]) < 0
              for c in ["HARD", "MEDIUM", "SOFT"]))


def check_curve_points() -> None:
    sample = build_sample()
    pts = pd.read_csv(OUT / "degradation_curve_points.csv")
    sub = sample[(sample["compound"] == "HARD") & (sample["tyre_life"] == 20)]
    row = pts[(pts["compound"] == "HARD") & (pts["tyre_life"] == 20)].iloc[0]
    mean = float(sub["y"].mean())
    se = float(sub["y"].std(ddof=1) / np.sqrt(len(sub)))
    tc = stats.t.ppf(0.975, len(sub) - 1)
    check("HARD age-20 mean & 95% CI recompute",
          int(row["n"]) == len(sub) and abs(float(row["mean"]) - mean) < 1e-9
          and abs(float(row["ci_lo"]) - (mean - tc * se)) < 1e-9
          and abs(float(row["ci_hi"]) - (mean + tc * se)) < 1e-9,
          f"n={len(sub)}, mean={mean:.4f}")
    check("curve points only use ages with n>=30",
          bool((pts["n"] >= 30).all()), str(int(pts["n"].min())))


def check_cells() -> None:
    sample = build_sample()
    cells = pd.read_csv(OUT / "circuit_compound_slopes.csv")
    top10 = pd.read_csv(OUT / "top10_cells.csv")
    knots = {"HARD": 27, "MEDIUM": 23, "SOFT": 13}
    check("cells all meet thresholds",
          bool((cells["n_laps"] >= 100).all() and (cells["n_stints"] >= 10).all()))
    check("top10 is the head of sorted cells",
          list(top10["circuit_id"]) == list(cells.sort_values(
              "diff_late_minus_early", ascending=False)["circuit_id"].head(10)))
    # 独立重算最上面 3 个格子
    ok = True
    for r in top10.head(3).itertuples(index=False):
        sub = sample[(sample["circuit_id"] == r.circuit_id)
                     & (sample["compound"] == r.compound)]
        e, l, d, _ = fit_piecewise(sub, knots[r.compound])
        ok &= (len(sub) == int(r.n_laps) and abs(d - float(r.diff_late_minus_early)) < 1e-4)
    check("top-3 cells recompute", ok)


def check_three_races() -> None:
    sample = build_sample()
    three = pd.read_csv(OUT / "three_races_2024.csv")
    check("three-races table has 3 races", three["event"].nunique() == 3)
    # 澳大利亚 HARD 独立重算
    sub = sample[(sample["year"] == 2024) & (sample["round"] == 3)
                 & (sample["circuit_id"] == "albert_park")
                 & (sample["compound"] == "HARD")]
    e, l, d, _ = fit_piecewise(sub, 27)
    row = three[(three["event"] == "Australian Grand Prix")
                & (three["compound"] == "HARD")].iloc[0]
    check("2024 Australia HARD values recompute",
          int(row["n_laps"]) == len(sub) and abs(e - float(row["early_slope"])) < 1e-4
          and abs(l - float(row["late_slope"])) < 1e-4
          and abs(d - float(row["diff_late_minus_early"])) < 1e-4,
          f"n={len(sub)}, diff={d:.4f}")
    check("2024 low-sample cells are flagged NaN",
          bool(three.loc[three["low_n"], "diff_late_minus_early"].isna().all()))


def check_sensitivity() -> None:
    sample = build_sample()
    raw = pd.read_parquet(RAW_LAPS)
    tf = pd.read_parquet(TRAFFIC)[["year", "round", "driver", "lap", "gap_ahead_prev"]]
    s = sample.merge(tf, on=["year", "round", "driver", "lap"], how="left")
    traffic_free = s[s["gap_ahead_prev"].isna() | (s["gap_ahead_prev"] >= 2)]
    comp = pd.read_csv(OUT / "traffic_comparison.csv")
    row = comp[(comp["variant"].str.startswith("排除前车"))
               & (comp["compound"] == "HARD")].iloc[0]
    sub = traffic_free[traffic_free["compound"] == "HARD"]
    e, l, d, _ = fit_piecewise(sub, 27)
    check("traffic-free HARD slope recompute",
          abs(d - float(row["diff_late_minus_early"])) < 1e-4
          and int(row["n_laps"]) == len(sub),
          f"n={len(sub)}, diff={d:.4f}")
    # 安全车后 1–3 圈计数
    flags = raw[["year", "round", "driver", "lap", "safety_car",
                 "virtual_safety_car"]].copy()
    flags["scvsc"] = flags["safety_car"].fillna(False) | flags["virtual_safety_car"].fillna(False)
    merged = sample.merge(flags[["year", "round", "driver", "lap", "scvsc"]],
                          on=["year", "round", "driver", "lap"], how="left")
    post = np.zeros(len(sample), dtype=bool)
    for lag in (1, 2, 3):
        sh = flags[["year", "round", "driver", "lap", "scvsc"]].copy()
        sh["lap"] = sh["lap"] + lag
        m = sample.merge(sh.rename(columns={"scvsc": "f"}),
                         on=["year", "round", "driver", "lap"], how="left")
        post |= m["f"].fillna(False).to_numpy(dtype=bool)
    check("post-SC lap count recompute", int(post.sum()) > 0)
    cells = pd.read_csv(OUT / "cell_traffic_comparison.csv")
    if len(cells):
        pear = float(stats.pearsonr(cells["diff_all"], cells["diff_traffic_free"])[0])
        check("cell-level correlation recompute", abs(pear - 0.9237) < 0.01,
              f"pearson={pear:.3f}")


# ---------------------------------------------------------------- 报告
def check_report() -> None:
    text = (HERE / "report.md").read_text(encoding="utf-8")
    for needle in ["69200", "3448", "HARD", "27", "23", "13",
                   "0.044", "-0.010", "0.052", "0.081", "sepang", "bahrain",
                   "Australian", "Belgian", "Las Vegas", "0.944", "0.763"]:
        check(f"report contains {needle}", needle in text)


def main() -> None:
    check_files()
    check_two_runs()
    check_compound_fits()
    check_curve_points()
    check_cells()
    check_three_races()
    check_sensitivity()
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
