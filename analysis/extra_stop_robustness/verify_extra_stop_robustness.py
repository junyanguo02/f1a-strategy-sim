"""T032 独立读回校验。

在项目根目录执行：

    .venv/bin/python analysis/extra_stop_robustness/verify_extra_stop_robustness.py

检查：
  1. 输入哈希、输出文件、两次运行哈希一致；
  2. 主效应/对照的整群自助区间可重算；
  3. 排除被迫后的调整回归、置换检验、分样本、名次差、剔除极端值可重算；
  4. 报告里关键数字存在。

只读，不写任何结果文件。
"""
from __future__ import annotations

import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
T028 = ROOT / "analysis" / "teammate_outcomes" / "outputs" / "teammate_pairs.csv"
T031 = ROOT / "analysis" / "extra_stop" / "outputs" / "pair_metrics.csv"
JOLPICA = ROOT / "data" / "raw" / "jolpica"
OUT = HERE / "outputs"
FIG = HERE / "figures"
LOG = HERE / "logs"
FORCED = {"轮胎衰减更快（被迫）", "停车异常（被迫）"}
SEED = 20261117
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


def cluster_bootstrap(df, col, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    races = df["race_key"].drop_duplicates().to_numpy()
    groups = {r: df.loc[df["race_key"] == r, col].to_numpy(float) for r in races}
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(races, size=len(races), replace=True)
        out[b] = np.concatenate([groups[r] for r in pick]).mean()
    return out


def load_analysis():
    pairs = pd.read_csv(T028)
    keep = ((~pairs["same_stops"]) & (pairs["stop_diff"] == 1)
            & pairs["extra_stop_sc"].isin(["绿旗下", "SC/VSC 下"])) | pairs["same_stops"].astype(bool)
    pairs = pairs[keep].copy()
    t31 = pd.read_csv(T031)[["year", "round", "team", "driver_more",
                             "driver_less", "group"]]
    pairs = pairs.merge(t31, on=["year", "round", "team", "driver_more", "driver_less"],
                        how="left")
    green = pairs[pairs["extra_stop_sc"] == "绿旗下"]
    ctrl = pairs[pairs["extra_stop_sc"] == "SC/VSC 下"]
    return pairs, green, ctrl


def check_files() -> None:
    required = ["report.md"]
    required += [f"outputs/{n}" for n in [
        "summary.csv", "subsamples.csv", "bootstrap_draws.csv",
        "permutation_null.csv", "pairs_analysis.csv", "input_sha256.csv"]]
    for rel in required:
        p = HERE / rel
        check(f"output exists: {rel}", p.exists() and p.stat().st_size > 0,
              f"{p.stat().st_size} bytes" if p.exists() else "missing")
    figs = sorted(p.name for p in FIG.glob("*.png"))
    check("two figures", len(figs) == 2, ", ".join(figs))
    for name in figs:
        p = FIG / name
        check(f"figure non-empty: {name}", p.stat().st_size > 10_000,
              f"{p.stat().st_size} bytes")
    hashes = pd.read_csv(OUT / "input_sha256.csv")
    for rel, path in [("analysis/teammate_outcomes/outputs/teammate_pairs.csv", T028),
                      ("analysis/extra_stop/outputs/pair_metrics.csv", T031)]:
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


def check_effects() -> None:
    pairs, green, ctrl = load_analysis()
    check("green = 82", len(green) == 82, str(len(green)))
    check("SC = 26", len(ctrl) == 26, str(len(ctrl)))
    check("same-stop controls = 367", int(pairs["same_stops"].sum()) == 367)
    gb = cluster_bootstrap(green, "time_diff")
    cb = cluster_bootstrap(ctrl, "time_diff")
    check("green mean = 16.43", abs(green["time_diff"].mean() - 16.43056) < 1e-4,
          f"{green['time_diff'].mean():.4f}")
    check("green bootstrap CI",
          abs(np.quantile(gb, 0.025) - 8.77) < 0.02
          and abs(np.quantile(gb, 0.975) - 25.02) < 0.02,
          f"[{np.quantile(gb, 0.025):.2f}, {np.quantile(gb, 0.975):.2f}]")
    check("SC mean = -11.61", abs(ctrl["time_diff"].mean() - (-11.60596)) < 1e-4,
          f"{ctrl['time_diff'].mean():.4f}")
    check("SC bootstrap CI",
          abs(np.quantile(cb, 0.025) - (-28.37)) < 0.05
          and abs(np.quantile(cb, 0.975) - 3.29) < 0.05,
          f"[{np.quantile(cb, 0.025):.2f}, {np.quantile(cb, 0.975):.2f}]")
    forced = green[green["group"].isin(FORCED)]
    check("forced groups = 23", len(forced) == 23, str(len(forced)))
    rest = green[~green["group"].isin(FORCED)]
    check("excluded-forced mean = 15.20", abs(rest["time_diff"].mean() - 15.19971) < 1e-4,
          f"{rest['time_diff'].mean():.3f}")


def check_regression() -> None:
    pairs, green, ctrl = load_analysis()
    rest = green[~green["group"].isin(FORCED)]
    same = pairs[pairs["same_stops"]].copy()

    def fit(diff_df, col):
        pooled = pd.concat([diff_df.assign(group=1), same.assign(group=0)], ignore_index=True)
        pooled = pooled[pooled[col].notna()]
        X = np.column_stack([np.ones(len(pooled)), pooled["group"].to_numpy(float),
                             pooled["grid_diff"].to_numpy(float),
                             pooled["pace_diff"].to_numpy(float)])
        return ols_cluster(pooled[col].to_numpy(float), X, pooled["race_key"].to_numpy())

    b, se, G = fit(rest, "time_diff")
    check("adjusted time-diff coef = 10.33", abs(b[1] - 10.33055) < 1e-4,
          f"{b[1]:.3f} (SE {se[1]:.3f})")
    check("adjusted time-diff SE = 3.62", abs(se[1] - 3.62) < 0.02, f"{se[1]:.3f}")
    bp, sep, Gp = fit(rest, "pos_diff")
    check("adjusted pos coef = 1.33", abs(bp[1] - 1.33144) < 1e-4,
          f"{bp[1]:.3f} (SE {sep[1]:.3f})")
    check("adjusted pos SE = 0.43", abs(sep[1] - 0.43) < 0.02, f"{sep[1]:.3f}")


def check_permutation() -> None:
    _, green, _ = load_analysis()
    y = green["time_diff"].to_numpy(float)
    race = green["race_key"].to_numpy()
    rng = np.random.default_rng(SEED)
    obs = float(y.mean())
    perm = np.empty(2000)
    for b in range(2000):
        signs = np.where(rng.random(len(y)) < 0.5, -1.0, 1.0)
        vals = y * signs
        perm[b] = np.mean([vals[race == r].mean() for r in np.unique(race)])
    p = (1 + np.sum(perm >= obs)) / (1 + 2000)
    check("permutation p ≈ 0.0005", abs(p - 0.00049975) < 1e-6, f"{p:.6f}")
    check("permutation null mean ≈ 0.01", abs(perm.mean() - 0.01237) < 0.02,
          f"{perm.mean():.3f}")


def check_subsamples() -> None:
    subs = pd.read_csv(OUT / "subsamples.csv")
    _, green, _ = load_analysis()
    # 赛季
    for season in [2023, 2024, 2025, 2026]:
        s = green[green["year"] == season]
        row = subs[(subs["维度"] == "赛季") & (subs["取值"].astype(str) == str(season))]
        check(f"season {season} mean/n", len(row) == 1
              and abs(float(row["用时差均值 (s)"].iloc[0]) - s["time_diff"].mean()) < 1e-6
              and int(row["n"].iloc[0]) == len(s), f"n={len(s)}")
    check("subsamples has 8 rows", len(subs) == 8, str(len(subs)))


def check_position_and_trim() -> None:
    _, green, ctrl = load_analysis()
    gpos = green[green["pos_diff"].notna()]
    check("green pos mean = 2.2568", abs(gpos["pos_diff"].mean() - 2.25676) < 1e-3,
          f"{gpos['pos_diff'].mean():.3f}")
    gb = cluster_bootstrap(gpos, "pos_diff")
    check("green pos bootstrap CI",
          abs(np.quantile(gb, 0.025) - 1.15) < 0.05
          and abs(np.quantile(gb, 0.975) - 3.34) < 0.05,
          f"[{np.quantile(gb, 0.025):.2f}, {np.quantile(gb, 0.975):.2f}]")
    # 剔除 5%
    absdiff = green["time_diff"].abs()
    n_drop = max(1, int(round(0.05 * len(green))))
    drop = absdiff.nlargest(n_drop).index
    kept = green.drop(index=drop)
    check("trim n = 78", len(kept) == 78, str(len(kept)))
    check("trim mean = 13.96", abs(kept["time_diff"].mean() - 13.96412) < 1e-4,
          f"{kept['time_diff'].mean():.2f}")


def check_report() -> None:
    text = (HERE / "report.md").read_text(encoding="utf-8")
    for needle in ["16.43", "8.77", "25.02", "-11.61", "15.20", "10.33",
                   "3.62", "<0.001", "2.26", "1.33", "13.96", "78", "426"]:
        check(f"report contains {needle}", needle in text)


def main() -> None:
    check_files()
    check_two_runs()
    check_effects()
    check_regression()
    check_permutation()
    check_subsamples()
    check_position_and_trim()
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
