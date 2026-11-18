"""T024 独立读回校验。

在项目根目录执行：

    .venv/bin/python analysis/tyre_management/verify_tyre_management.py

检查：
  1. 输入哈希、输出文件、两次运行哈希一致；
  2. 主样本计数、段长口径能从原始数据独立重算；
  3. 分箱表、主回归、稳健性/对照版本、分站格子、2024 三场可重算；
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
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROCESSED = ROOT / "data" / "processed"
MODEL = PROCESSED / "model_laps.parquet"
STINTS = ROOT / "analysis" / "actual_strategies" / "stints.csv"
LAPS = PROCESSED / "laps_pit_filled.parquet"
EPISODES = PROCESSED / "fcy_episodes.parquet"
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


def build_sample():
    ml = pd.read_parquet(MODEL)
    st = pd.read_csv(STINTS).rename(columns={"stint_number": "stint"})
    lf = pd.read_parquet(LAPS)
    for c in ["pit_in_filled", "safety_car", "virtual_safety_car"]:
        lf[c] = lf[c].fillna(False).astype(bool)
    ml["t_corr"] = ml["lap_time_s"].astype(float) - 0.06 * ml["laps_remaining"].astype(float)
    stm = st[["year", "round", "driver", "stint", "lap_start", "lap_end"]].copy()
    stm["length"] = stm["lap_end"] - stm["lap_start"] + 1
    mx = stm.groupby(["year", "round", "driver"])["stint"].max().rename("max_stint").reset_index()
    stm = stm.merge(mx, on=["year", "round", "driver"], how="left")
    stm["is_last"] = stm["stint"] == stm["max_stint"]
    fl = lf[["year", "round", "driver", "lap", "pit_in_filled",
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
    stm["sc_short"] = stm["pit_end"].fillna(False) & (~stm["is_last"]) & stm["sc_end"].fillna(False)
    ml = ml.merge(stm[["year", "round", "driver", "stint", "length", "is_last", "sc_short"]],
                  on=["year", "round", "driver", "stint"], how="left")
    ml["dr_key"] = ml["year"].astype(str) + "|" + ml["round"].astype(str) + "|" + ml["driver"]
    sample = ml[(ml["length"].notna()) & (~ml["is_last"].fillna(False))
                & (~ml["sc_short"].fillna(False))].copy()
    sample["dr_mean"] = sample.groupby("dr_key")["t_corr"].transform("mean")
    sample["y"] = sample["t_corr"] - sample["dr_mean"]
    return ml, sample


def build_design(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["len10"] = df["length"].to_numpy(float) / 10.0
    X["age"] = df["tyre_life"].to_numpy(float)
    for c in ["MEDIUM", "SOFT"]:
        X[f"age_x_{c}"] = X["age"] * (df["compound"] == c).astype(float)
    cc = pd.get_dummies(df["circuit_id"].astype(str) + "|" + df["compound"].astype(str),
                        prefix="cc", drop_first=True, dtype=float)
    return pd.concat([X, cc], axis=1)


def within(mat: pd.DataFrame, groups: pd.Series) -> np.ndarray:
    m = mat.to_numpy(float)
    g = groups.astype(str).to_numpy()
    uniq, inv = np.unique(g, return_inverse=True)
    cnt = np.bincount(inv)
    out = m.copy()
    for j in range(m.shape[1]):
        out[:, j] -= (np.bincount(inv, weights=m[:, j]) / cnt)[inv]
    return out


def fit_fe(df: pd.DataFrame, fe: str = "dr_key"):
    X = build_design(df)
    Xw = within(X, df[fe])
    yw = within(df[["t_corr"]], df[fe])[:, 0]
    keep = Xw.std(axis=0) > 1e-9
    Xw = Xw[:, keep]
    names = [c for c, k in zip(X.columns, keep) if k]
    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    XtXi = np.linalg.pinv(Xw.T @ Xw)
    g = df["stint_uid"].astype(str).to_numpy()
    uniq, inv = np.unique(g, return_inverse=True)
    scores = np.zeros((len(uniq), Xw.shape[1]))
    np.add.at(scores, inv, Xw * resid[:, None])
    V = XtXi @ (scores.T @ scores * len(uniq) / max(len(uniq) - 1, 1)) @ XtXi
    i = names.index("len10")
    se = float(np.sqrt(max(V[i, i], 0)))
    return float(beta[i]), se, len(df), len(uniq)


def fit_cell(df: pd.DataFrame):
    y = df["y"].to_numpy(float)
    X = np.column_stack([np.ones(len(df)), df["length"].to_numpy(float) / 10.0,
                         df["tyre_life"].to_numpy(float)])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtXi = np.linalg.pinv(X.T @ X)
    g = df["stint_uid"].astype(str).to_numpy()
    uniq, inv = np.unique(g, return_inverse=True)
    scores = np.zeros((len(uniq), X.shape[1]))
    np.add.at(scores, inv, X * resid[:, None])
    V = XtXi @ (scores.T @ scores * len(uniq) / max(len(uniq) - 1, 1)) @ XtXi
    return float(beta[1]), float(np.sqrt(max(V[1, 1], 0)))


# ---------------------------------------------------------------- 文件与哈希
def check_files() -> None:
    required = ["report.md"]
    required += [f"outputs/{n}" for n in [
        "length_bins.csv", "regression_main.csv", "robustness.csv",
        "cells.csv", "race_2024_cells.csv", "stint_meta.csv", "input_sha256.csv"]]
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
                      ("data/processed/laps_pit_filled.parquet", LAPS),
                      ("data/processed/fcy_episodes.parquet", EPISODES)]:
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


# ---------------------------------------------------------------- 数据核对
def check_sample(ml: pd.DataFrame, sample: pd.DataFrame) -> None:
    check("main laps = 34675", len(sample) == 34675, str(len(sample)))
    check("main stints = 1962", sample["stint_uid"].nunique() == 1962,
          str(sample["stint_uid"].nunique()))
    last = ml.loc[ml["length"].notna() & ml["is_last"].fillna(False), "stint_uid"].nunique()
    sc = ml.loc[ml["length"].notna() & ml["sc_short"].fillna(False), "stint_uid"].nunique()
    check("excluded final stints = 1510", last == 1510, str(last))
    check("excluded SC-shortened stints = 485", sc == 485, str(sc))
    check("length equals lap_end-lap_start+1",
          bool((ml["length"].dropna() >= 1).all()))
    # 观测段长系统性更短（主样本口径）
    obs = sample.groupby("stint_uid")["tyre_life"].agg(["min", "max"]).reset_index()
    obs["obs_len"] = obs["max"] - obs["min"] + 1
    m = sample.drop_duplicates("stint_uid")[["stint_uid", "length"]].merge(obs, on="stint_uid")
    diff = (m["obs_len"] - m["length"])
    check("observed length median diff ≈ -3", abs(float(diff.median()) - (-3.0)) < 0.5,
          f"{diff.median():.1f}")


def check_bins(sample: pd.DataFrame) -> None:
    bins = pd.read_csv(OUT / "length_bins.csv")
    # 独立重算：中期 11–20 圈 HARD 31–40
    s = sample[(sample["tyre_life"].between(11, 20)) & (sample["compound"] == "HARD")].copy()
    s["cell"] = s["circuit_id"].astype(str) + "|" + s["compound"].astype(str) + "|" + s["tyre_life"].astype(str)
    s["z"] = s["y"] - s.groupby("cell")["y"].transform("mean")
    b = s[(s["length"] >= 31) & (s["length"] <= 40)]
    expect = float(b["z"].mean())
    row = bins[(bins["window"] == "中期 11–20 圈") & (bins["compound"] == "HARD")
               & (bins["length_bin"] == "31–40")].iloc[0]
    check("mid HARD 31-40 bin recompute", abs(float(row["mean_dev_s"]) - expect) < 1e-9,
          f"{expect:.4f}")
    check("bin rows = 2 windows x 4 compound x 6 bins = 48", len(bins) == 48, str(len(bins)))


def check_regression(ml: pd.DataFrame, sample: pd.DataFrame) -> None:
    reg = pd.read_csv(OUT / "regression_main.csv")
    rob = pd.read_csv(OUT / "robustness.csv")
    coef, se, n, G = fit_fe(sample)
    row = reg[reg["variant"].str.startswith("主回归（计划段长")].iloc[0]
    check("main regression coefficient recompute",
          abs(float(row["coef_len10"]) - coef) < 1e-6
          and abs(float(row["se"]) - se) < 1e-6 and int(row["n_laps"]) == n,
          f"{coef:.4f} (SE {se:.4f}, n={n})")
    check("main coefficient negative", coef < 0 and abs(coef) > 2 * se)
    # 对照：不排除最后一段（独立重算）
    s_last = ml[(ml["length"].notna()) & (~ml["sc_short"].fillna(False))].copy()
    s_last["dr_mean"] = s_last.groupby("dr_key")["t_corr"].transform("mean")
    s_last["y"] = s_last["t_corr"] - s_last["dr_mean"]
    c2, se2, n2, _ = fit_fe(s_last)
    row2 = rob[rob["variant"].str.startswith("对照：不排除比赛最后一段")].iloc[0]
    check("control include-final-stints recompute",
          abs(float(row2["coef_len10"]) - c2) < 1e-6 and abs(float(row2["se"]) - se2) < 1e-6,
          f"{c2:.4f} (SE {se2:.4f})")
    # 只用无 SC/VSC 比赛
    ep = pd.read_parquet(EPISODES)
    sc = set(map(tuple, ep[ep["type"].isin(["SC", "VSC"])][["year", "round"]]
                 .drop_duplicates().values))
    nosc = sample[[(int(y), int(r)) not in sc for y, r in zip(sample["year"], sample["round"])]]
    c3, se3, n3, _ = fit_fe(nosc)
    row3 = rob[rob["variant"].str.startswith("稳健性：只用无 SC")].iloc[0]
    check("no-SC races recompute",
          abs(float(row3["coef_len10"]) - c3) < 1e-6 and int(row3["n_laps"]) == n3,
          f"{c3:.4f} (n={n3})")
    check("regression rows (main) = 3", len(reg) == 3, str(len(reg)))
    check("robustness rows = 6", len(rob) == 6, str(len(rob)))


def check_cells(sample: pd.DataFrame) -> None:
    cells = pd.read_csv(OUT / "cells.csv")
    check("cells = 54", len(cells) == 54, str(len(cells)))
    for circuit, comp, expect in [("losail", "HARD", 0.3186), ("spa", "SOFT", -1.0338)]:
        s = sample[(sample["circuit_id"] == circuit) & (sample["compound"] == comp)]
        c, se = fit_cell(s)
        row = cells[(cells["circuit_id"] == circuit) & (cells["compound"] == comp)].iloc[0]
        check(f"cell recompute {circuit}-{comp}",
              abs(float(row["coef_len10"]) - c) < 1e-6 and abs(c - expect) < 5e-3,
              f"{c:.4f}")
    check("cells all meet thresholds",
          bool((cells["n_laps"] >= 100).all() and (cells["n_stints"] >= 10).all()))
    rc = pd.read_csv(OUT / "race_2024_cells.csv")
    check("2024 races rows = 9", len(rc) == 9, str(len(rc)))
    for circuit, comp, expect in [("albert_park", "HARD", -0.1370),
                                  ("vegas", "MEDIUM", -0.8792)]:
        s = sample[(sample["year"] == 2024) & (sample["circuit_id"] == circuit)
                   & (sample["compound"] == comp)]
        c, se = fit_cell(s)
        row = rc[(rc["circuit_id"] == circuit) & (rc["compound"] == comp)].iloc[0]
        check(f"2024 cell recompute {circuit}-{comp}",
              abs(float(row["coef_len10"]) - c) < 1e-6 and abs(c - expect) < 5e-3,
              f"{c:.4f}")


def check_report() -> None:
    text = (HERE / "report.md").read_text(encoding="utf-8")
    for needle in ["34675", "1962", "1510", "485", "-0.092", "0.031",
                   "losail", "spa", "澳大利亚", "比利时", "拉斯维加斯",
                   "0.009", "-0.133", "无 SC/VSC"]:
        check(f"report contains {needle}", needle in text)


def main() -> None:
    check_files()
    check_two_runs()
    ml, sample = build_sample()
    check_sample(ml, sample)
    check_bins(sample)
    check_regression(ml, sample)
    check_cells(sample)
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
