#!/usr/bin/env python3
"""T042 独立校验：clean parquet、极端分类、池统计、回退清单、两次运行。

在项目根目录执行：
    .venv/bin/python analysis/pitloss_audit/verify_pitloss_audit.py
只读，不写结果。
"""
from __future__ import annotations
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1a.sim.race import pit_tables  # noqa: E402

OUT = HERE / "outputs"
LOG = HERE / "logs"
ORIG = ROOT / "analysis/fcy_inputs/pit_loss_relative.parquet"
DET = ROOT / "analysis/sc_pit_timing/outputs/pit_match_detail.csv"
FAILS, PASSES = [], []


def check(name, ok, detail=""):
    (PASSES if ok else FAILS).append(name)
    print(("PASS: " if ok else "FAIL: ") + name + (f" — {detail}" if detail else ""))


def canon(o):
    return hashlib.sha256(o.encode()).hexdigest()


def main():
    orig = pd.read_parquet(ORIG)
    clean = pd.read_parquet(OUT / "pit_loss_relative_clean.parquet")
    det = pd.read_csv(DET)
    ext = pd.read_csv(OUT / "extreme_classification.csv")
    normal = pd.read_csv(OUT / "pool_stats_normal.csv")
    fcy = pd.read_csv(OUT / "pool_stats_fcy.csv")
    fb = pd.read_csv(OUT / "fallback_races.csv")
    check("original 3402x35", orig.shape == (3402, 35), str(orig.shape))
    check("original untouched (no clean cols)", "clean_keep" not in orig.columns)
    want_cols = list(orig.columns) + ["clean_keep", "clean_reason"]
    check("clean columns = original + clean_keep/clean_reason", list(clean.columns) == want_cols)
    check("clean rows 3402", len(clean) == 3402)
    check("clean_keep bool non-null", clean.clean_keep.dtype == bool and clean.clean_keep.notna().all())
    check("clean keys unique", not clean.duplicated(["year", "round", "driver", "lap"]).any())
    check("clean reasons non-empty", clean.clean_reason.astype(str).str.len().gt(0).all())
    check("det keys all decided", set(det[["year", "round", "driver", "lap"]].itertuples(index=False)) <=
          set(clean[["year", "round", "driver", "lap"]].itertuples(index=False)))
    check("clean_keep true count 2380", int(clean.clean_keep.sum()) == 2380, str(int(clean.clean_keep.sum())))

    # 池统计独立重算
    n = clean[clean.relative_usable & (clean.condition == "正常")]
    b, a = n.relative_loss_s, n.loc[n.clean_keep, "relative_loss_s"]
    row = normal[normal.pool == "normal:ALL"].iloc[0]
    check("normal ALL n before/after", int(row.n_before) == len(b) and int(row.n_after) == len(a),
          f"{len(b)}/{len(a)}")
    check("normal ALL mean before/after", abs(row.mean_before - b.mean()) < 1e-6
          and abs(row.mean_after - a.mean()) < 1e-6, f"{b.mean():.3f}/{a.mean():.3f}")
    check("normal ALL mean after ~23.5", 23.3 < a.mean() < 23.7, f"{a.mean():.3f}")
    check("normal per-circuit rows", len(normal) == 1 + n.circuit_id.nunique())
    # FCY
    flags = clean.set_index(["year", "round", "driver", "lap"]).clean_keep
    d = det[det.dt_group.notna() & det.relative_loss_s.notna()].copy()
    d["keep"] = flags.reindex(pd.MultiIndex.from_frame(d[["year", "round", "driver", "lap"]])).to_numpy()
    check("det clean_keep no missing", d.keep.notna().all())
    groups = {1: ("宣布后0-60秒", "60秒-1圈"), 2: ("1-2圈",), 3: ("2圈以后",)}
    for t in ("SC", "VSC"):
        for ph, gs in groups.items():
            s = d[(d.condition == t) & d.dt_group.isin(gs)]
            row = fcy[fcy.pool == f"fcy:{t}:ph{ph}"].iloc[0]
            ok = (int(row.n_before) == len(s) and abs(row.mean_before - s.relative_loss_s.mean()) < 1e-6
                  and int(row.n_after) == int(s.keep.sum())
                  and abs(row.mean_after - s.loc[s.keep, "relative_loss_s"].mean()) < 1e-6)
            check(f"fcy pool {t} ph{ph}", ok)

    # 分类计数
    check("extreme rows 299", len(ext) == 299, str(len(ext)))
    counts = ext.groupby("extreme_category").size().to_dict()
    expected = {"雨战/换雨胎": 78, "慢进站（保留）": 48, "其他": 165,
                "参考圈问题": 3, "红旗/暂停": 3, "罚时/通过维修区": 2}
    check("category counts", counts == expected, str(counts))
    # 极端 normal：只有慢进站/FCY? normal extreme kept only 慢进站
    nx = ext[ext.subset == "normal"]
    check("normal extreme kept iff slow",
          all((r.extreme_category == "慢进站（保留）") == bool(r.clean_keep) for r in nx.itertuples(index=False)))
    check("normal extreme kept n=20", int(nx.clean_keep.sum()) == 20, str(int(nx.clean_keep.sum())))
    # 雨/红旗/罚时/参考圈全部删除
    for c in ["雨战/换雨胎", "红旗/暂停", "罚时/通过维修区", "参考圈问题"]:
        s = ext[ext.extreme_category == c]
        check(f"category dropped {c}", not s.clean_keep.any())
    # FCY 极端保留（除数据问题）
    fcy_ext = ext[ext.subset.isin(["SC", "VSC"])]
    check("SC/VSC extreme kept unless data problem",
          all(bool(r.clean_keep) == (r.extreme_category == "其他" or r.extreme_category == "慢进站（保留）")
              for r in fcy_ext.itertuples(index=False)))
    # 回退
    check("fallback rows 81", len(fb) == 81, str(len(fb)))
    check("fallback pre counts 26/29", int(fb.pre_fallback_before.sum()) == 26
          and int(fb.pre_fallback_after.sum()) == 29,
          f"{fb.pre_fallback_before.sum()}/{fb.pre_fallback_after.sum()}")
    check("fallback post counts 0/1", int(fb.post_fallback_before.sum()) == 0
          and int(fb.post_fallback_after.sum()) == 1,
          f"{fb.post_fallback_before.sum()}/{fb.post_fallback_after.sum()}")
    # race.py clean 模式可读
    pl2, det2 = pit_tables(ROOT, "clean")
    check("pit_tables(clean) works", len(pl2) == int(clean.clean_keep.sum()) and len(det2) > 0,
          f"{len(pl2)}/{len(det2)}")
    # 原文件哈希与 report 输入哈希一致
    report = (HERE / "report.md").read_text(encoding="utf-8")
    h_orig = hashlib.sha256(ORIG.read_bytes()).hexdigest()
    check("report records original hash", h_orig in report)
    # 两次运行
    r1, r2 = LOG / "run1.sha256", LOG / "run2.sha256"
    check("two-run logs exist", r1.exists() and r2.exists())
    check("two runs identical", r1.read_text().strip() == r2.read_text().strip())
    check("two_runs marker", (LOG / "two_runs.txt").exists()
          and "TWO RUNS IDENTICAL" in (LOG / "two_runs.txt").read_text())
    # 报告关键内容
    for needle in ["T042", "雨战/换雨胎", "慢进站（保留）", "参考圈问题",
                   "normal:ALL", "23.52", "回退", "这些数据不能说明什么"]:
        check(f"report contains {needle}", needle in report)
    print()
    print(f"TOTAL: {len(PASSES)} passed, {len(FAILS)} failed")
    if FAILS:
        for f in FAILS:
            print(" -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
