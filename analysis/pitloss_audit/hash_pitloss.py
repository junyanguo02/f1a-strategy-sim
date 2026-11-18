#!/usr/bin/env python3
"""T042：审计输出规范化哈希（报告去掉生成时间，parquet 用内容 CSV 规范化）。"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/pitloss_audit/outputs"
REPORT = ROOT / "analysis/pitloss_audit/report.md"


def main():
    files = sorted(OUT.glob("*"))
    for p in files:
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
            canon = df.to_csv(index=False)
        elif p.suffix == ".csv":
            canon = p.read_text(encoding="utf-8")
        else:
            continue
        h = hashlib.sha256(canon.encode()).hexdigest()
        print(f"{h}  {p.relative_to(ROOT).as_posix()}")
    text = REPORT.read_text(encoding="utf-8")
    text = re.sub(r"生成时间：.*", "生成时间：<generated_at>", text)
    h = hashlib.sha256(text.encode()).hexdigest()
    print(f"{h}  {REPORT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
