#!/usr/bin/env python3
"""T051：输出规范化哈希（CSV 文本、报告文本、PNG 字节）。"""
from __future__ import annotations
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis/dirty_air/outputs'
REP=ROOT/'analysis/dirty_air/report.md'
FIG=ROOT/'analysis/dirty_air/figures'
def main():
    for p in sorted(OUT.glob('*.csv')):
        print(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
    for p in [REP]+sorted((OUT/'figures').glob('*.png')):
        print(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
    for p in sorted(FIG.glob('t048_fixed_*.png')):
        print(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
if __name__=='__main__': main()
