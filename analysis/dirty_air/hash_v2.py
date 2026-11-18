#!/usr/bin/env python3
"""T053：v2 输出与 T048 替换图的规范化哈希。"""
from __future__ import annotations
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis/dirty_air/outputs_v2'
REP=ROOT/'analysis/dirty_air/report_v2.md'
T48=ROOT/'analysis/track_geometry/outputs/figures'
def main():
    for p in sorted(OUT.glob('*.csv')):
        print(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
    for p in [REP]+sorted((OUT/'figures').glob('*.png')):
        print(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
    for p in sorted(T48.glob('*.png'))+sorted((T48/'orig').glob('*.png')):
        print(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
if __name__=='__main__': main()
