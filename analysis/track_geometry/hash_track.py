#!/usr/bin/env python3
"""T048：赛道几何输出规范化哈希（parquet 用内容 CSV）。"""
from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/track_geometry/outputs"
REPORT = ROOT / "analysis/track_geometry/report.md"
def main():
    for p in sorted(OUT.glob("*.parquet")):
        df = pd.read_parquet(p)
        print(f"{hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
    for p in [OUT / "tracks.json", REPORT] + sorted((OUT / "figures").glob("*.png")):
        print(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
if __name__ == "__main__":
    main()
