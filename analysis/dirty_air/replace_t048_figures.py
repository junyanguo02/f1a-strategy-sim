#!/usr/bin/env python3
"""T053：备份 T048 原图到 figures/orig/，替换为带标题/轴标题/色标的版本。"""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'analysis/dirty_air/figures'
DST=ROOT/'analysis/track_geometry/outputs/figures'
ORIG=DST/'orig'
def main():
    ORIG.mkdir(parents=True,exist_ok=True)
    for rid in ['2024-1','2024-3','2024-10','2024-13','2024-16']:
        src=SRC/f't048_fixed_{rid}.png'; dst=DST/f'{rid}.png'
        if not src.exists(): raise FileNotFoundError(src)
        if dst.exists() and not (ORIG/dst.name).exists():
            shutil.copy2(dst,ORIG/dst.name)
        shutil.copy2(src,dst)
        print('replaced',dst,'backup',ORIG/dst.name)
if __name__=='__main__': main()
