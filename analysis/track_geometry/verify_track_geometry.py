#!/usr/bin/env python3
"""T048 独立校验：5 场中心线、长度误差、曲率/半径、参考速度、tracks.json、两次运行。"""
from __future__ import annotations
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs"
CACHE = ROOT / "data" / "raw" / "openf1_track"
LOG = HERE / "logs"
FAIL, PASS = [], []
RACES = {"2024-1": 5412.0, "2024-3": 5278.0, "2024-10": 4657.0, "2024-13": 4381.0, "2024-16": 5793.0}
COLS = ["s_m", "x_m", "y_m", "curvature_1pm", "speed_ref_kmh_mean",
        "speed_ref_kmh_min", "speed_ref_kmh_max", "throttle_mean", "brake_frac"]


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(("PASS: " if ok else "FAIL: ") + name + (f" — {detail}" if detail else ""))


def main():
    tracks = json.loads((OUT / "tracks.json").read_text(encoding="utf-8"))
    check("tracks.json has 5 tracks", len(tracks["tracks"]) == 5)
    check("tracks units/source", "units" in tracks and "source" in tracks)
    by = {t["race_id"]: t for t in tracks["tracks"]}
    check("race ids", set(by) == set(RACES))
    for rid, official in RACES.items():
        t = by[rid]
        p = OUT / f"{rid}.parquet"
        check(f"{rid} parquet exists", p.exists())
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        check(f"{rid} columns exact", list(d.columns) == COLS)
        N = int(t["n_points"])
        check(f"{rid} rows N", len(d) == N, f"{len(d)}/{N}")
        check(f"{rid} s grid 5m", abs(float(d.s_m.iloc[1] - d.s_m.iloc[0]) - 5) < 1e-6
              and abs(float(d.s_m.iloc[-1]) - 5 * (N - 1)) < 1e-6)
        L = 0.0
        x = d.x_m.to_numpy(); y = d.y_m.to_numpy()
        L = float(np.sum(np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]]))))
        check(f"{rid} length = N*5", abs(L - N * 5) < 0.2, f"{L:.2f}")
        err = (L / official - 1) * 100
        check(f"{rid} length error <=1.5%", abs(err) <= 1.5, f"{err:+.3f}%")
        check(f"{rid} tracks length matches", abs(t["centerline_length_m"] - L) < 0.2)
        k = d.curvature_1pm.to_numpy()
        rmin = 1.0 / max(float(np.max(np.abs(k))), 1e-12)
        check(f"{rid} min radius", abs(rmin - t["min_radius_m"]) < 0.2, f"{rmin:.1f} vs {t['min_radius_m']:.1f}")
        for c in ["speed_ref_kmh_mean", "speed_ref_kmh_min", "speed_ref_kmh_max",
                  "throttle_mean", "brake_frac"]:
            check(f"{rid} finite {c}", np.isfinite(d[c]).all())
        check(f"{rid} speed ordering", (d.speed_ref_kmh_min <= d.speed_ref_kmh_mean + 1e-6).all()
              and (d.speed_ref_kmh_mean <= d.speed_ref_kmh_max + 1e-6).all())
        check(f"{rid} speed range", d.speed_ref_kmh_min.min() > 20 and d.speed_ref_kmh_max.max() < 400,
              f"{d.speed_ref_kmh_min.min():.0f}-{d.speed_ref_kmh_max.max():.0f}")
        check(f"{rid} throttle range", d.throttle_mean.min() >= -1 and d.throttle_mean.max() <= 101)
        check(f"{rid} brake frac", d.brake_frac.min() >= 0 and d.brake_frac.max() <= 1)
        check(f"{rid} ref lap", 60 < t["reference_lap_time_s"] < 180, str(t["reference_lap_time_s"]))
        check(f"{rid} race laps", t["race_laps"] in (53, 57, 58, 66, 70), str(t["race_laps"]))
        sel = t["selected_laps"]
        check(f"{rid} five selected laps", len(sel) == 5 and len({x["driver_number"] for x in sel}) == 5)
        for x in sel:
            tag = f"session{t['session_key']}_driver{int(x['driver_number'])}_lap{int(x['lap'])}"
            check(f"{rid} cache {x['driver']} {x['lap']}",
                  (CACHE / f"location_{tag}.json").exists() and (CACHE / f"car_data_{tag}.json").exists())
        check(f"{rid} figure", (OUT / "figures" / f"{rid}.png").stat().st_size > 1000)
    cache = list(CACHE.glob("*.json"))
    check("cache 80 files", len(cache) == 80, str(len(cache)))
    check("cache bytes > 3MB", sum(p.stat().st_size for p in cache) > 3_000_000)
    report = (HERE / "report.md").read_text(encoding="utf-8")
    for needle in ["T048", "长度误差", "最小半径", "参考圈时", "平滑", "这些数据不能说明什么", "1.05"]:
        check(f"report contains {needle}", needle in report)
    r1 = LOG / "run1.sha256"; r2 = LOG / "run2.sha256"
    check("two-run logs", r1.exists() and r2.exists())
    if r1.exists() and r2.exists():
        check("two runs identical", r1.read_text().strip() == r2.read_text().strip())
        check("two_runs marker", (LOG / "two_runs.txt").exists()
              and "TWO RUNS IDENTICAL" in (LOG / "two_runs.txt").read_text())
        # 当前输出 == run2
        cur = []
        for p in sorted(OUT.glob("*.parquet")):
            df = pd.read_parquet(p)
            cur.append(f"{hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
        for p in [OUT / "tracks.json", HERE / "report.md"] + sorted((OUT / "figures").glob("*.png")):
            cur.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
        check("current matches run2", "\n".join(cur) == r2.read_text().strip(), f"{len(cur)} files")
    print()
    print(f"TOTAL: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print(" -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
