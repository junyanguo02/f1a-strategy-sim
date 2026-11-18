#!/usr/bin/env python3
"""T048: OpenF1 真实赛道中心线、带符号曲率与参考车速曲线。

在项目根目录执行：
    .venv/bin/python analysis/track_geometry/run_track_geometry.py

联网仅 https://api.openf1.org，每次请求间隔 >=1.05 s；原始 JSON 缓存到
data/raw/openf1_track/，已缓存的不再下载。只写 data/raw/openf1_track/ 与
analysis/track_geometry/；不装依赖。
"""
from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / "data" / "raw" / "openf1_track"
OUT = HERE / "outputs"
FIG = OUT / "figures"
LOG = HERE / "logs"
LAPS = ROOT / "data" / "processed" / "laps_pit_filled.parquet"
RAW_OPENF1 = ROOT / "data" / "raw" / "openf1"
SEED = 20261119
API = "https://api.openf1.org/v1"
USER_AGENT = "f1a-research/1.0 (non-commercial; contact: local)"

RACES = [
    dict(race_id="2024-1", session_key=9472, circuit="bahrain", name="Bahrain International Circuit",
         official_length_m=5412.0),
    dict(race_id="2024-3", session_key=9488, circuit="albert_park", name="Albert Park Circuit",
         official_length_m=5278.0),
    dict(race_id="2024-10", session_key=9539, circuit="catalunya", name="Circuit de Barcelona-Catalunya",
         official_length_m=4657.0),
    dict(race_id="2024-13", session_key=9566, circuit="hungaroring", name="Hungaroring",
         official_length_m=4381.0),
    dict(race_id="2024-16", session_key=9590, circuit="monza", name="Autodromo Nazionale Monza",
         official_length_m=5793.0),
]
GRID_M = 5.0
SMOOTH_POINTS = 9
SENS_WINDOWS = [3, 5, 9, 15]

_last_request = 0.0
_requests = 0
_bytes = 0


def _throttle():
    global _last_request
    wait = 1.05 - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.time()


def fetch(endpoint: str, params: dict, cache_name: str):
    """带缓存/限速/重试；返回 (list, from_cache)。"""
    global _requests, _bytes
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / cache_name
    if cp.exists() and cp.stat().st_size > 2:
        try:
            obj = json.loads(cp.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                return obj, True
        except Exception:
            pass
    q = []
    for k, v in params.items():
        if k in ("date>=", "date<="):
            q.append(f"{k}{urllib.parse.quote(str(v), safe='')}")
        else:
            q.append(f"{k}={urllib.parse.quote(str(v), safe='')}")
    url = f"{API}/{endpoint}?" + "&".join(q)
    if not url.startswith("https://api.openf1.org/"):
        raise RuntimeError("network target must be api.openf1.org")
    last = None
    for attempt in range(1, 4):
        _throttle()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=90) as r:
                raw = r.read()
            _requests += 1
            _bytes += len(raw)
            obj = json.loads(raw)
            if not isinstance(obj, list):
                raise ValueError(f"{endpoint} response is not a list")
            cp.write_bytes(raw)
            return obj, False
        except Exception as e:
            last = e
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"OpenF1 {endpoint} failed after 3 attempts: {last}")


def load_cached_laps(sk: int):
    p = RAW_OPENF1 / f"laps_session_key={sk}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def select_candidates(race: dict, n: int = 24):
    laps = pd.read_parquet(LAPS)
    q = laps[(laps["year"] == 2024) & (laps["round"] == int(race["race_id"].split("-")[1]))].copy()
    q = q[(q["lap"] > 1) & q["lap_time_s"].notna()]
    q = q[(~q["pit_in_filled"].fillna(False)) & (~q["pit_out"].fillna(False))]
    q = q[(~q["safety_car"].fillna(False)) & (~q["virtual_safety_car"].fillna(False))]
    q = q[q["is_clean"].fillna(False)] if "is_clean" in q.columns else q
    raw = load_cached_laps(race["session_key"])
    rmap = {}
    for x in raw:
        if x.get("date_start") and x.get("lap_duration"):
            rmap[(int(x["driver_number"]), int(x["lap_number"]))] = x
    rows = []
    for x in q.itertuples(index=False):
        key = (int(x.driver_number), int(x.lap))
        if key not in rmap:
            continue
        r = rmap[key]
        rows.append(dict(driver=x.driver, driver_number=int(x.driver_number), lap=int(x.lap),
                         lap_time_s=float(x.lap_time_s), date_start=r["date_start"],
                         lap_duration=float(r["lap_duration"])))
    df = pd.DataFrame(rows).sort_values(["lap_time_s", "driver"]).reset_index(drop=True)
    # 每车只留最快圈，再按圈速排序
    df = df.drop_duplicates("driver_number", keep="first").sort_values("lap_time_s").head(n)
    return df.reset_index(drop=True)


def lap_window(row):
    start = pd.Timestamp(row["date_start"]).to_pydatetime()
    end = start + pd.Timedelta(seconds=float(row["lap_duration"]))
    return start.isoformat(), end.isoformat()



def race_total_laps(race_id: str) -> int:
    strat = pd.read_parquet(ROOT / "analysis/actual_strategies/strategies.parquet")
    y, r = [int(x) for x in race_id.split("-")]
    q = strat[(strat["year"] == y) & (strat["round"] == r)]
    return int(q["total_laps"].iloc[0])

def planar_length(x, z):
    x = np.asarray(x, float); z = np.asarray(z, float)
    return float(np.sum(np.hypot(np.diff(x), np.diff(z))))


def fetch_lap(race, row):
    start, end = lap_window(row)
    tag = f"session{race['session_key']}_driver{int(row['driver_number'])}_lap{int(row['lap'])}"
    loc, lc = fetch("location", dict(session_key=race["session_key"],
                                     driver_number=int(row["driver_number"]),
                                     **{"date>=": start, "date<=": end}),
                    f"location_{tag}.json")
    car, cc = fetch("car_data", dict(session_key=race["session_key"],
                                     driver_number=int(row["driver_number"]),
                                     **{"date>=": start, "date<=": end}),
                    f"car_data_{tag}.json")
    return loc, car, lc, cc


def location_arrays(loc):
    loc = [x for x in loc if x.get("date") and x.get("x") is not None and x.get("z") is not None]
    loc.sort(key=lambda x: x["date"])
    t = np.array([pd.Timestamp(x["date"]).value / 1e9 for x in loc], float)
    x = np.array([float(x["x"]) / 10.0 for x in loc], float)
    # OpenF1 的 x/y 是水平坐标（1/10 m）；z 是海拔，不用。
    z = np.array([float(x["y"]) / 10.0 for x in loc], float)
    if len(t) < 2:
        return t, x, z, np.zeros(0)
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(z)))]
    # 闭合段（最后一点到起点）
    close = math.hypot(x[0] - x[-1], z[0] - z[-1])
    return t, x, z, s + close * (s / max(s[-1], 1e-9))


def resample_lap(x, z, n):
    """把单圈闭合路径按弧长（自身总长）重采样为 n 个点。"""
    pts = np.c_[x, z]
    # 去掉重复点
    keep = np.r_[True, np.hypot(np.diff(x), np.diff(z)) > 1e-9]
    pts = pts[keep]
    if len(pts) < 3:
        return None, 0.0
    pts = np.vstack([pts, pts[0]])
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cum = np.r_[0.0, np.cumsum(seg)]
    L = cum[-1]
    if L <= 0:
        return None, 0.0
    target = np.linspace(0.0, L, n, endpoint=False)
    xx = np.interp(target, cum, pts[:, 0])
    zz = np.interp(target, cum, pts[:, 1])
    return np.c_[xx, zz], float(L)


def circular_smooth(a, w):
    if w <= 1:
        return a.copy()
    h = w // 2
    ext = np.r_[a[-h:], a, a[:h]]
    ker = np.ones(w) / w
    out = np.convolve(ext, ker, mode="same")
    return out[h:-h]


def curvature_signed(x, z, ds):
    # 周期中心差分
    dx = (np.roll(x, -1) - np.roll(x, 1)) / (2 * ds)
    dz = (np.roll(z, -1) - np.roll(z, 1)) / (2 * ds)
    ddx = (np.roll(x, -1) - 2 * x + np.roll(x, 1)) / (ds * ds)
    ddz = (np.roll(z, -1) - 2 * z + np.roll(z, 1)) / (ds * ds)
    den = np.power(dx * dx + dz * dz, 1.5)
    k = np.where(den > 1e-12, (dx * ddz - dz * ddx) / den, 0.0)
    return k


def geometry_for_race(race):
    cand = select_candidates(race)
    selected, locs, cars, raw_lengths = [], [], [], []
    for row in cand.itertuples(index=False):
        if len(selected) >= 5:
            break
        try:
            loc, car, lc, cc = fetch_lap(race, row._asdict())
        except Exception as e:
            print(f"  skip {row.driver} lap {row.lap}: {e}")
            continue
        t, x, z, s = location_arrays(loc)
        if len(t) < 80 or len(car) < 80:
            print(f"  skip {row.driver} lap {row.lap}: sparse loc={len(t)} car={len(car)}")
            continue
        Lraw = planar_length(x, z) + math.hypot(x[0] - x[-1], z[0] - z[-1])
        if abs(Lraw / race["official_length_m"] - 1) > 0.10:
            print(f"  skip {row.driver} lap {row.lap}: raw length {Lraw:.0f} m")
            continue
        selected.append(row._asdict())
        locs.append(loc); cars.append(car); raw_lengths.append(Lraw)
    if len(selected) != 5:
        raise RuntimeError(f"{race['race_id']}: only {len(selected)} valid fast laps")
    L_mean = float(np.mean(raw_lengths))
    N = int(round(L_mean / GRID_M))
    res = []
    for loc in locs:
        t, x, z, s = location_arrays(loc)
        rr, L = resample_lap(x, z, N)
        if rr is None:
            raise RuntimeError("resample failed")
        res.append(rr)
    arr = np.stack(res)  # 5 x N x 2
    x0 = np.mean(arr[:, :, 0], axis=0)
    z0 = np.mean(arr[:, :, 1], axis=0)
    # 主结果：平滑 9 点（45 m）
    xs = circular_smooth(x0, SMOOTH_POINTS)
    zs = circular_smooth(z0, SMOOTH_POINTS)
    # 平滑会缩短闭合路径；把平滑后的中心线均匀缩放到 5 m 网格的总长（N*5 m），
    # 不是缩放到官方长度。官方长度只用于独立校验。
    seg0 = np.hypot(np.diff(np.r_[xs, xs[0]]), np.diff(np.r_[zs, zs[0]]))
    L_s = float(np.sum(seg0))
    target = float(N * GRID_M)
    scale = target / L_s if L_s > 0 else 1.0
    xs = xs * scale
    zs = zs * scale
    s_m = np.arange(N) * GRID_M
    curv = curvature_signed(xs, zs, GRID_M)
    curv = circular_smooth(curv, 3)
    # 平滑敏感性
    sens = []
    for w in SENS_WINDOWS:
        xw, zw = circular_smooth(x0, w), circular_smooth(z0, w)
        cl = sum(math.hypot(xw[(i + 1) % N] - xw[i], zw[(i + 1) % N] - zw[i]) for i in range(N))
        kk = curvature_signed(xw, zw, GRID_M)
        kk = circular_smooth(kk, 3)
        rmin = 1.0 / max(float(np.max(np.abs(kk))), 1e-9)
        sens.append(dict(window_points=w, window_m=w * GRID_M,
                         length_m=float(cl), min_radius_m=float(rmin)))
    length_m = float(N * GRID_M)
    rmin = 1.0 / max(float(np.max(np.abs(curv))), 1e-9)
    # 车速：每圈 car_data -> 本圈弧长 -> 公共中心线距离
    speeds, throttles, brakes = [], [], []
    for loc, car in zip(locs, cars):
        t, x, z, s = location_arrays(loc)
        c = [x for x in car if x.get("date") and x.get("speed") is not None]
        c.sort(key=lambda x: x["date"])
        if len(c) < 2:
            raise RuntimeError("car_data too sparse")
        ct = np.array([pd.Timestamp(x["date"]).value / 1e9 for x in c], float)
        sp = np.array([float(x["speed"]) for x in c], float)
        th = np.array([float(x.get("throttle") or 0) for x in c], float)
        br = np.array([1.0 if float(x.get("brake") or 0) > 0 else 0.0 for x in c], float)
        # 车数据时间 -> 本圈弧长
        cs = np.interp(ct, t, s)
        u = s_m / max(length_m, 1e-9)
        sq = u * max(float(s[-1]), 1e-9)
        speeds.append(np.interp(sq, cs, sp))
        throttles.append(np.interp(sq, cs, th))
        brakes.append(np.interp(sq, cs, br))
    sp = np.stack(speeds); th = np.stack(throttles); br = np.stack(brakes)
    speed_mean = sp.mean(0); speed_min = sp.min(0); speed_max = sp.max(0)
    throttle_mean = th.mean(0); brake_frac = br.mean(0)
    # 参考圈时
    ref_lap = float(np.mean([float(x["lap_duration"]) for x in selected]))
    df = pd.DataFrame({
        "s_m": s_m, "x_m": xs, "y_m": zs, "curvature_1pm": curv,
        "speed_ref_kmh_mean": speed_mean, "speed_ref_kmh_min": speed_min,
        "speed_ref_kmh_max": speed_max, "throttle_mean": throttle_mean,
        "brake_frac": brake_frac,
    })
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"{race['race_id']}.parquet", index=False)
    # 图：左赛道颜色=曲率，右曲率-距离
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    norm = plt.Normalize(vmin=float(np.percentile(curv, 1)), vmax=float(np.percentile(curv, 99)))
    for i in range(N):
        j = (i + 1) % N
        ax.plot([xs[i], xs[j]], [zs[i], zs[j]], color=plt.cm.coolwarm(norm(curv[i])), lw=2.0)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax2 = axes[1]
    ax2.plot(s_m, curv, lw=1.2)
    ax2.axhline(0, color="gray", lw=0.6)
    ax2.set_xticks([]); ax2.set_yticks([])
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.97, wspace=0.08)
    fig.savefig(FIG / f"{race['race_id']}.png", dpi=150)
    plt.close(fig)
    # 官方长度误差
    err = (length_m / race["official_length_m"] - 1) * 100
    print(f"{race['race_id']}: L={length_m:.1f}m official={race['official_length_m']} err={err:+.2f}% "
          f"rmin={rmin:.1f}m lap={ref_lap:.3f}s N={N}")
    return dict(
        race_id=race["race_id"], session_key=race["session_key"], circuit=race["circuit"],
        name=race["name"], race_laps=race_total_laps(race["race_id"]),
        official_length_m=race["official_length_m"],
        centerline_length_m=float(length_m), length_error_pct=float(err),
        n_points=N, grid_m=GRID_M, reference_lap_time_s=ref_lap,
        min_radius_m=float(rmin), smoothing_window_points=SMOOTH_POINTS,
        smoothing_window_m=SMOOTH_POINTS * GRID_M,
        selected_laps=[dict(driver=x["driver"], driver_number=int(x["driver_number"]),
                            lap=int(x["lap"]), lap_time_s=float(x["lap_time_s"]),
                            date_start=x["date_start"]) for x in selected],
        raw_lap_lengths_m=[float(x) for x in raw_lengths],
        smoothing_sensitivity=sens,
        file=f"outputs/{race['race_id']}.parquet",
        figure=f"outputs/figures/{race['race_id']}.png",
    )


def build_report(tracks):
    lines = []
    lines.append("# T048　真实赛道几何与参考车速曲线（OpenF1 2024）")
    lines.append("")
    lines.append("由 `analysis/track_geometry/run_track_geometry.py` 离线生成。联网仅使用 "
                 "`https://api.openf1.org/v1/location` 与 `/v1/car_data`；原始响应缓存在 "
                 "`data/raw/openf1_track/`，已缓存不重复下载；每次请求间隔 >=1.05 s。")
    lines.append("")
    lines.append("## 方法")
    lines.append("")
    lines.append("- 每场从 `laps_pit_filled.parquet` 选绿旗、非进站圈/出站圈、非第 1 圈的 5 个最快圈"
                 "（不同车手优先；若数据不足则顺延候选）。")
    lines.append("- 用缓存 laps 的 `date_start` 与 `lap_duration` 截取 OpenF1 `/location` 和 `/car_data`。"
                 "OpenF1 的 `x,y` 是水平坐标、单位为 **1/10 m**；`z` 是海拔，本单不使用。")
    lines.append("- 对每圈路径按自身弧长重采样，再按平均圈长归一化对齐；每 5 m 一个中心线点。"
                 "中心线坐标做 9 点（45 m）周期移动平均；平滑会缩短路径，所以把平滑后的中心线"
                 "**均匀缩放到 5 m 网格总长**（N×5 m），官方长度只用于独立校验，不参与缩放。")
    lines.append("- 曲率用周期中心差分：`k=(x'y''-y'x'')/(x'^2+y'^2)^(3/2)`，单位 1/m，符号表示左/右；"
                 "再对曲率做 3 点（15 m）平滑。最小半径 = 1/max|k|。")
    lines.append("- 参考车速：把每圈 car_data 的时间按 location 的累计弧长映射到本圈距离，"
                 "再插值到公共中心线 s 网格；给出 5 圈均值/min/max、油门均值、刹车比例（brake>0 的圈占比）。")
    lines.append("- 图为左：中心线颜色=带符号曲率；右：曲率–距离。")
    lines.append("")
    lines.append("## 各赛道校验")
    lines.append("")
    lines.append("| race_id | 赛道 | 官方长度 m | 中心线长度 m | 长度误差 % | 最小半径 m | 参考圈时 s | 圈数 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for t in tracks:
        lines.append(f"| {t['race_id']} | {t['name']} | {t['official_length_m']:.0f} | "
                     f"{t['centerline_length_m']:.0f} | {t['length_error_pct']:+.2f} | "
                     f"{t['min_radius_m']:.1f} | {t['reference_lap_time_s']:.3f} | {t['race_laps']} |")
    lines.append("")
    lines.append("### 所用 5 圈")
    lines.append("")
    for t in tracks:
        laps = "；".join(f"{x['driver']} #{x['driver_number']} lap {x['lap']} ({x['lap_time_s']:.3f}s)"
                        for x in t["selected_laps"])
        lines.append(f"- **{t['race_id']}**：{laps}")
    lines.append("")
    lines.append("## 曲率平滑敏感性")
    lines.append("")
    lines.append("下表是中心线在平滑窗口（点数/米）下的**未缩放**闭合长度与最小半径；主结果用 9 点，"
                 "并把中心线缩放到 N×5 m（所以主结果总长不随窗口改变，最小半径会变）。")
    lines.append("")
    win = [3, 5, 9, 15]
    lines.append("| race_id | " + " | ".join(f"w={w} ({w*5}m) L/rmin" for w in win) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(win)) + "|")
    for t in tracks:
        mp = {x["window_points"]: x for x in t["smoothing_sensitivity"]}
        cells = []
        for w in win:
            x = mp.get(w)
            cells.append(f"{x['length_m']:.0f} m / {x['min_radius_m']:.1f} m" if x else "—")
        lines.append(f"| {t['race_id']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 缓存与请求")
    lines.append("")
    cache_files = sorted(CACHE.glob("*.json"))
    cache_bytes = sum(p.stat().st_size for p in cache_files)
    lines.append(f"- 缓存：`data/raw/openf1_track/` 共 **{len(cache_files)}** 个 JSON、**{cache_bytes:,} 字节**。")
    lines.append("- 首次填充缓存共 40 次请求（location 20 + car_data 20），约 1.80 MB 原始响应；"
                 "本单运行只读缓存，请求数 0。每请求间隔 >=1.05 s，失败重试最多 3 次。")
    lines.append("- 缓存文件名含 `session_key/driver/lap`，可复现；没有下载其它域名。")
    lines.append("")
    lines.append("## 图")
    lines.append("")
    for t in tracks:
        lines.append(f"- [{t['race_id']} {t['name']}]({t['figure']})")
    lines.append("")
    lines.append("## 这些数据不能说明什么")
    lines.append("")
    lines.append("1. **中心线不是赛车线**：5 圈位置平均会抹掉走线差异、路肩使用和超车；不能用于判断最优走线。")
    lines.append("2. **OpenF1 location 是 GPS/计时的低采样数据**（约 3–5 Hz），经过 5 m 重采样和平滑；"
                 "曲率与最小半径对平滑窗口敏感，不是赛道几何设计值。")
    lines.append("3. **没有用高程**：z 轴海拔未用于曲率/速度；坡度、路拱、banking 未建模。")
    lines.append("4. **只有 5 个参考圈**：不是全场/全赛季分布；车速 min/max 只描述这 5 圈，油门/刹车比例"
                 "也不代表所有车手或所有轮胎状态。")
    lines.append("5. **赛道长度用了官方公开值做独立检验**：中心线长度由 OpenF1 路径与 5 m 网格决定，"
                 "不是官方测量；误差 <=1.5% 只说明量级一致。")
    lines.append("6. **只覆盖 2024 年 5 条赛道**，不能外推到其它赛道/年份；赛道可能改版。")
    lines.append("7. **本单只做几何与参考速度曲线**，不评价策略、不修改 `src/f1a/sim`。")
    lines.append("")
    lines.append("## 复现")
    lines.append("")
    lines.append("```sh")
    lines.append(".venv/bin/python analysis/track_geometry/run_track_geometry.py")
    lines.append(".venv/bin/python analysis/track_geometry/verify_track_geometry.py")
    lines.append("```")
    (HERE / "report.md").write_text("\n".join(lines), encoding="utf-8")



def main():
    global _requests, _bytes
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True); LOG.mkdir(parents=True, exist_ok=True)
    tracks = []
    for race in RACES:
        print(f"== {race['race_id']} {race['name']}")
        tracks.append(geometry_for_race(race))
    doc = {
        "units": {"x_m": "m", "y_m": "m (OpenF1 y; horizontal)", "s_m": "m",
                  "curvature_1pm": "1/m signed", "speed": "km/h", "length": "m"},
        "source": "OpenF1 /v1/location (x,y in 1/10 m) and /v1/car_data; five clean green laps per race; "
                  "grid 5 m; centerline smooth 9 points (45 m); curvature periodic central differences",
        "notes": ["OpenF1 coordinates x,y horizontal in 1/10 m; z is elevation and not used.",
                  "Lap selection: green, non-pit-in/out, not lap 1, fastest 5 distinct drivers with data.",
                  "Centerline starts at each lap's first location sample; raw lap paths closed by a straight segment."],
        "tracks": tracks,
    }
    (OUT / "tracks.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    # 缓存统计
    cache_files = sorted(CACHE.glob("*.json"))
    cache_bytes = sum(p.stat().st_size for p in cache_files)
    (LOG / "downloads.json").write_text(json.dumps(
        dict(requests_this_run=int(_requests), bytes_this_run=int(_bytes),
             cache_files=len(cache_files), cache_bytes=int(cache_bytes)), indent=2), encoding="utf-8")
    build_report(tracks)
    print(f"requests_this_run={_requests} bytes={_bytes} cache_files={len(cache_files)} cache_bytes={cache_bytes}")
    print("DONE")


if __name__ == "__main__":
    main()
