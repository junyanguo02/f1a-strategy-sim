"""Build simulator inputs from the local race cache."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import pandas as pd

YEARS = [2023, 2024, 2025, 2026]

HERE = Path(__file__).resolve().parent

ROOT = HERE.parents[1]

LAPS = [ROOT / "data" / "processed" / f"laps_{y}.parquet" for y in YEARS]

PIT_FILLED = ROOT / "data" / "processed" / "laps_pit_filled.parquet"

MODEL = ROOT / "data" / "processed" / "model_laps.parquet"

OUT_PARQUET = HERE / "traffic_features.parquet"

def load_data():
    model = pd.read_parquet(MODEL)
    raw = pd.concat([pd.read_parquet(p) for p in LAPS], ignore_index=True)
    pitf = pd.read_parquet(PIT_FILLED)[["year", "round", "driver", "lap", "pit_in_filled"]]

    keys = ["year", "round", "driver", "lap"]
    race_keys = model[keys[:2]].drop_duplicates()
    raw = raw.merge(race_keys, on=keys[:2], how="inner")
    raw = raw.merge(pitf, on=keys, how="left")
    raw["pit_eff"] = (
        raw["pit_in"].astype("boolean").fillna(raw["pit_in_filled"].astype("boolean")).fillna(False).astype(bool)
    )
    return model, raw

def build_race(race_df: pd.DataFrame) -> dict:
    """为一场比赛构造每名车手的累计时间/覆盖数组、逐圈排名状态、pit 标记。"""
    drivers: dict[str, dict] = {}
    all_max_lap = 0
    for driver, g in race_df.groupby("driver"):
        g = g.sort_values("lap")
        laps = g["lap"].to_numpy(dtype=int)
        lts = g["lap_time_s"].to_numpy(dtype=float)
        maxlap = int(laps.max()) if len(laps) else 0
        all_max_lap = max(all_max_lap, maxlap)
        cum = np.full(maxlap + 1, np.nan)
        cov = np.zeros(maxlap + 1, dtype=bool)
        cum_f = np.array([], dtype=float)
        lap_f = np.array([], dtype=int)
        break_reason = "none"
        s = 0.0
        prev = 0
        for j in range(len(laps)):
            k = int(laps[j])
            if k != prev + 1:
                break_reason = "lap_missing" if k > prev + 1 else "sequence_start"
                break
            if np.isnan(lts[j]):
                break_reason = "lap_time_missing"
                break
            s += float(lts[j])
            cum[k] = s
            cov[k] = True
            prev = k
        fin = np.where(np.isfinite(cum))[0]
        if len(fin):
            cum_f = cum[fin]
            lap_f = fin
        drivers[driver] = {
            "cum": cum,
            "cov": cov,
            "cum_f": cum_f,
            "lap_f": lap_f,
            "break_reason": break_reason,
            "maxlap": maxlap,
        }

    # 比赛时间轴上的“已完成圈数”前缀最大值，用于查领先者圈数
    ev_t, ev_l = [], []
    for info in drivers.values():
        for k in range(1, len(info["cum"])):
            if np.isfinite(info["cum"][k]):
                ev_t.append(info["cum"][k])
                ev_l.append(k)
    if ev_t:
        order = np.argsort(ev_t)
        ev_t = np.asarray(ev_t, dtype=float)[order]
        ev_l = np.asarray(ev_l, dtype=int)[order]
        leader_prefix = np.maximum.accumulate(ev_l)
    else:
        ev_t = np.array([], dtype=float)
        leader_prefix = np.array([], dtype=int)

    # 逐圈排名状态（只含该圈 cum 可用的车手）
    states: dict[int, tuple] = {}
    for k in range(1, all_max_lap + 1):
        pairs = []
        for d, info in drivers.items():
            if k < len(info["cum"]) and np.isfinite(info["cum"][k]):
                pairs.append((float(info["cum"][k]), d))
        if pairs:
            pairs.sort()
            times = np.array([p[0] for p in pairs])
            names = [p[1] for p in pairs]
            states[k] = (times, names, {d: i for i, d in enumerate(names)})

    pit = {
        (str(dr), int(lp)): bool(v)
        for dr, lp, v in race_df[["driver", "lap", "pit_eff"]].itertuples(index=False)
    }
    return {"drivers": drivers, "states": states, "ev_t": ev_t,
            "leader_prefix": leader_prefix, "pit": pit}

def laps_at(info: dict, t: float) -> int:
    """车手在比赛时间 t 时已完成的圈数。"""
    if len(info["cum_f"]) == 0:
        return 0
    idx = int(np.searchsorted(info["cum_f"], t, side="right"))
    return int(info["lap_f"][idx - 1]) if idx > 0 else 0

def leader_laps(race: dict, t: float) -> int:
    if len(race["ev_t"]) == 0:
        return 0
    idx = int(np.searchsorted(race["ev_t"], t, side="right"))
    return int(race["leader_prefix"][idx - 1]) if idx > 0 else 0

def compute_features(model: pd.DataFrame, raw: pd.DataFrame):
    """对 model_laps 的每一行计算交通特征；返回 dict（key=year,round,driver,lap）。"""
    records: dict[tuple, dict] = {}
    miss = {"no_prev_reliable": 0, "lap_time_missing": 0, "lap_missing": 0,
            "sequence_start": 0, "no_car_ahead": 0, "no_car_behind": 0}

    for (year, rnd), rows in model.groupby(["year", "round"], sort=False):
        race_raw = raw[(raw["year"] == year) & (raw["round"] == rnd)]
        race = build_race(race_raw)
        drivers = race["drivers"]
        states = race["states"]
        pit = race["pit"]
        for row in rows.itertuples(index=False):
            driver = str(row.driver)
            lap = int(row.lap)
            prev = lap - 1
            rec = {
                "gap_ahead_prev": np.nan,
                "gap_behind_prev": np.nan,
                "pos_prev": np.nan,
                "ahead_lapped": np.nan,
                "ahead_pitting": np.nan,
                "drs_window_prev": np.nan,
                "gap_source_ok": False,
            }
            info = drivers.get(driver)
            ok = (info is not None and prev >= 1 and prev < len(info["cum"])
                  and np.isfinite(info["cum"][prev]))
            if not ok:
                reason = info["break_reason"] if info is not None else "lap_time_missing"
                if reason in ("lap_time_missing", "lap_missing", "sequence_start"):
                    miss[reason] += 1
                else:
                    miss["no_prev_reliable"] += 1
                records[(int(year), int(rnd), driver, lap)] = rec
                continue
            state = states.get(prev)
            if state is None or driver not in state[2]:
                miss["no_prev_reliable"] += 1
                records[(int(year), int(rnd), driver, lap)] = rec
                continue
            times, names, where = state
            i = where[driver]
            t = float(times[i])
            rec["gap_source_ok"] = True
            rec["pos_prev"] = i + 1
            if i > 0:
                ahead = names[i - 1]
                ga = t - float(times[i - 1])
                rec["gap_ahead_prev"] = ga
                rec["ahead_lapped"] = bool(laps_at(drivers[ahead], t) < leader_laps(race, t))
                rec["ahead_pitting"] = bool(pit.get((ahead, prev), False))
                rec["drs_window_prev"] = bool(ga < 1.0)
            else:
                miss["no_car_ahead"] += 1
            if i + 1 < len(times):
                rec["gap_behind_prev"] = float(times[i + 1]) - t
            else:
                miss["no_car_behind"] += 1
            records[(int(year), int(rnd), driver, lap)] = rec

    # 按 model_laps 原始顺序组装数组
    n = len(model)
    out = {
        "gap_ahead_prev": np.full(n, np.nan),
        "gap_behind_prev": np.full(n, np.nan),
        "pos_prev": np.full(n, np.nan, dtype=object),
        "ahead_lapped": np.full(n, np.nan, dtype=object),
        "ahead_pitting": np.full(n, np.nan, dtype=object),
        "drs_window_prev": np.full(n, np.nan, dtype=object),
        "gap_source_ok": np.zeros(n, dtype=bool),
    }
    for idx, row in enumerate(model.itertuples(index=False)):
        rec = records[(int(row.year), int(row.round), str(row.driver), int(row.lap))]
        for c in out:
            out[c][idx] = rec[c]
    return out, miss

def main():
    model, raw = load_data()
    feats, miss = compute_features(model, raw)

    out = model[["year", "round", "driver", "lap"]].copy()
    out["gap_ahead_prev"] = feats["gap_ahead_prev"]
    out["gap_behind_prev"] = feats["gap_behind_prev"]
    out["pos_prev"] = pd.array(feats["pos_prev"], dtype="Int64")
    out["ahead_lapped"] = pd.array(feats["ahead_lapped"], dtype="boolean")
    out["ahead_pitting"] = pd.array(feats["ahead_pitting"], dtype="boolean")
    out["drs_window_prev"] = pd.array(feats["drs_window_prev"], dtype="boolean")
    out["gap_source_ok"] = pd.array(feats["gap_source_ok"], dtype="boolean")
    out.to_parquet(OUT_PARQUET, index=False)


if __name__ == "__main__":
    main()
