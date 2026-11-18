"""Build simulator inputs from the local race cache."""
from __future__ import annotations

import json

from pathlib import Path

import numpy as np

import pandas as pd

HERE = Path(__file__).resolve().parent

ROOT = HERE.parents[1]

PROCESSED = ROOT / "data" / "processed"

RAW_OPENF1 = ROOT / "data" / "raw" / "openf1"

PIT_LOSS = PROCESSED / "pit_loss.parquet"

SUMMARY = PROCESSED / "fcy_race_summary.csv"

LAPS = PROCESSED / "laps_pit_filled.parquet"

def race_control(session_key: int) -> list[dict]:
    path = RAW_OPENF1 / f"race_control_session_key={session_key}.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

def build_red_flag_periods(summary: pd.DataFrame) -> pd.DataFrame:
    """Race-control red-flag windows as [start_lap, end_lap] leader laps.

    A red flag starts on a Track RED message or a message beginning with
    "RED FLAG".  The window ends at the latest restart marker within three
    laps (SESSION STARTED / STANDING START / ROLLING START / RACE START /
    pit-exit green), or at start_lap+1 when no marker exists.  This covers
    the suspension laps without relying on the lap-time magnitude.
    """
    rows = []
    for race in summary.itertuples(index=False):
        msgs = sorted(race_control(int(race.session_key)), key=lambda m: m["date"])
        starts = []
        for msg in msgs:
            text = (msg.get("message") or "").upper()
            lap = msg.get("lap_number")
            if lap is None:
                continue
            if ((msg.get("flag") == "RED" and msg.get("scope") == "Track")
                    or text.startswith("RED FLAG")):
                starts.append((pd.Timestamp(msg["date"]), int(lap)))
        for ts, start in starts:
            end = start + 1
            for msg in msgs:
                if pd.Timestamp(msg["date"]) <= ts:
                    continue
                text = (msg.get("message") or "").upper()
                lap = msg.get("lap_number")
                if lap is None:
                    continue
                lap = int(lap)
                if lap > start + 3:
                    break
                if any(key in text for key in (
                        "SESSION STARTED", "STANDING START", "ROLLING START",
                        "RACE START", "GREEN LIGHT - PIT EXIT OPEN")):
                    end = max(end, lap)
            rows.append(dict(year=int(race.year), round=int(race.round),
                             start_lap=start, end_lap=min(end, int(race.total_laps)),
                             session_key=int(race.session_key)))
    periods = pd.DataFrame(rows)
    if len(periods):
        periods = periods.drop_duplicates(["year", "round", "start_lap", "end_lap"])
        periods = periods.sort_values(["year", "round", "start_lap", "end_lap"]).reset_index(drop=True)
        merged = []
        for item in periods.itertuples(index=False):
            if (merged and item.year == merged[-1]["year"]
                    and item.round == merged[-1]["round"]
                    and item.start_lap <= merged[-1]["end_lap"]):
                merged[-1]["end_lap"] = max(merged[-1]["end_lap"], item.end_lap)
            else:
                merged.append(dict(year=int(item.year), round=int(item.round),
                                   start_lap=int(item.start_lap), end_lap=int(item.end_lap),
                                   session_key=int(item.session_key)))
        periods = pd.DataFrame(merged)
    return periods

def build_red_lap_set(periods: pd.DataFrame, laps: pd.DataFrame) -> set[tuple[int, int, int]]:
    """(year, round, lap) keys touched by a red flag (message window or status 5)."""
    keys: set[tuple[int, int, int]] = set()
    for row in periods.itertuples(index=False):
        for lap in range(int(row.start_lap), int(row.end_lap) + 1):
            keys.add((int(row.year), int(row.round), lap))
    status = laps["track_status"].astype("string").str.contains("5", regex=False)
    for year, rnd, lap in laps.loc[status, ["year", "round", "lap"]].itertuples(index=False):
        keys.add((int(year), int(rnd), int(lap)))
    return keys

def build_reference_pool(laps: pd.DataFrame, red_laps: set[tuple[int, int, int]]):
    """Median-pool rows: valid lap time, no pit in/out, no red-flag lap."""
    pool = laps[
        laps["lap_time_s"].notna()
        & ~laps["pit_in_filled"].fillna(False)
        & ~laps["pit_out"].fillna(False)
    ].copy()
    keep = [key not in red_laps for key in
            zip(pool["year"].astype(int), pool["round"].astype(int), pool["lap"].astype(int))]
    pool = pool.loc[keep]
    groups: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for key, sub in pool.groupby(["year", "round", "lap"], sort=False):
        groups[(int(key[0]), int(key[1]), int(key[2]))] = (
            sub["driver_number"].to_numpy(),
            sub["lap_time_s"].to_numpy(dtype=float),
        )
    return groups

def reference_median(groups, year: int, rnd: int, lap: int, driver_number: int):
    data = groups.get((int(year), int(rnd), int(lap)))
    if data is None:
        return np.nan, 0
    drivers, times = data
    values = times[drivers != int(driver_number)]
    if values.size == 0:
        return np.nan, 0
    return float(np.median(values)), int(values.size)

def build_relative(pit: pd.DataFrame, groups, red_laps: set[tuple[int, int, int]]) -> pd.DataFrame:
    out = pit.copy()

    out["red_in"] = [key in red_laps for key in
                     zip(out["year"], out["round"], out["lap"])]
    out["red_out"] = [key in red_laps for key in
                      zip(out["year"], out["round"], out["out_lap"])]

    in_refs = [reference_median(groups, y, r, lap, dn) for y, r, lap, dn in
               out[["year", "round", "lap", "driver_number"]].itertuples(index=False)]
    out_refs = [reference_median(groups, y, r, lap, dn) for y, r, lap, dn in
                out[["year", "round", "out_lap", "driver_number"]].itertuples(index=False)]
    out["reference_in_s"] = [x[0] for x in in_refs]
    out["n_reference_in"] = [x[1] for x in in_refs]
    out["reference_out_s"] = [x[0] for x in out_refs]
    out["n_reference_out"] = [x[1] for x in out_refs]

    out["relative_in_s"] = out["in_lap_time_s"] - out["reference_in_s"]
    out["relative_out_s"] = out["out_lap_time_s"] - out["reference_out_s"]
    out["relative_loss_s"] = out["relative_in_s"] + out["relative_out_s"]

    reason = pd.Series("", index=out.index, dtype="object")
    reason[~out["usable"].fillna(False)] = "not_usable"
    reason[(out["usable"].fillna(False)) & (out["condition"] == "unknown")] = "condition_unknown"
    mask = out["usable"].fillna(False) & (out["condition"] != "unknown")
    reason[mask & (out["red_in"] | out["red_out"])] = "red_flag"
    mask2 = mask & ~(out["red_in"] | out["red_out"])
    reason[mask2 & (out["n_reference_in"] < 1)] = "no_reference_in"
    reason[mask2 & (out["n_reference_in"] >= 1) & (out["n_reference_out"] < 1)] = "no_reference_out"
    out["exclusion_reason"] = reason
    out["relative_usable"] = out["exclusion_reason"].eq("")
    return out.sort_values(["year", "round", "driver_number", "lap"]).reset_index(drop=True)

def main():
    summary = pd.read_csv(SUMMARY)
    pit = pd.read_parquet(PIT_LOSS)
    laps = pd.read_parquet(LAPS)

    periods = build_red_flag_periods(summary)
    red_laps = build_red_lap_set(periods, laps)
    groups = build_reference_pool(laps, red_laps)
    relative = build_relative(pit, groups, red_laps)
    relative.to_parquet(HERE / "pit_loss_relative.parquet", index=False)

if __name__ == "__main__":
    main()
