"""Build simulator inputs from the local race cache."""
from __future__ import annotations

import glob

import json

from pathlib import Path

import numpy as np

import pandas as pd

YEARS = [2023, 2024, 2025, 2026]

HERE = Path(__file__).resolve().parent

ROOT = HERE.parents[1]

PROCESSED = ROOT / "data" / "processed"

RAW = ROOT / "data" / "raw" / "openf1"

RELATIVE = ROOT / "analysis" / "fcy_inputs" / "pit_loss_relative.parquet"

EPISODES = PROCESSED / "fcy_episodes.parquet"

LAPS = PROCESSED / "laps_pit_filled.parquet"

OUT = HERE / "outputs"

def load_data():
    rel = pd.read_parquet(RELATIVE)
    eps = pd.read_parquet(EPISODES)
    laps = pd.read_parquet(LAPS)
    eps["start_ts"] = pd.to_datetime(eps["start_time"], format="ISO8601", utc=True)
    eps["end_ts"] = pd.to_datetime(eps["end_time"], format="ISO8601", utc=True, errors="coerce")

    pits = pd.concat(
        [pd.DataFrame(json.load(open(f, encoding="utf-8")))
         for f in glob.glob(str(RAW / "pit_session_key=*.json"))],
        ignore_index=True,
    )
    pits["date"] = pd.to_datetime(pits["date"], format="ISO8601", utc=True)
    pits["session_key"] = pits["session_key"].astype(int)

    lap_rows = []
    for f in glob.glob(str(RAW / "laps_session_key=*.json")):
        for r in json.load(open(f, encoding="utf-8")):
            lap_rows.append((int(r["session_key"]), int(r["driver_number"]),
                             int(r["lap_number"]), r.get("date_start")))
    raw_laps = pd.DataFrame(lap_rows, columns=["session_key", "driver_number", "lap", "lap_start"])
    raw_laps["lap_start"] = pd.to_datetime(raw_laps["lap_start"], format="ISO8601", utc=True)

    rc_rows = []
    for f in glob.glob(str(RAW / "race_control_session_key=*.json")):
        sk = int(Path(f).name.split("=")[1].split(".")[0])
        for m in json.load(open(f, encoding="utf-8")):
            if m.get("category") == "SafetyCar":
                rc_rows.append({"session_key": sk, "date": m.get("date"),
                                "lap_number": m.get("lap_number"),
                                "message": str(m.get("message", "")).upper()})
    rc = pd.DataFrame(rc_rows)
    rc["date"] = pd.to_datetime(rc["date"], format="ISO8601", utc=True)

    sessions = []
    for y in YEARS:
        sessions += json.load(open(RAW / f"sessions_session_name=Race&year={y}.json", encoding="utf-8"))
    sessions = pd.DataFrame(sessions)
    sessions["date"] = pd.to_datetime(sessions["date_start"], format="ISO8601", utc=True)
    return rel, eps, laps, pits, raw_laps, rc, sessions

def build_session_map(sessions: pd.DataFrame, eps: pd.DataFrame) -> pd.DataFrame:
    """年份 + 站次 → session_key：sessions 文件里非取消的 Race 按 date_start 排序的序号即站次。"""
    s = sessions[sessions["is_cancelled"] == False].sort_values(["year", "date"]).copy()
    s["round"] = s.groupby("year").cumcount() + 1
    smap = s[["year", "round", "session_key"]].copy()
    smap["session_key"] = smap["session_key"].astype(int)
    # 与 fcy_episodes 里已有的映射核对
    ref = eps[["year", "round", "session_key"]].drop_duplicates()
    cmp = ref.merge(smap, on=["year", "round"], how="left", suffixes=("_eps", "_sessions"))
    mism = int((cmp["session_key_eps"].astype(int) != cmp["session_key_sessions"].astype(int)).sum())
    if mism:
        raise RuntimeError(f"session map mismatch for {mism} episodes")
    return smap

def match_pits(cohort: pd.DataFrame, pits: pd.DataFrame, smap: pd.DataFrame) -> pd.DataFrame:
    """SC/VSC 可用进站与 OpenF1 pit 记录按 session_key + driver_number + lap_number 匹配。"""
    df = cohort.merge(smap, on=["year", "round"], how="left")
    if df["session_key"].isna().any():
        raise RuntimeError("missing session_key for some pits")
    df["session_key"] = df["session_key"].astype(int)
    pit_sel = pits.rename(columns={"lap_number": "lap"})[
        ["session_key", "driver_number", "lap", "date", "lane_duration",
         "stop_duration", "pit_duration"]]
    dup = pit_sel.duplicated(["session_key", "driver_number", "lap"], keep=False)
    if dup.any():
        raise RuntimeError(f"duplicate OpenF1 pit records: {int(dup.sum())}")
    out = df.merge(pit_sel, on=["session_key", "driver_number", "lap"], how="left")
    sessions_with_pit = set(pits["session_key"].unique())
    out["match_status"] = np.where(out["date"].notna(), "matched",
                                   np.where(out["session_key"].isin(sessions_with_pit),
                                            "no_record", "no_pit_cache"))
    out = out.rename(columns={"date": "pit_date"})
    return out

def associate_events(matched: pd.DataFrame, eps: pd.DataFrame,
                     raw_laps: pd.DataFrame) -> pd.DataFrame:
    """把每次匹配到的进站归到同场同类型的某一段事件，并算 dt = 进站 date − 事件部署 start_time。

    归属规则：优先取与该车进站圈（用该圈 date_start 到进站 date 的区间表示）重叠最多的事件段；
    没有重叠时取之后最早的一段（说明是在宣布前进站），再否则取之前最近的一段。
    """
    rows = []
    for r in matched.itertuples(index=False):
        rec = {
            "year": int(r.year), "round": int(r.round), "driver": r.driver,
            "driver_number": int(r.driver_number),
            "lap": int(r.lap), "condition": r.condition,
            "relative_loss_s": float(r.relative_loss_s),
            "session_key": int(r.session_key),
            "match_status": r.match_status,
            "pit_date": r.pit_date if pd.notna(r.pit_date) else pd.NaT,
            "lane_duration": r.lane_duration if pd.notna(r.lane_duration) else np.nan,
            "stop_duration": r.stop_duration if pd.notna(r.stop_duration) else np.nan,
            "event_start_lap": np.nan, "event_end_lap": np.nan,
            "episode_start_ts": pd.NaT, "assoc": None, "dt_s": np.nan,
        }
        if r.match_status != "matched":
            rows.append(rec)
            continue
        cand = eps[(eps["session_key"].astype(int) == int(r.session_key))
                   & (eps["type"] == r.condition)]
        ls = raw_laps[(raw_laps["session_key"] == int(r.session_key))
                      & (raw_laps["driver_number"] == int(r.driver_number))
                      & (raw_laps["lap"] == int(r.lap))]["lap_start"]
        lap_start = ls.iloc[0] if len(ls) else pd.NaT
        best = None
        best_ov = 0.0
        for x in cand.itertuples(index=False):
            st = x.start_ts
            et = x.end_ts if pd.notna(x.end_ts) else st + pd.Timedelta(seconds=600)
            if pd.notna(lap_start):
                ov = (min(et, r.pit_date) - max(st, lap_start)).total_seconds()
            else:
                ov = -abs((r.pit_date - st).total_seconds())
            if ov > best_ov:
                best_ov = ov
                best = x
        if best is None:
            future = cand[cand["start_ts"] >= r.pit_date]
            if len(future):
                best = future.sort_values("start_ts").iloc[0]
                rec["assoc"] = "future"
            else:
                past = cand[cand["end_ts"] <= (lap_start if pd.notna(lap_start) else r.pit_date)]
                if len(past):
                    best = past.sort_values("end_ts").iloc[-1]
                    rec["assoc"] = "past"
        if best is not None:
            rec["event_start_lap"] = int(best.start_lap)
            rec["event_end_lap"] = int(best.end_lap)
            rec["episode_start_ts"] = best.start_ts
            rec["dt_s"] = (r.pit_date - best.start_ts).total_seconds()
            if rec["assoc"] is None:
                rec["assoc"] = "overlap"
        rows.append(rec)
    return pd.DataFrame(rows)

def add_lap_length(detail: pd.DataFrame, laps: pd.DataFrame) -> pd.DataFrame:
    """每个进站用该车此前最近一个干净圈的圈时作为圈长；没有干净圈时用该车本场干净圈中位数。"""
    lf = laps[["year", "round", "driver", "lap", "lap_time_s", "is_clean"]].copy()
    lf["is_clean"] = lf["is_clean"].fillna(False).astype(bool)
    prev_clean = []
    fallback = []
    for r in detail.itertuples(index=False):
        sub = lf[(lf["year"] == r.year) & (lf["round"] == r.round)
                 & (lf["driver"] == r.driver) & (lf["lap"] < r.lap) & (lf["is_clean"])]
        if len(sub):
            prev_clean.append(float(sub.sort_values("lap").iloc[-1]["lap_time_s"]))
            fallback.append(False)
        else:
            med = lf[(lf["year"] == r.year) & (lf["round"] == r.round)
                     & (lf["driver"] == r.driver) & (lf["is_clean"])]["lap_time_s"].median()
            prev_clean.append(float(med) if pd.notna(med) else np.nan)
            fallback.append(True)
    detail = detail.copy()
    detail["lap_len_s"] = prev_clean
    detail["lap_len_fallback"] = fallback
    return detail

def dt_group(dt, lap_len) -> str:
    if pd.isna(dt):
        return "无法计算"
    if dt < 0:
        return "宣布前进站"
    if dt < 60:
        return "宣布后0-60秒"
    if pd.isna(lap_len):
        return "无圈长参考"
    if dt < lap_len:
        return "60秒-1圈"
    if dt < 2 * lap_len:
        return "1-2圈"
    return "2圈以后"

def add_groups(detail: pd.DataFrame) -> pd.DataFrame:
    detail = detail.copy()
    detail["dt_group"] = [dt_group(d, l) for d, l in zip(detail["dt_s"], detail["lap_len_s"])]
    return detail

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rel, eps, laps, pits, raw_laps, rc, sessions = load_data()
    smap = build_session_map(sessions, eps)

    cohort = rel[(rel["relative_usable"].fillna(False).astype(bool))
                 & (rel["condition"].isin(["SC", "VSC"]))].copy()
    matched = match_pits(cohort, pits, smap)
    detail = associate_events(matched, eps, raw_laps)
    detail = add_lap_length(detail, laps)
    detail = add_groups(detail)

    detail.to_csv(OUT / "pit_match_detail.csv", index=False)

if __name__ == "__main__":
    main()
