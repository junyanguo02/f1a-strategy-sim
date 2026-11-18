"""Build simulator inputs from the local race cache."""
from __future__ import annotations

import glob

import json

from pathlib import Path

import numpy as np

import pandas as pd

COMPOUND_LETTER = {"SOFT": "S", "MEDIUM": "M", "HARD": "H",
                   "INTERMEDIATE": "I", "WET": "W"}

WET_COMPOUNDS = {"INTERMEDIATE", "WET"}

FINISHED_STATUSES = {"Finished", "Lapped"}

MAINSTREAM_SHARE = 0.50    # 主流策略：最常见策略占比 >= 50%

DISPERSED_SHARE = 0.40     # 策略分散：最常见策略占比 < 40%

MAX_STINTS = 8

HERE = Path(__file__).resolve().parent

ROOT = HERE.parents[1]

PROCESSED = ROOT / "data" / "processed"

RAW = ROOT / "data" / "raw" / "openf1"

JOLPICA = ROOT / "data" / "raw" / "jolpica"

LAPS = PROCESSED / "laps_pit_filled.parquet"

EPISODES = PROCESSED / "fcy_episodes.parquet"

SUMMARY = PROCESSED / "fcy_race_summary.csv"

QUALITY = PROCESSED / "race_quality.csv"

OUT = HERE / "outputs"

def strategy_letter(compound) -> str:
    if compound is None or (isinstance(compound, float) and pd.isna(compound)):
        return "?"
    return COMPOUND_LETTER.get(str(compound).upper(), "?")

def status_class(status) -> str:
    if status in FINISHED_STATUSES:
        return "Finisher"
    if status == "Disqualified":
        return "DSQ"
    if status in ("Did not start", "Withdrew"):
        return "DNS"
    if status in ("Retired", "Accident", "Collision damage", "Undertray",
                  "Engine", "Gearbox", "Brakes", "Suspension", "Power Unit",
                  "Electrical", "Hydraulics", "Overheating", "Fuel", "Oil"):
        return "DNF"
    return "Other"

def load_data():
    laps = pd.read_parquet(LAPS)
    for c in ["pit_in_filled", "safety_car", "virtual_safety_car"]:
        laps[c] = laps[c].fillna(False).astype(bool)
    laps["track_status"] = laps["track_status"].astype("string")

    episodes = pd.read_parquet(EPISODES)
    summary = pd.read_csv(SUMMARY)
    quality = pd.read_csv(QUALITY)

    stints = pd.concat(
        [pd.DataFrame(json.load(open(f, encoding="utf-8")))
         for f in glob.glob(str(RAW / "stints_session_key=*.json"))],
        ignore_index=True,
    )

    jrows = []
    for f in glob.glob(str(JOLPICA / "*_results_limit=100.json")):
        stem = Path(f).name.replace("_results_limit=100.json", "")
        y, r = stem.split("_")
        race = json.load(open(f, encoding="utf-8"))["MRData"]["RaceTable"]["Races"][0]
        for x in race["Results"]:
            jrows.append({
                "year": int(y), "round": int(r),
                "driver": x["Driver"].get("code"),
                "constructor": x["Constructor"]["name"],
                "finish_status": x["status"],
                "finish_positionText": str(x.get("positionText")),
                "finish_position_raw": x.get("position"),
                "finish_laps": int(x.get("laps", 0) or 0),
            })
    jolpica = pd.DataFrame(jrows)
    return laps, episodes, summary, quality, stints, jolpica

def driver_map(laps: pd.DataFrame) -> pd.DataFrame:
    return laps[["year", "round", "driver", "driver_number", "team"]].drop_duplicates()

def prepare_raw_stints(stints: pd.DataFrame, summary: pd.DataFrame,
                       dmap: pd.DataFrame) -> pd.DataFrame:
    """OpenF1 stint 原始记录 → 带 year/round/driver/team 的干净表。"""
    st = stints.copy()
    st = st.merge(summary[["year", "round", "session_key"]], on="session_key", how="left")
    st = st.merge(dmap, on=["year", "round", "driver_number"], how="left")
    st["valid"] = (st["driver"].notna()
                   & st["lap_end"].notna()
                   & st["lap_start"].notna()
                   & (st["lap_start"] >= 1)
                   & (st["lap_end"] >= st["lap_start"])
                   & (st["lap_end"] > 0))
    st["stint_number"] = st["stint_number"].astype(int)
    st["strategy_letter"] = st["compound"].map(strategy_letter)
    return st

def episode_lookup(episodes: pd.DataFrame) -> dict:
    d = {}
    for race_key, g in episodes[episodes["type"].isin(["SC", "VSC"])].groupby(["year", "round"]):
        d[race_key] = g[["type", "start_lap", "end_lap"]].to_dict("records")
    return d

def episode_label(look: dict, year: int, rnd: int, lap: int) -> str:
    """pit 是否在 fcy_episodes 的 SC/VSC 事件段内（按圈号，允许 ±1 圈的位置偏差）。"""
    cand = look.get((year, rnd), [])
    for off in (1, 0):   # 允许进站圈比事件段晚 1 圈（车手自己的圈号与领先者圈号错位）
        hit = [x for x in cand if x["start_lap"] <= lap + off <= x["end_lap"]]
        if hit:
            types = {x["type"] for x in hit}
            return "SC/VSC" if len(types) > 1 else hit[0]["type"]
    # out-lap 落入事件段
    for x in cand:
        if x["start_lap"] <= lap + 1 <= x["end_lap"]:
            return x["type"]
    return ""

def flag_label(laps: pd.DataFrame) -> dict:
    """每次进站的 SC/VSC 标签：进站圈或出站圈的旗标（laps_pit_filled 的 safety_car/virtual）。"""
    df = laps[["year", "round", "driver", "lap", "safety_car", "virtual_safety_car"]].copy()
    nxt = df[["year", "round", "driver", "lap", "safety_car", "virtual_safety_car"]].copy()
    nxt["lap"] = nxt["lap"] - 1
    nxt = nxt.rename(columns={"safety_car": "out_sc", "virtual_safety_car": "out_vsc"})
    df = df.merge(nxt, on=["year", "round", "driver", "lap"], how="left")
    df[["out_sc", "out_vsc"]] = df[["out_sc", "out_vsc"]].apply(
        lambda x: x.map(lambda v: bool(v) if pd.notna(v) else False))
    out = {}
    for r in df.itertuples(index=False):
        sc = bool(r.safety_car) or bool(r.out_sc)
        vsc = bool(r.virtual_safety_car) or bool(r.out_vsc)
        if sc and vsc:
            out[(int(r.year), int(r.round), str(r.driver), int(r.lap))] = "SC/VSC"
        elif sc:
            out[(int(r.year), int(r.round), str(r.driver), int(r.lap))] = "SC"
        elif vsc:
            out[(int(r.year), int(r.round), str(r.driver), int(r.lap))] = "VSC"
    return out

def build_strategies(laps, summary, quality, raw, dmap, jolpica, ep_look):
    """每行 = 一场比赛的一辆车；stint 结构来自 OpenF1 原始 stint 记录。"""
    fl = flag_label(laps)
    races = summary[["year", "round", "event", "circuit_id", "session_key",
                     "is_complete", "is_wet", "total_laps"]].copy()
    races = races.merge(quality[["year", "round", "n_drivers", "median_coverage"]],
                        on=["year", "round"], how="left")
    red_races = set(
        map(tuple, laps[laps["track_status"].str.contains("5", na=False)]
            [["year", "round"]].drop_duplicates().values)
    )
    races["has_red_flag"] = [(int(y), int(r)) in red_races
                             for y, r in zip(races["year"], races["round"])]

    participants = dmap.copy()
    raw_ok = raw[raw["valid"]].sort_values(
        ["year", "round", "driver", "stint_number"]).copy()
    raw_ok = raw_ok.merge(races[["year", "round", "event", "circuit_id", "session_key",
                                 "is_complete", "is_wet", "total_laps", "has_red_flag"]],
                          on=["year", "round"], how="left")

    rows = []
    for r in participants.itertuples(index=False):
        key = (int(r.year), int(r.round), str(r.driver))
        rec = {
            "year": int(r.year), "round": int(r.round), "driver": str(r.driver),
            "driver_number": int(r.driver_number), "team": r.team,
            "has_stint_data": False, "n_stints": 0,
            "start_compound": None, "start_tyre_age": np.nan,
            "start_tyre_status": "未知", "strategy": "",
            "n_pit_stops": 0, "pit_laps": "", "pit_under_sc_vsc": "",
            "pit_under_flags": "", "n_pits_sc": 0, "n_pits_vsc": 0,
            "wet_tyre_used": False, "raw_first_stint_number": np.nan,
            "stint_number_gap": False,
        }
        st = raw_ok[(raw_ok["year"] == rec["year"])
                    & (raw_ok["round"] == rec["round"])
                    & (raw_ok["driver"] == rec["driver"])]
        if len(st):
            st = st.sort_values("stint_number")
            rec["has_stint_data"] = True
            rec["n_stints"] = int(len(st))
            rec["raw_first_stint_number"] = int(st["stint_number"].min())
            rec["stint_number_gap"] = bool(
                int(st["stint_number"].min()) > 1
                or list(st["stint_number"]) != list(range(int(st["stint_number"].min()),
                                                          int(st["stint_number"].max()) + 1)))
            first = st.iloc[0]
            rec["start_compound"] = first["compound"]
            rec["start_tyre_age"] = (float(first["tyre_age_at_start"])
                                     if pd.notna(first["tyre_age_at_start"]) else np.nan)
            if pd.isna(rec["start_tyre_age"]):
                rec["start_tyre_status"] = "未知"
            elif float(rec["start_tyre_age"]) == 0:
                rec["start_tyre_status"] = "新胎"
            else:
                rec["start_tyre_status"] = "旧胎"
            rec["strategy"] = "".join(st["strategy_letter"].astype(str))
            rec["wet_tyre_used"] = bool(st["compound"].isin(WET_COMPOUNDS).any())
            pit_laps = [int(x) for x in list(st["lap_end"])[:-1]]
            rec["n_pit_stops"] = len(pit_laps)
            rec["pit_laps"] = ";".join(str(x) for x in pit_laps)
            ep_labels = [episode_label(ep_look, rec["year"], rec["round"], L) for L in pit_laps]
            fg_labels = [fl.get((rec["year"], rec["round"], rec["driver"], L), "")
                         for L in pit_laps]
            rec["pit_under_sc_vsc"] = ";".join(x if x else "-" for x in ep_labels)
            rec["pit_under_flags"] = ";".join(x if x else "-" for x in fg_labels)
            rec["n_pits_sc"] = sum(1 for x in ep_labels if x in ("SC", "SC/VSC"))
            rec["n_pits_vsc"] = sum(1 for x in ep_labels if x in ("VSC", "SC/VSC"))
            # 宽表 stint 明细
            for i in range(MAX_STINTS):
                if i < len(st):
                    row = st.iloc[i]
                    rec[f"stint{i + 1}_compound"] = row["compound"]
                    rec[f"stint{i + 1}_start"] = int(row["lap_start"])
                    rec[f"stint{i + 1}_end"] = int(row["lap_end"])
                    rec[f"stint{i + 1}_laps"] = int(row["lap_end"]) - int(row["lap_start"]) + 1
                    rec[f"stint{i + 1}_age"] = (float(row["tyre_age_at_start"])
                                                if pd.notna(row["tyre_age_at_start"]) else np.nan)
                else:
                    rec[f"stint{i + 1}_compound"] = None
                    rec[f"stint{i + 1}_start"] = np.nan
                    rec[f"stint{i + 1}_end"] = np.nan
                    rec[f"stint{i + 1}_laps"] = np.nan
                    rec[f"stint{i + 1}_age"] = np.nan
        else:
            for i in range(MAX_STINTS):
                rec[f"stint{i + 1}_compound"] = None
                rec[f"stint{i + 1}_start"] = np.nan
                rec[f"stint{i + 1}_end"] = np.nan
                rec[f"stint{i + 1}_laps"] = np.nan
                rec[f"stint{i + 1}_age"] = np.nan
        rec["strategy_len"] = len(rec["strategy"])
        rows.append(rec)
    df = pd.DataFrame(rows)

    # 比赛属性
    df = df.merge(races, on=["year", "round"], how="left")
    # jolpica
    df = df.merge(
        jolpica[["year", "round", "driver", "constructor", "finish_status",
                 "finish_positionText", "finish_position_raw", "finish_laps"]],
        on=["year", "round", "driver"], how="left")
    df["finish_position"] = pd.to_numeric(
        df["finish_position_raw"], errors="coerce")
    df.loc[~df["finish_status"].isin(FINISHED_STATUSES), "finish_position"] = np.nan
    df["status_class"] = df["finish_status"].map(status_class)
    df["finisher"] = df["finish_status"].isin(FINISHED_STATUSES)
    df["race_usable"] = df["is_complete"].fillna(False).astype(bool)
    df["race_has_red_flag"] = df["has_red_flag"].fillna(False).astype(bool)
    df = df.rename(columns={"is_wet": "race_is_wet"})
    df = df.sort_values(["year", "round", "finish_position", "driver"],
                        kind="stable").reset_index(drop=True)
    return df

def race_tables(df: pd.DataFrame, summary: pd.DataFrame, episodes: pd.DataFrame):
    races = summary[["year", "round", "event", "circuit_id", "is_complete", "is_wet"]].copy()
    wet_tyre_used = df.groupby(["year", "round"])["wet_tyre_used"].any().rename("wet_tyre_used_any").reset_index()
    races = races.merge(wet_tyre_used, on=["year", "round"], how="left")
    races["dry_for_analysis"] = (~races["is_wet"].fillna(False)) & (~races["wet_tyre_used_any"].fillna(False))
    n_sc = episodes[episodes["type"] == "SC"].groupby(["year", "round"]).size()
    n_vsc = episodes[episodes["type"] == "VSC"].groupby(["year", "round"]).size()
    red_races = set(
        map(tuple, df[df["race_has_red_flag"]][["year", "round"]].drop_duplicates().values))
    dry_map = {(int(r.year), int(r.round)): bool(r.dry_for_analysis)
               for r in races.itertuples(index=False)}

    share_rows, top10_rows, race_rows = [], [], []
    for r in races.itertuples(index=False):
        key = (int(r.year), int(r.round))
        g = df[(df["year"] == key[0]) & (df["round"] == key[1])]
        fin = g[g["finisher"] & dry_map.get(key, False) & g["has_stint_data"]]
        counts = fin["strategy"].value_counts()
        n_fin = int(len(fin))
        n_strat = int(counts.size)
        top = counts.index[0] if n_strat else None
        top_share = float(counts.iloc[0] / n_fin) if n_fin else np.nan
        for strat, cnt in counts.items():
            share_rows.append({"year": key[0], "round": key[1], "strategy": strat,
                               "n": int(cnt), "share": float(cnt / n_fin)})
        top10 = fin[fin["finish_position"].between(1, 10)].sort_values("finish_position")
        for x in top10.itertuples(index=False):
            top10_rows.append({"year": key[0], "round": key[1],
                               "position": int(x.finish_position), "driver": x.driver,
                               "strategy": x.strategy})
        t10_counts = top10["strategy"].value_counts()
        sc_events = int(n_sc.get(key, 0))
        vsc_events = int(n_vsc.get(key, 0))
        sc_pits = int(g["n_pits_sc"].sum())
        vsc_pits = int(g["n_pits_vsc"].sum())
        total_pits = int(g["n_pit_stops"].sum())
        race_rows.append({
            "year": key[0], "round": key[1], "event": r.event,
            "circuit_id": r.circuit_id,
            "usable": bool(r.is_complete),
            "is_wet": bool(r.is_wet),
            "wet_tyre_used_any": bool(r.wet_tyre_used_any),
            "dry_for_analysis": bool(r.dry_for_analysis),
            "has_red_flag": key in red_races,
            "n_starters": int(len(g)),
            "n_finishers": int(g["finisher"].sum()),
            "n_dry_finishers": n_fin,
            "n_strategies": n_strat,
            "top_strategy": top,
            "top_share": top_share,
            "mainstream": bool(n_fin and top_share >= MAINSTREAM_SHARE),
            "dispersed": bool(n_fin and top_share < DISPERSED_SHARE),
            "n_sc_events": sc_events, "n_vsc_events": vsc_events,
            "n_sc_pits": sc_pits, "n_vsc_pits": vsc_pits,
            "n_total_pits": total_pits,
            "sc_vsc_pit_share": float((sc_pits + vsc_pits) / total_pits) if total_pits else 0.0,
            "top10_available": int(len(top10)),
            "top10_dominant": t10_counts.index[0] if len(t10_counts) else None,
            "top10_top_share": float(t10_counts.iloc[0] / len(top10)) if len(top10) else np.nan,
            "top10_n_strategies": int(t10_counts.size),
        })
    shares = pd.DataFrame(share_rows)
    top10 = pd.DataFrame(top10_rows)
    race_tab = pd.DataFrame(race_rows).sort_values(["year", "round"]).reset_index(drop=True)
    return shares, top10, race_tab

def build_stints_long(raw: pd.DataFrame) -> pd.DataFrame:
    st = raw[raw["valid"]].sort_values(
        ["year", "round", "driver", "stint_number"]).copy()
    st["n_laps"] = st["lap_end"] - st["lap_start"] + 1
    st = st.rename(columns={"tyre_age_at_start": "start_tyre_age"})
    return st[["year", "round", "driver", "driver_number", "team", "stint_number",
               "compound", "strategy_letter", "lap_start", "lap_end", "n_laps",
               "start_tyre_age"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    laps, episodes, summary, quality, stints_raw, jolpica = load_data()
    dmap = driver_map(laps)
    raw = prepare_raw_stints(stints_raw, summary, dmap)
    ep_look = episode_lookup(episodes)

    df = build_strategies(laps, summary, quality, raw, dmap, jolpica, ep_look)
    shares, top10, race_tab = race_tables(df, summary, episodes)
    build_stints_long(raw).to_csv(HERE / "stints.csv", index=False)
    df.to_csv(HERE / "strategies.csv", index=False)
    df.to_parquet(HERE / "strategies.parquet", index=False)
    race_tab.to_csv(OUT / "race_summary.csv", index=False)

if __name__ == "__main__":
    main()
