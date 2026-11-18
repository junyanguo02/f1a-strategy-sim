"""FastF1 race data; paths are relative to the project working directory."""
from pathlib import Path
import fastf1
import pandas as pd

REQUIRED = ["driver", "team", "lap", "lap_time_s", "compound", "tyre_life",
            "stint", "pit_in", "pit_out", "track_status", "air_temp", "track_temp"]


def enable_cache():
    """Keep upstream responses inside the project, never the global cache."""
    path = Path("data/raw/fastf1-cache")
    path.mkdir(parents=True, exist_ok=True)
    # FastF1 >=3.7 otherwise stores authentication state outside the project.
    # Keep even optional account state local; no credentials are supplied here.
    try:
        from fastf1.internals.f1account import token_manager
    except ImportError:
        pass  # FastF1 3.6 has no account token manager.
    else:
        token_manager._token_path = path / "f1auth_token.json"
    fastf1.Cache.enable_cache(str(path))


def load_race(year: int, round: int, source: str = "openf1") -> pd.DataFrame:
    """Return normalized race laps from the selected source."""
    if source == "openf1":
        from f1a.openf1 import load_openf1
        return load_openf1(year, round)
    if source == "fastf1":
        return _load_fastf1(year, round)
    raise ValueError(f"Unknown race source: {source}")


def _load_fastf1(year: int, round: int) -> pd.DataFrame:
    """Load every upstream race lap, retaining missing and unusual laps.

    pit_in/out are boolean flags; track_status retains FastF1's combined codes.
    Weather uses FastF1's first sample in each lap (previous sample fallback).
    No deletion, imputation, or accurate-lap filtering is applied.
    """
    enable_cache()
    session = fastf1.get_session(year, round, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    laps = session.laps
    if laps.empty:
        raise ValueError(f"No lap data for {year} round {round}")
    weather = laps.get_weather_data().reset_index(drop=True)
    frame = pd.DataFrame(laps).reset_index(drop=True)
    result = frame.rename(columns={
        "Driver": "driver", "Team": "team", "LapNumber": "lap",
        "Compound": "compound", "TyreLife": "tyre_life", "Stint": "stint",
        "TrackStatus": "track_status",
    })[["driver", "team", "lap", "compound", "tyre_life", "stint", "track_status"]].copy()
    result["lap_time_s"] = frame["LapTime"].dt.total_seconds()
    result["pit_in"] = frame["PitInTime"].notna()
    result["pit_out"] = frame["PitOutTime"].notna()
    result["air_temp"] = weather["AirTemp"]
    result["track_temp"] = weather["TrackTemp"]
    status = result["track_status"].astype("string")
    result["safety_car"] = status.str.contains("4", regex=False)
    result["virtual_safety_car"] = status.str.contains("[67]", regex=True)
    result["first_lap"] = result["lap"].eq(1)
    result["year"] = year
    result["round"] = round
    return result.sort_values(["driver", "lap"], kind="stable").reset_index(drop=True)
