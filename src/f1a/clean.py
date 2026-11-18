"""Label laps that pass the T005 dry, uninterrupted pace filters."""
import pandas as pd
from f1a.pit import fill_pit_in


REASONS = (
    "first_two_laps", "pit_in", "pit_out", "safety_car", "virtual_safety_car",
    "unknown_safety_status", "red_flag", "wet_compound", "unknown_compound",
    "missing_lap_time", "slow_lap",
)


def clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve all rows and record every applicable reason, joined by |."""
    result = fill_pit_in(df)
    laps = result.reset_index(drop=True)
    sc = laps["safety_car"].astype("boolean")
    vsc = laps["virtual_safety_car"].astype("boolean")
    compound = laps["compound"].astype("string").str.strip().str.upper()
    masks = {
        "first_two_laps": laps["lap"].isin([1, 2]),
        "pit_in": laps["pit_in_filled"].eq(True),
        "pit_out": laps["pit_out"].eq(True),
        "safety_car": sc,
        "virtual_safety_car": vsc,
        "unknown_safety_status": sc.isna() | vsc.isna(),
        "red_flag": laps["track_status"].astype("string").str.contains("5", regex=False),
        "wet_compound": compound.isin(["INTERMEDIATE", "WET"]),
        "unknown_compound": compound.isna() | compound.isin(["UNKNOWN", ""]),
        "missing_lap_time": laps["lap_time_s"].isna(),
    }
    flags = pd.DataFrame(masks).fillna(False).astype(bool)
    eligible = ~flags.any(axis=1)
    identity = "driver_number" if "driver_number" in laps else "driver"
    keys = [laps["year"], laps["round"], laps[identity]]
    lap_time = laps["lap_time_s"].astype("Float64")
    median = lap_time.where(eligible).groupby(keys).transform("median")
    flags["slow_lap"] = eligible & lap_time.gt(median * 1.07).fillna(False)
    result["is_clean"] = (~flags.any(axis=1)).to_numpy()
    result["drop_reason"] = pd.array(
        ["|".join(reason for reason, active in zip(REASONS, row) if active)
         for row in flags.itertuples(index=False, name=None)], dtype="string"
    )
    return result
