"""T007 stint-boundary inference; never overwrite provider pit flags."""
import pandas as pd


def infer_pit_in(df):
    """Mark each observed stint's last lap, excluding the driver's last row/lap."""
    identity = "driver_number" if "driver_number" in df else "driver"
    keys = ["year", "round", identity]
    last_lap = df.groupby(keys)["lap"].transform("max")
    stint_end = df.groupby(keys + ["stint"])["lap"].transform("max")
    return (df["stint"].notna() & df["lap"].eq(stint_end)
            & df["lap"].ne(last_lap)).fillna(False).astype(bool)


def fill_pit_in(df):
    """Infer only in races with entirely missing pit_in; preserve partial gaps."""
    result = df.copy()
    missing_race = df["pit_in"].isna().groupby(
        [df["year"], df["round"]]).transform("all")
    result["pit_in_filled"] = df["pit_in"].astype("boolean")
    result["pit_in_inferred"] = False
    if missing_race.any():
        predicted = infer_pit_in(df)
        # Unknown stint rows remain unknown, rather than asserting no stop.
        known = missing_race & df["stint"].notna()
        result.loc[known, "pit_in_filled"] = predicted.loc[known]
        result.loc[known & predicted, "pit_in_inferred"] = True
    return result


def validate_pit_in(df):
    """Micro precision/recall on known provider labels, including final laps."""
    predicted = infer_pit_in(df)
    known = df["pit_in"].notna()
    actual = df["pit_in"].astype("boolean").fillna(False).astype(bool)
    tp = int((known & predicted & actual).sum())
    fp = int((known & predicted & ~actual).sum())
    fn = int((known & ~predicted & actual).sum())
    return dict(tp=tp, fp=fp, fn=fn, evaluated=int(known.sum()),
                unknown=int((~known).sum()),
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None)
