"""Validate stored laps without network access, preserving source anomalies."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from f1a.data import REQUIRED


def main():
    paths = sorted(Path("data/processed").glob("laps_*.parquet"))
    assert paths, "No season parquet files found"
    for path in paths:
        laps = pd.read_parquet(path)
        assert not laps.empty, path
        assert set(REQUIRED) <= set(laps.columns), path
        for round_number, race in laps.groupby("round"):
            assert not race[REQUIRED].isna().all().any(), (path, round_number)
            assert race.sort_values("lap").groupby("driver")["stint"].apply(
                lambda stint: stint.dropna().is_monotonic_increasing
            ).all(), (path, round_number)
        print(f"{path.name}: {len(laps)} rows, {laps['round'].nunique()} rounds; schema/nonempty/stint checks passed")


if __name__ == "__main__":
    main()
