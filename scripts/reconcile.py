"""Compare all cached OpenF1 driver lap counts against Jolpica results."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import pandas as pd

from f1a.openf1 import calendar, results
from scripts.build_cache import atomic_write

YEARS = (2023, 2024, 2025, 2026)
PROCESSED = Path("data/processed")
COLUMNS = ["year", "round", "driver", "laps_openf1", "laps_jolpica", "diff"]


def compare(year, round_number, laps, classifications):
    counts = laps.groupby("driver_number").size().to_dict() if not laps.empty else {}
    names = (laps.groupby("driver_number")["driver"].first().to_dict()
             if not laps.empty else {})
    published = {int(item["number"]): item for item in classifications}
    rows = []
    for number in sorted(counts.keys() | published.keys()):
        record = published.get(number, {})
        name = record.get("Driver", {}).get("code") or names.get(number)
        a = counts.get(number, 0) if not laps.empty else None
        b = record.get("laps")
        b = int(b) if b not in (None, "") else None
        rows.append((year, round_number, name if pd.notna(name) else str(number),
                     a, b, a - b if a is not None and b is not None else None))
    return rows


def main():
    rows, coverage = [], []
    errors = 0
    now = pd.Timestamp.now(tz="UTC")
    for year in YEARS:
        path = PROCESSED / f"laps_{year}.parquet"
        season = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        groups = dict(tuple(season.groupby("round"))) if not season.empty else {}
        schedule = calendar(year)
        rounds = {int(race["round"]): race for race in schedule}
        for number in sorted(rounds.keys() | groups.keys()):
            race = rounds.get(number, {})
            laps = groups.get(number, pd.DataFrame())
            classifications = []
            start = pd.Timestamp(f"{race['date']}T{race.get('time') or '23:59:59Z'}") if race else None
            status = "compared"
            if start is not None and start >= now:
                status = "not_held"
            else:
                try:
                    classifications = results(year, number)
                    if not classifications:
                        raise ValueError("No published results")
                except Exception as exc:
                    status = f"results_unavailable: {type(exc).__name__}: {exc}"
                    errors += 1
                else:
                    if laps.empty:
                        status = "openf1_unavailable"
            coverage.append((year, number, race.get("raceName", ""), status))
            rows.extend(compare(year, number, laps, classifications))
        print(f"{year}: {len(groups)}/{len(rounds)} cached rounds", flush=True)
    frame = pd.DataFrame(rows, columns=COLUMNS).astype(
        {name: "Int64" for name in COLUMNS[3:]}
    )
    atomic_write(frame, PROCESSED / "reconcile.csv")
    audit = pd.DataFrame(coverage, columns=["year", "round", "race", "status"])
    atomic_write(audit, PROCESSED / "reconcile_coverage.csv")
    print(audit["status"].value_counts().to_string())
    print(f"{len(frame)} driver rows; {frame['diff'].eq(0).sum()} equal; "
          f"{frame['diff'].dropna().ne(0).sum()} different; "
          f"{frame['diff'].isna().sum()} not comparable")
    return int(errors > 0)


if __name__ == "__main__":
    raise SystemExit(main())
