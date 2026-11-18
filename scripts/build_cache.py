"""Run from the project root: python scripts/build_cache.py --years 2024."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fastf1
import pandas as pd
from f1a.data import enable_cache, load_race
from f1a.openf1 import calendar


def atomic_write(frame, path):
    """Publish a complete season checkpoint only after serialization succeeds."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    temporary.replace(path)


def build_cache(years, source="openf1"):
    """Checkpoint each successful round; retry failures on the next invocation."""
    if source == "fastf1":
        enable_cache()
    output = Path("data/processed")
    output.mkdir(parents=True, exist_ok=True)
    failure_path = output / "failed.csv"
    failures = (pd.read_csv(failure_path).to_dict("records") if failure_path.exists() else [])
    total_failures = 0
    for year in years:
        suffix = "" if source == "openf1" else "_fastf1"
        path = output / f"laps_{year}{suffix}.parquet"
        existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        completed = set(existing["round"].unique()) if not existing.empty else set()
        try:
            if source == "openf1":
                schedule = pd.DataFrame([
                    {"RoundNumber": race["round"], "EventName": race["raceName"]}
                    for race in calendar(year)
                ])
            else:
                schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:
            failures.append(dict(year=year, round=0, event="schedule", error=f"{type(exc).__name__}: {exc}"))
            atomic_write(pd.DataFrame(failures), failure_path)
            print(f"{year} schedule FAILED: {exc}", flush=True)
            total_failures += 1
            continue
        for _, event in schedule.iterrows():
            round_number = int(event["RoundNumber"])
            name = event["EventName"]
            if round_number in completed:
                count = int(existing["round"].eq(round_number).sum())
                print(f"{year} R{round_number:02d} {name}: {count} laps (cached)", flush=True)
                continue
            try:
                laps = load_race(year, round_number, source=source)
                combined = pd.concat([existing, laps], ignore_index=True)
                atomic_write(combined, path)
                existing = combined
                completed.add(round_number)
                print(f"{year} R{round_number:02d} {name}: {len(laps)} laps", flush=True)
            except Exception as exc:
                total_failures += 1
                failures.append(dict(year=year, round=round_number, event=name, error=f"{type(exc).__name__}: {exc}"))
                print(f"{year} R{round_number:02d} {name}: FAILED {type(exc).__name__}: {exc}", flush=True)
            # Append-only failure history: successful retries do not erase evidence.
            atomic_write(pd.DataFrame(failures, columns=["year", "round", "event", "error"]), failure_path)
        print(f"{year}: {len(completed)}/{len(schedule)} rounds cached, {len(existing)} laps", flush=True)
    print(f"Finished: {total_failures} failures this run", flush=True)
    return total_failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2023, 2027)))
    parser.add_argument("--source", choices=["openf1", "fastf1"], default="openf1")
    args = parser.parse_args()
    return int(build_cache(args.years, source=args.source) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
