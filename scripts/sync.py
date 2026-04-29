"""Sync live-input CSVs for the current (or given) F1 season.

Usage:
    python scripts/sync.py              # current year
    python scripts/sync.py --year 2025  # specific year
    python scripts/sync.py --all        # all years from 2022

Checks which races have already happened, skips ones that already have a CSV,
and fetches the rest. Run this once after each race weekend.
"""
import argparse
import datetime
from pathlib import Path

import fastf1
import pandas as pd

from scripts.build_live_from_fastf1 import build_live_from_fastf1

fastf1.Cache.enable_cache("data_cache")

LIVE_DIR = Path("data/live_inputs")
START_YEAR = 2022


def sync_year(year: int) -> tuple[int, int]:
    """Sync all completed races for one season. Returns (added, skipped)."""
    print(f"\n── {year} ──────────────────────────────")
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        print(f"  Could not fetch schedule: {e}")
        return 0, 0

    today = datetime.date.today()
    added = skipped = 0

    for _, event in schedule.iterrows():
        event_date = pd.to_datetime(event["EventDate"]).date()
        if event_date > today:
            print(f"  skip  {event['EventName']} (not yet raced: {event_date})")
            continue

        rnd = int(event["RoundNumber"])
        race_id = f"{year}_{rnd}"
        out_csv = LIVE_DIR / f"{race_id}.csv"

        if out_csv.exists():
            print(f"  ok    {race_id}  {event['EventName']}")
            skipped += 1
            continue

        print(f"  fetch {race_id}  {event['EventName']} ...", end=" ", flush=True)
        try:
            build_live_from_fastf1(
                year=year,
                event_name=event["EventName"],
                race_id=race_id,
                out_csv=out_csv,
            )
            print("done")
            added += 1
        except Exception as e:
            print(f"FAILED: {e}")

    return added, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync F1 live-input CSVs from FastF1.")
    parser.add_argument("--year", type=int, default=datetime.date.today().year,
                        help="Season year (default: current year)")
    parser.add_argument("--all", action="store_true",
                        help=f"Sync all seasons from {START_YEAR} to current year")
    args = parser.parse_args()

    LIVE_DIR.mkdir(parents=True, exist_ok=True)

    years = range(START_YEAR, datetime.date.today().year + 1) if args.all else [args.year]

    total_added = total_skipped = 0
    for year in years:
        added, skipped = sync_year(year)
        total_added += added
        total_skipped += skipped

    print(f"\nDone. {total_added} new CSV(s) added, {total_skipped} already up to date.")


if __name__ == "__main__":
    main()
