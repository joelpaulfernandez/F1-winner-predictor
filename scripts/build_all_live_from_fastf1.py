import subprocess
from pathlib import Path

import pandas as pd

RACES_PARQUET = Path("data/silver/races.parquet")
LIVE_DIR = Path("data/live_inputs")

def main():
    if not RACES_PARQUET.exists():
        raise SystemExit(f"Missing {RACES_PARQUET}, cannot list races.")

    LIVE_DIR.mkdir(parents=True, exist_ok=True)

    races = pd.read_parquet(RACES_PARQUET)
    races = races[races["year"] >= 2022].copy()
    races = races.sort_values(["year", "roundnumber"])

    for _, row in races.iterrows():
        year = int(row["year"])
        rnd = int(row["roundnumber"])
        eventname = str(row["eventname"])
        race_id = f"{year}_{rnd}"
        out_csv = LIVE_DIR / f"{race_id}.csv"

        print(f"🟦 Building live CSV for {year} round {rnd}: {eventname} -> {out_csv}")

        cmd = [
            "python",
            "scripts/build_live_from_fastf1.py",
            "--year", str(year),
            "--event-name", eventname,
            "--race-id", race_id,
            "--out-csv", str(out_csv),
        ]
        subprocess.run(cmd, check=True)

    print("Done.")

if __name__ == "__main__":
    main()