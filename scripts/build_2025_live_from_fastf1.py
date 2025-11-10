import subprocess
from pathlib import Path

import fastf1

# Enable FastF1 cache (adjust path if your cache is elsewhere)
fastf1.Cache.enable_cache("data_cache")

LIVE_DIR = Path("data/live_inputs")


def main():
    LIVE_DIR.mkdir(parents=True, exist_ok=True)

    year = 2025

    # Try a reasonable range of rounds; FastF1 will tell us which ones exist.
    for rnd in range(1, 30):
        try:
            session = fastf1.get_session(year, rnd, "R")
            session.load(laps=False, telemetry=False)
        except Exception as e:
            print(f"⚠️ Skipping {year} round {rnd}: {e}")
            continue

        # Get a nice event name from the loaded session
        try:
            eventname = session.event["EventName"]
        except Exception:
            # Fallback if event dict is not present as expected
            eventname = getattr(session, "eventName", f"Round {rnd}")

        race_id = f"{year}_{rnd}"
        out_csv = LIVE_DIR / f"{race_id}.csv"

        if out_csv.exists():
            print(f"✅ {race_id} already exists at {out_csv}, skipping.")
            continue

        print(f"🟦 Building live CSV for {year} round {rnd}: {eventname} -> {out_csv}")

        cmd = [
            "python",
            "scripts/build_live_from_fastf1.py",
            "--year",
            str(year),
            "--event-name",
            str(eventname),
            "--race-id",
            race_id,
            "--out-csv",
            str(out_csv),
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed for {race_id} ({eventname}): {e}")
            continue

    print("Done building 2025 live inputs.")


if __name__ == "__main__":
    main()