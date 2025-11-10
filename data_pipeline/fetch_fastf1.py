import fastf1
import pandas as pd
from pathlib import Path

fastf1.Cache.enable_cache('data_cache')

YEARS = [2022, 2023, 2024]
all_rows = []

for year in YEARS:
    schedule = fastf1.get_event_schedule(year)
    for _, race in schedule.iterrows():
        gp_name = race["EventName"]
        rnd = race["RoundNumber"]
        print(f"Fetching {year} {gp_name} (round {rnd}) ...")
        try:
            session = fastf1.get_session(year, gp_name, 'R')
            session.load(laps=True)
            results = session.results
            results["EventName"] = gp_name
            results["Year"] = year
            results["RoundNumber"] = rnd
            results["RaceId"] = f"{year}_{rnd}"
            all_rows.append(results)
        except Exception as e:
            print(f"⚠️ Skipped {gp_name}: {e}")

df = pd.concat(all_rows)
df.to_csv("data/raw/f1_raw.csv", index=False)
print(f"✅ Wrote data/raw/f1_raw.csv with {len(df)} rows")