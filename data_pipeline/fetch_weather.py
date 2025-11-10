# data_pipeline/fetch_weather.py
import fastf1
import pandas as pd
from pathlib import Path

fastf1.Cache.enable_cache("data_cache")

races = pd.read_parquet("data/silver/races.parquet")[["raceid","eventname","year"]].drop_duplicates()
rows = []

for raceid, ev, yr in races.itertuples(index=False):
    print(f"Weather → {yr} {ev} ({raceid})")
    try:
        ses = fastf1.get_session(int(yr), ev, "R")  # race session
        ses.load()
        wd = ses.weather_data.copy()
        if wd is None or wd.empty:
            continue
        # Aggregate to simple pre-model stats (you can expand later)
        rain_prob = (wd["Rainfall"] > 0).mean() if "Rainfall" in wd else None
        rows.append({
            "raceid": raceid,
            "airtempmean": float(wd["AirTemp"].mean()) if "AirTemp" in wd else None,
            "tracktempmean": float(wd["TrackTemp"].mean()) if "TrackTemp" in wd else None,
            "humidity_mean": float(wd["Humidity"].mean()) if "Humidity" in wd else None,
            "rainprobproxy": float(rain_prob) if rain_prob is not None else None,
            "windspeed_mean": float(wd["WindSpeed"].mean()) if "WindSpeed" in wd else None,
        })
    except Exception as e:
        print(f"  ⚠️ skipped {yr} {ev}: {e}")

out = pd.DataFrame(rows)
Path("data/silver").mkdir(parents=True, exist_ok=True)
out.to_parquet("data/silver/weather.parquet", index=False)
print(f"✅ wrote data/silver/weather.parquet [{len(out)} rows]")