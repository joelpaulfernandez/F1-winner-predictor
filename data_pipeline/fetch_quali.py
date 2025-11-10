# data_pipeline/fetch_quali.py
import fastf1
import pandas as pd
from pathlib import Path

fastf1.Cache.enable_cache("data_cache")

# Use the races already in your silver layer to know what to fetch
races = pd.read_parquet("data/silver/races.parquet")[["raceid","eventname","year"]].drop_duplicates()
rows = []

def to_ms(td):
    if pd.isna(td): return None
    # FastF1 returns pandas Timedelta for lap times
    return float(td.total_seconds() * 1000.0)

for raceid, ev, yr in races.itertuples(index=False):
    print(f"Quali → {yr} {ev} ({raceid})")
    try:
        ses = fastf1.get_session(int(yr), ev, "Q")  # classic quali
        ses.load(laps=True, telemetry=False)
        res = ses.results  # classification-like table
        # get best lap per driver from laps() (more robust than res.BestTime sometimes)
        best = (
            ses.laps[ses.laps["LapTime"].notna()]
            .groupby("DriverNumber")["LapTime"].min()
            .reset_index()
            .rename(columns={"DriverNumber":"driver_id","LapTime":"best_td"})
        )
        best["quali_best_ms"] = best["best_td"].apply(to_ms)
        # rank by best_ms within race
        best["quali_rank"] = best["quali_best_ms"].rank(method="min").astype("Int64")

        # Merge team names if you want (optional)
        best["raceid"] = raceid

        # gap to pole (min best)
        pole_ms = best["quali_best_ms"].min()
        best["quali_gap_to_pole_ms"] = best["quali_best_ms"] - pole_ms
        rows.append(best[["raceid","driver_id","quali_best_ms","quali_rank","quali_gap_to_pole_ms"]])
    except Exception as e:
        print(f"  ⚠️ skipped {yr} {ev}: {e}")

out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
    columns=["raceid","driver_id","quali_best_ms","quali_rank","quali_gap_to_pole_ms"]
)
Path("data/silver").mkdir(parents=True, exist_ok=True)
out.to_parquet("data/silver/quali_results.parquet", index=False)
print(f"✅ wrote data/silver/quali_results.parquet [{len(out)} rows]")