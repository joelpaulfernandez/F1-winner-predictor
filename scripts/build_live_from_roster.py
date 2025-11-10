# scripts/build_live_from_roster.py
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def main(year: int, event_name: str, race_id: str, out_csv: str):
    rr = pd.read_parquet("data/silver/race_results.parquet")

    # Make sure these exist numerically for sorting
    for c in ["year", "roundnumber"]:
        if c in rr.columns:
            rr[c] = pd.to_numeric(rr[c], errors="coerce")

    # take each driver's most recent appearance (gives driver_id + team_id)
    latest = (
        rr.sort_values(["year", "roundnumber"])
          .groupby("driver_id", as_index=False, sort=False)
          .tail(1)[["driver_id", "team_id"]]
          .drop_duplicates()
    )

    # ensure we actually have 20 rows (if not, still proceed with what we have)
    print(f"Found {len(latest)} drivers from most recent race in your data.")

    # placeholders for pre-race inputs (you can edit later if you want)
    latest["raceid"] = race_id
    latest["eventname"] = event_name
    latest["gridposition"] = np.nan
    latest["quali_best_ms"] = np.nan
    latest["quali_gap_to_pole_ms"] = np.nan
    latest["quali_rank"] = np.nan
    latest["airtempmean"] = np.nan
    latest["tracktempmean"] = np.nan
    latest["humidity_mean"] = np.nan
    latest["rainprobproxy"] = np.nan
    latest["windspeed_mean"] = np.nan

    cols = [
        "raceid","driver_id","team_id","eventname",
        "gridposition","quali_best_ms","quali_gap_to_pole_ms","quali_rank",
        "airtempmean","tracktempmean","humidity_mean","rainprobproxy","windspeed_mean",
    ]
    latest = latest[cols].sort_values("driver_id").reset_index(drop=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    latest.to_csv(out_csv, index=False)
    print(f"🗂  wrote {out_csv} with {len(latest)} drivers")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--event-name", required=True)
    ap.add_argument("--race-id", required=True)
    ap.add_argument("--out-csv", default="data/live_inputs/auto_live.csv")
    args = ap.parse_args()
    main(args.year, args.event_name, args.race_id, args.out_csv)