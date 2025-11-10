import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import fastf1

# Enable FastF1 cache (adjust path if your cache is elsewhere)
fastf1.Cache.enable_cache("data_cache")


def _best_quali_ms(row: pd.Series) -> float:
    """Return the best qualifying time in milliseconds from Q1/Q2/Q3 columns.

    If none of Q1/Q2/Q3 are available, returns NaN.
    """
    times = []
    for col in ["Q1", "Q2", "Q3"]:
        if col in row.index:
            v = row[col]
            if pd.notna(v):
                times.append(v)
    if not times:
        return np.nan
    best = min(times)
    try:
        # FastF1 stores these as pandas Timedelta; convert to ms
        return best.total_seconds() * 1000.0
    except Exception:
        return np.nan


def build_live_from_fastf1(year: int, event_name: str, race_id: str, out_csv: Path) -> None:
    """Build a 'live-like' CSV for a race using FastF1.

    The primary goal is to ensure that we always get at least:
      - driver_id (driver number)
      - team_id (team name)
      - quali_rank (qualifying position)
      - gridposition (proxy from quali_rank if needed)

    Additional quali metrics (best lap, gap to pole) are included when available.
    """
    # --- Try Qualifying first ---
    quali_df = None
    try:
        q = fastf1.get_session(year, event_name, "Q")
        q.load(laps=False, telemetry=False)
        res = q.results

        # Basic identifiers
        cols = [
            "DriverNumber",
            "TeamName",
            "Position",
        ]
        # Q1/Q2/Q3 may not all exist depending on the year/session
        for c in ["Q1", "Q2", "Q3"]:
            if c in res.columns:
                cols.append(c)

        base = res[cols].copy()
        base = base.rename(
            columns={
                "DriverNumber": "driver_id",
                "TeamName": "team_id",
                "Position": "quali_rank",
            }
        )

        # Ensure numeric driver_id
        base["driver_id"] = pd.to_numeric(base["driver_id"], errors="coerce").astype("Int64")

        # Compute best quali time and gap to pole if Q1/Q2/Q3 are available
        if any(c in base.columns for c in ["Q1", "Q2", "Q3"]):
            base["quali_best_ms"] = base.apply(_best_quali_ms, axis=1)
            pole_ms = base["quali_best_ms"].min()
            base["quali_gap_to_pole_ms"] = base["quali_best_ms"] - pole_ms
        else:
            base["quali_best_ms"] = np.nan
            base["quali_gap_to_pole_ms"] = np.nan

        # Use quali_rank as gridposition by default; this is a reasonable proxy
        base["gridposition"] = base["quali_rank"]

        quali_df = base
        print(f"✅ Loaded Qualifying for {year} {event_name}: {len(quali_df)} drivers")

    except Exception as e:
        print(f"⚠️ Could not load Qualifying for {year} {event_name}: {e}")

    # --- If Qualifying failed, try Race classification to at least get a roster ---
    roster_df = None
    if quali_df is None:
        try:
            r = fastf1.get_session(year, event_name, "R")
            r.load(laps=False, telemetry=False)
            res = r.results

            base = res[["DriverNumber", "TeamName"]].copy()
            base = base.rename(
                columns={
                    "DriverNumber": "driver_id",
                    "TeamName": "team_id",
                }
            )
            base["driver_id"] = pd.to_numeric(base["driver_id"], errors="coerce").astype("Int64")

            # No quali here; leave quali_* and gridposition as NaN
            base["quali_rank"] = np.nan
            base["quali_best_ms"] = np.nan
            base["quali_gap_to_pole_ms"] = np.nan
            base["gridposition"] = np.nan

            roster_df = base
            print(f"✅ Loaded Race classification for {year} {event_name}: {len(roster_df)} drivers")
        except Exception as e:
            print(f"❌ Could not load Race for {year} {event_name}: {e}")

    # Decide which DataFrame to use as the base live input
    if quali_df is not None:
        live = quali_df
    elif roster_df is not None:
        live = roster_df
    else:
        raise SystemExit(f"No usable data for {year} {event_name}")

    # Attach race/event metadata
    live["raceid"] = str(race_id)
    live["eventname"] = str(event_name)

    # Reorder columns for consistency
    cols_order = [
        "raceid",
        "eventname",
        "driver_id",
        "team_id",
        "gridposition",
        "quali_rank",
        "quali_best_ms",
        "quali_gap_to_pole_ms",
    ]
    live = live[[c for c in cols_order if c in live.columns] + [c for c in live.columns if c not in cols_order]]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    live.to_csv(out_csv, index=False)
    print(f"🗂  wrote {out_csv} with {len(live)} drivers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build live-like CSV from FastF1.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--event-name", type=str, required=True)
    parser.add_argument("--race-id", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    args = parser.parse_args()

    build_live_from_fastf1(
        year=args.year,
        event_name=args.event_name,
        race_id=args.race_id,
        out_csv=Path(args.out_csv),
    )