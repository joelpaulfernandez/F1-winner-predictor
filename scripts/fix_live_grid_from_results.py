import pandas as pd
from pathlib import Path

SILVER_RR_PATH = Path("data/silver/race_results.parquet")
LIVE_DIR = Path("data/live_inputs")

def main():
    if not SILVER_RR_PATH.exists():
        raise SystemExit(f"Missing {SILVER_RR_PATH}, cannot backfill gridposition.")

    rr = pd.read_parquet(SILVER_RR_PATH)

    # Normalize keys in race_results
    if "raceid" not in rr.columns:
        # If your parquet uses different naming, adjust this block:
        # e.g. raceId / race_id
        raise SystemExit("race_results.parquet does not have a 'raceid' column.")
    rr["raceid"] = rr["raceid"].astype(str)
    rr["driver_id"] = pd.to_numeric(rr["driver_id"], errors="coerce").astype("Int64")

    if "gridposition" not in rr.columns:
        raise SystemExit("race_results.parquet does not have a 'gridposition' column.")

    rr_grid = rr[["raceid", "driver_id", "gridposition"]].drop_duplicates()

    if not LIVE_DIR.exists():
        raise SystemExit(f"No live inputs directory at {LIVE_DIR}")

    for csv_path in sorted(LIVE_DIR.glob("*.csv")):
        print(f"➡️  Fixing {csv_path} ...")
        df = pd.read_csv(csv_path)

        if "raceid" not in df.columns or "driver_id" not in df.columns:
            print(f"   ⚠️  Skipping {csv_path} (missing raceid/driver_id)")
            continue

        df["raceid"] = df["raceid"].astype(str)
        df["driver_id"] = pd.to_numeric(df["driver_id"], errors="coerce").astype("Int64")

        # Merge in grid from race_results
        merged = df.merge(
            rr_grid,
            on=["raceid", "driver_id"],
            how="left",
            suffixes=("", "_from_rr"),
        )

        # Ensure we have a gridposition column
        if "gridposition" not in merged.columns:
            merged["gridposition"] = pd.NA

        if "gridposition_from_rr" in merged.columns:
            # Fill missing gridposition from race_results
            mask = merged["gridposition"].isna()
            merged.loc[mask, "gridposition"] = merged.loc[mask, "gridposition_from_rr"]
            merged = merged.drop(columns=["gridposition_from_rr"])

        # Write back
        merged.to_csv(csv_path, index=False)
        print(f"   ✅ saved with gridposition backfilled")

    print("Done backfilling gridposition for live inputs.")

if __name__ == "__main__":
    main()