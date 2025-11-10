# ingest/normalize.py
import pandas as pd
from pathlib import Path

RAW = "data/raw/f1_raw.csv"
SILVER = Path("data/silver"); SILVER.mkdir(parents=True, exist_ok=True)

# Load (auto-detect delimiter)
df = pd.read_csv(RAW, engine="python", sep=None)
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

# Normalize headers
cols = {c.lower(): c for c in df.columns}
def get(*cands, default=None):
    for c in cands:
        if c.lower() in cols: 
            return cols[c.lower()]
    return default  # not found

# Required/optional source columns (with aliases)
c_raceid        = get("raceid")
c_driver        = get("drivernumber","driver_number","number","driverid")
c_team          = get("teamname","constructor","constructorname")
c_grid          = get("gridposition","grid","grid_pos")
c_quali         = get("qualiposition","quali_position","qualipos","quali")  # may be None
c_position      = get("position","classifiedposition","finish_position","pos")
c_points        = get("points","pts")
c_status        = get("status")
c_event         = get("eventname","name","grand_prix")
c_year          = get("year","season")
c_round         = get("roundnumber","round")

need = [c_raceid, c_driver, c_team, c_grid, c_position]
if any(n is None for n in need):
    raise SystemExit(f"Missing required columns among: raceid, driver, team, grid, position. "
                     f"Found mapping: {need}")

out = pd.DataFrame({
    "raceid":        df[c_raceid],
    "driver_id":     df[c_driver],
    "team_id":       df[c_team],
    "gridposition":  pd.to_numeric(df[c_grid], errors="coerce"),
    "position":      pd.to_numeric(df[c_position], errors="coerce"),
    "points":        pd.to_numeric(df[c_points], errors="coerce") if c_points else 0.0,
    "status":        df[c_status] if c_status else "Finished",
    "eventname":     df[c_event] if c_event else None,
    "year":          pd.to_numeric(df[c_year], errors="coerce") if c_year else None,
    "roundnumber":   pd.to_numeric(df[c_round], errors="coerce") if c_round else None,
})

# De-duplicate: one row per (raceid, driver_id)
# Prefer Classified Position if duplicates exist
out = out.sort_values(
    by=["raceid", "driver_id", "position"],
    na_position="last"
).drop_duplicates(subset=["raceid", "driver_id"], keep="first")

# Optional quali position
out["qualiposition"] = pd.to_numeric(df[c_quali], errors="coerce") if c_quali else pd.NA

# Derivations
out["winner"] = (out["position"] == 1).astype(int)
out["dnf"]    = (~out["status"].fillna("Finished").str.contains("Finished", case=False)).astype(int)

# Persist tables we actually use
drivers = out[["driver_id","team_id"]].drop_duplicates()
races   = out[["raceid","eventname","year","roundnumber"]].drop_duplicates()

drivers.to_parquet(SILVER/"drivers.parquet", index=False)
races.to_parquet(SILVER/"races.parquet", index=False)
out.to_parquet(SILVER/"race_results.parquet", index=False)

print("✅ Wrote silver tables: drivers.parquet, races.parquet, race_results.parquet")