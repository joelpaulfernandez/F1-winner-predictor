# data_pipeline/build_mappings.py
import pandas as pd
from pathlib import Path

rr = pd.read_parquet("data/silver/race_results.parquet")

# Try to infer readable labels from columns you may have
c_driver_name = None
for cand in ["fullname", "driver_name", "driver", "abbreviation"]:
    if cand in rr.columns:
        c_driver_name = cand
        break

c_team_name = None
for cand in ["teamname", "constructor", "constructor_name"]:
    if cand in rr.columns:
        c_team_name = cand
        break

drv_cols = ["driver_id"]
if c_driver_name: drv_cols.append(c_driver_name)
drivers = rr[drv_cols].drop_duplicates()

team_cols = ["team_id"]
if c_team_name: team_cols.append(c_team_name)
teams = rr[team_cols].drop_duplicates()

Path("data/ref").mkdir(parents=True, exist_ok=True)
drivers.to_parquet("data/ref/drivers.parquet", index=False)
teams.to_parquet("data/ref/teams.parquet", index=False)
print("✅ wrote data/ref/drivers.parquet and data/ref/teams.parquet")