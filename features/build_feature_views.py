# features/build_feature_views.py
import pandas as pd
from pathlib import Path

# base tables
rr = pd.read_parquet("data/silver/race_results.parquet").copy()
rr = rr[~rr["eventname"].str.contains("Test", case=False, na=False)]  # drop pre-season tests

# types & order
for c in ["position","points","year","roundnumber","gridposition","qualiposition"]:
    if c in rr.columns:
        rr[c] = pd.to_numeric(rr[c], errors="coerce")
rr = rr.sort_values(["year","roundnumber"]).reset_index(drop=True)
rr["finish_position"] = rr["position"]

# --- attach Quali (left join) ---
try:
    q = pd.read_parquet("data/silver/quali_results.parquet")
    rr = rr.merge(q, on=["raceid","driver_id"], how="left")
except Exception:
    rr[["quali_best_ms","quali_rank","quali_gap_to_pole_ms"]] = pd.NA

# --- attach Weather (left join by race) ---
try:
    w = pd.read_parquet("data/silver/weather.parquet")
    rr = rr.merge(w, on="raceid", how="left")
except Exception:
    rr[["airtempmean","tracktempmean","humidity_mean","rainprobproxy","windspeed_mean"]] = pd.NA

# ---------- rolling form ----------
driver_form = (
    rr.groupby("driver_id", group_keys=False)
      .apply(lambda x: x.assign(
          driver_avg_finish_last5 = x["finish_position"].rolling(5, min_periods=1).mean().shift(1),
          driver_avg_points_last5 = x["points"].rolling(5, min_periods=1).mean().shift(1),
      ))
)
team_form = (
    driver_form.groupby("team_id", group_keys=False)
      .apply(lambda x: x.assign(
          team_avg_points_last5   = x["points"].rolling(5, min_periods=1).mean().shift(1),
          team_avg_finish_last5   = x["finish_position"].rolling(5, min_periods=1).mean().shift(1),
      ))
)

# ---------- same-event history (past up to 3) ----------
def add_event_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["year","roundnumber"])
    df["driver_event_avg_finish_hist3"] = (
        df.groupby(["driver_id","eventname"])["finish_position"]
          .apply(lambda s: s.rolling(3, min_periods=1).mean().shift(1)).values
    )
    df["driver_event_avg_points_hist3"] = (
        df.groupby(["driver_id","eventname"])["points"]
          .apply(lambda s: s.rolling(3, min_periods=1).mean().shift(1)).values
    )
    df["team_event_avg_finish_hist3"] = (
        df.groupby(["team_id","eventname"])["finish_position"]
          .apply(lambda s: s.rolling(3, min_periods=1).mean().shift(1)).values
    )
    df["team_event_avg_points_hist3"] = (
        df.groupby(["team_id","eventname"])["points"]
          .apply(lambda s: s.rolling(3, min_periods=1).mean().shift(1)).values
    )
    return df

feat = add_event_history(team_form)

# ---------- select + fill ----------
cols = [
    "raceid","eventname","driver_id","team_id",
    "gridposition","qualiposition",                      # from race table if present
    "quali_best_ms","quali_rank","quali_gap_to_pole_ms", # from Q
    "driver_avg_finish_last5","driver_avg_points_last5",
    "team_avg_points_last5","team_avg_finish_last5",
    "driver_event_avg_finish_hist3","driver_event_avg_points_hist3",
    "team_event_avg_finish_hist3","team_event_avg_points_hist3",
    "airtempmean","tracktempmean","humidity_mean","rainprobproxy","windspeed_mean",  # weather
    "finish_position","winner","dnf"
]
for c in cols:
    if c not in feat.columns:
        feat[c] = pd.NA

# defaults for early races without history
feat = feat.fillna({
    "driver_avg_finish_last5": 10.0, "driver_avg_points_last5": 2.0,
    "team_avg_points_last5": 2.0,    "team_avg_finish_last5": 10.0,
    "driver_event_avg_finish_hist3": 10.0, "driver_event_avg_points_hist3": 2.0,
    "team_event_avg_finish_hist3": 10.0,   "team_event_avg_points_hist3": 2.0
})

Path("data/gold").mkdir(parents=True, exist_ok=True)
feat[cols].to_parquet("data/gold/fv_preRace.parquet", index=False)
print(f"✅ wrote data/gold/fv_preRace.parquet [{len(feat)} rows]")