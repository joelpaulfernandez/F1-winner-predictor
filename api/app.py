from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Reuse all the logic from predict_live.py so the API matches the CLI
from predict_live import (
    _get,
    _parse_time_to_ms,
    add_rollups_from_history,
    fallback_prior,
    make_name_lookup,
    SILVER_RR_PATH,
)

app = FastAPI(
    title="F1 Live Winner Predictor",
    description="Predict win probabilities from a live F1 grid CSV using the same logic as predict_live.py.",
)

# Serve static files (e.g., live.html) from api/static
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Allow local frontends; you can lock this down later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_live_features(race_id: str, event_name: str, csv_path: Path) -> pd.DataFrame:
    """
    Replicate the preprocessing in predict_live.main, but as a function.

    This reads the live CSV, coalesces grid/quali columns, adds race/event metadata,
    and enriches with history using add_rollups_from_history.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    live = pd.read_csv(csv_path)

    # --- coalesce grid/quali into canonical names (same as predict_live.main) ---
    cand_grid = _get(
        live,
        "gridposition",
        "grid",
        "grid_pos",
        "starting_grid",
        "start_grid",
        "start_position",
    )
    if cand_grid is not None:
        live["gridposition"] = pd.to_numeric(cand_grid, errors="coerce")

    cand_qrank = _get(
        live,
        "quali_rank",
        "qualirank",
        "qualiposition",
        "qual_pos",
        "qualifying_position",
    )
    if cand_qrank is not None:
        live["quali_rank"] = pd.to_numeric(cand_qrank, errors="coerce")

    cand_qbest = _get(
        live,
        "quali_best_ms",
        "quali_best_time",
        "best_qual_time",
        "qualifying_time",
        "qtime",
    )
    if cand_qbest is not None:
        live["quali_best_ms"] = pd.to_numeric(cand_qbest, errors="coerce")
        if live["quali_best_ms"].isna().any():
            live.loc[live["quali_best_ms"].isna(), "quali_best_ms"] = (
                live.loc[live["quali_best_ms"].isna(), cand_qbest.name]
                .map(_parse_time_to_ms)
            )

    cand_qgap = _get(
        live,
        "quali_gap_to_pole_ms",
        "q_gap_ms",
        "quali_gap",
        "gap_to_pole",
    )
    if cand_qgap is not None:
        live["quali_gap_to_pole_ms"] = pd.to_numeric(cand_qgap, errors="coerce")

    live["raceid"] = race_id
    live["eventname"] = event_name

    # Enrich with history
    feat = add_rollups_from_history(live)

    # Ensure team_id present for output; try to backfill from history if missing
    if "team_id" not in feat.columns or feat["team_id"].isna().all():
        try:
            if SILVER_RR_PATH.exists():
                rr = pd.read_parquet(SILVER_RR_PATH)
                team_map = (
                    rr.sort_values(["year", "roundnumber"], kind="mergesort")
                    .groupby("driver_id", as_index=False)
                    .tail(1)[["driver_id", "team_id"]]
                    .drop_duplicates()
                )
                feat = feat.merge(
                    team_map,
                    on="driver_id",
                    how="left",
                    suffixes=("", "_from_hist"),
                )
                if "team_id_from_hist" in feat.columns:
                    feat["team_id"] = feat["team_id"].fillna(feat["team_id_from_hist"])
                    feat = feat.drop(
                        columns=[c for c in ["team_id_from_hist"] if c in feat.columns]
                    )
        except Exception:
            # If history lookup fails, just proceed with what we have
            pass
        if "team_id" not in feat.columns:
            feat["team_id"] = pd.NA

    return feat


def _compute_probs(feat: pd.DataFrame, top: int) -> pd.DataFrame:
    """
    Use fallback_prior to compute probabilities and attach names/team labels.

    Returns a DataFrame sorted by prob_win desc, limited to top rows.
    """
    prob_raw = fallback_prior(feat)

    out = feat[["raceid", "driver_id", "team_id", "eventname"]].copy()
    out["prob_win"] = prob_raw
    out["prob_win"] = out["prob_win"] / out.groupby("raceid")["prob_win"].transform(
        "sum"
    ).replace(0, 1)

    # Attach names/team label (same as CLI)
    if SILVER_RR_PATH.exists():
        rr = pd.read_parquet(SILVER_RR_PATH)
        names = make_name_lookup(rr)
        out = out.merge(names, on="driver_id", how="left")
    else:
        out["driver_name"] = out["driver_id"].astype(str)
        out["team_label"] = out["team_id"].astype(str)

    save_cols = [
        "raceid",
        "driver_id",
        "driver_name",
        "team_id",
        "team_label",
        "prob_win",
        "eventname",
    ]
    save_cols = [c for c in save_cols if c in out.columns]

    out_sorted = out.sort_values(["raceid", "prob_win"], ascending=[True, False])
    return out_sorted[save_cols].head(top)


# New endpoint: /live_races
@app.get("/live_races")
async def live_races():
    """List available live-input races based on CSVs in data/live_inputs.

    Returns a list of objects: {race_id, eventname, csv_path}.
    """
    base = Path("data/live_inputs")
    races = []

    if base.exists():
        for csv_path in sorted(base.glob("*.csv")):
            try:
                df = pd.read_csv(csv_path, nrows=1)
            except Exception:
                continue

            # Try to read race_id from the file
            raceid_series = _get(df, "raceid", "race_id")
            race_id = None
            if raceid_series is not None and not raceid_series.empty:
                race_id = str(raceid_series.iloc[0])

            # Try to read event name
            event_series = _get(df, "eventname", "event_name")
            eventname = None
            if event_series is not None and not event_series.empty:
                eventname = str(event_series.iloc[0])

            # Fallback label based on filename
            if not eventname:
                eventname = csv_path.stem.replace("_", " ").title()

            races.append(
                {
                    "race_id": race_id,
                    "eventname": eventname,
                    "csv_path": str(csv_path),
                }
            )

    return {"races": races}


@app.get("/live_predict")
async def live_predict(
    race_id: str,
    event_name: str,
    csv_path: str,
    top: int = 12,
):
    """
    Predict win probabilities from a live grid CSV.

    Query parameters:
    - race_id: e.g. "2024_1"
    - event_name: e.g. "Bahrain Grand Prix"
    - csv_path: path to a CSV like data/live_inputs/bahrain_2024.csv
    - top: how many drivers to return (default 12)
    """
    try:
        feat = _build_live_features(race_id, event_name, Path(csv_path))
        top_df = _compute_probs(feat, top)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Sanitize DataFrame so JSON encoding does not choke on NaN/inf
    top_df = top_df.replace([np.inf, -np.inf], np.nan)
    top_df = top_df.where(pd.notna(top_df), None)

    # Convert to plain JSON-serializable records; keep prob_win as float 0–1
    records = []
    for _, row in top_df.iterrows():
        rec = row.to_dict()
        if "prob_win" in rec:
            try:
                rec["prob_win"] = float(rec["prob_win"])
            except Exception:
                pass
        records.append(rec)

    return {
        "race_id": race_id,
        "eventname": event_name,
        "top": top,
        "drivers": records,
    }