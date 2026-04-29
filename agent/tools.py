"""LangChain tools that expose the F1 prediction pipeline to the AI agent."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import fastf1
import pandas as pd
from langchain_core.tools import tool

# Reuse feature-building and probability logic already defined in the API module
from api.app import _build_live_features, _compute_probs
from predict_live import _get

LIVE_INPUTS = Path("data/live_inputs")


def _get_race_dates(year: int) -> dict[str, date]:
    """Return mapping of event name -> race date for a given year from FastF1."""
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        return {
            row["EventName"]: row["Session5Date"].date()
            for _, row in schedule.iterrows()
            if pd.notna(row.get("Session5Date"))
        }
    except Exception:
        return {}


@tool
def list_available_races(year: Optional[int] = None) -> str:
    """List F1 races available in the prediction system, optionally filtered by year.

    Args:
        year: 4-digit year to filter by (e.g. 2026). Omit to list all races.

    Returns a newline-separated list of entries formatted as:
        race_id | event_name | race_date | status
    Status is either "predictions available" or "no qualifying data yet".
    Only call get_race_prediction for races marked "predictions available".
    Race dates are included so you can determine which races are upcoming vs past.
    """
    if not LIVE_INPUTS.exists():
        return "No races found — data/live_inputs/ directory is missing."

    # Pre-fetch race dates per year to avoid redundant API calls
    date_cache: dict[int, dict[str, date]] = {}

    rows = []
    for csv_path in sorted(LIVE_INPUTS.glob("*.csv")):
        stem = csv_path.stem
        if year and not stem.startswith(str(year)):
            continue
        has_data = True
        try:
            df = pd.read_csv(csv_path, nrows=5)
            if df.empty or len(df.columns) <= 2 or len(df) == 0:
                has_data = False
            elif "gridposition" in df.columns and df["gridposition"].isna().all():
                has_data = False
            race_id_s = _get(df, "raceid", "race_id")
            event_s = _get(df, "eventname", "event_name")
            race_id = str(race_id_s.iloc[0]) if race_id_s is not None and len(race_id_s) > 0 else stem
            event_name = (
                str(event_s.iloc[0])
                if event_s is not None and len(event_s) > 0
                else stem.replace("_", " ").title()
            )
        except Exception:
            race_id = stem
            event_name = stem.replace("_", " ").title()
            has_data = False

        # Look up race date from FastF1 schedule
        try:
            yr = int(race_id.split("_")[0])
        except (ValueError, IndexError):
            yr = date.today().year
        if yr not in date_cache:
            date_cache[yr] = _get_race_dates(yr)
        race_date = date_cache[yr].get(event_name)
        date_str = race_date.isoformat() if race_date else "unknown"

        status = "predictions available" if has_data else "no qualifying data yet"
        rows.append(f"{race_id} | {event_name} | {date_str} | {status}")

    if not rows:
        return f"No races found{f' for year {year}' if year else ''}."
    return "\n".join(rows)


@tool
def get_race_prediction(race_id: str, event_name: str, top: int = 10) -> str:
    """Get ML-based win probability predictions for a specific F1 race.

    Args:
        race_id: The race identifier from list_available_races (e.g. '2025_bahrain').
        event_name: The full event name (e.g. 'Bahrain Grand Prix').
        top: Number of top drivers to return (default 10).

    Returns a ranked list of drivers with their predicted win probabilities,
    derived from grid position, qualifying performance, and rolling form stats.
    """
    csv_path = LIVE_INPUTS / f"{race_id}.csv"
    if not csv_path.exists():
        return (
            f"No data file found for race_id='{race_id}'. "
            "Use list_available_races to find valid race IDs."
        )

    # Check for empty/template CSV (no qualifying data yet)
    try:
        df_check = pd.read_csv(csv_path)
        if df_check.empty or len(df_check.columns) <= 2:
            return (
                f"No qualifying data available for '{event_name}' ({race_id}) yet. "
                "Predictions require grid positions from qualifying. "
                "This race has not had qualifying yet."
            )
        if "gridposition" in df_check.columns and df_check["gridposition"].isna().all():
            return (
                f"Qualifying data missing for '{event_name}' ({race_id}). "
                "Driver roster exists but grid positions are empty — qualifying has not run yet. "
                "Predictions without qualifying data would be unreliable."
            )
    except Exception:
        pass

    try:
        feat = _build_live_features(race_id, event_name, csv_path)
        probs = _compute_probs(feat, top)
    except Exception as exc:
        return f"Prediction failed: {exc}"

    lines = [f"Win probability predictions — {event_name} ({race_id}):", ""]
    for rank, (_, row) in enumerate(probs.iterrows(), start=1):
        driver = row.get("driver_name") or str(row["driver_id"])
        team = row.get("team_label") or str(row.get("team_id", ""))
        pct = f"{float(row['prob_win']) * 100:.1f}%"
        lines.append(f"  {rank}. {driver} ({team}): {pct}")

    return "\n".join(lines)
