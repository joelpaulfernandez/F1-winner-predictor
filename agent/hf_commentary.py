"""Hugging Face race commentary generator.

Uses a local flan-t5-base model (text2text-generation) to produce
natural-language race summaries from structured prediction data.
Integrates with the LangGraph agent as a LangChain tool.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

_pipeline = None  # lazy-loaded on first call


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=200,
        )
    return _pipeline


def _format_prompt(race_data: dict[str, Any]) -> str:
    """Build a flan-t5 instruction prompt from structured race result data."""
    event = race_data.get("event_name", "Unknown Grand Prix")
    drivers = race_data.get("drivers", [])

    driver_lines = []
    for d in drivers[:5]:
        name = d.get("driver_name") or d.get("name", "Unknown")
        team = d.get("team_label") or d.get("team", "")
        prob = d.get("prob_win", 0)
        pct = f"{float(prob) * 100:.1f}%" if prob else "N/A"
        driver_lines.append(f"{name} ({team}): {pct} win probability")

    standings = "\n".join(driver_lines) if driver_lines else "No data"

    return (
        f"Write a one-paragraph Formula 1 race preview for the {event}. "
        f"Use these predicted win probabilities:\n{standings}\n"
        "Focus on the top contender, key rivals, and end with a punchy verdict. "
        "Write in an engaging sports journalism style."
    )


def generate_race_commentary(race_data: dict[str, Any]) -> str:
    """Generate a natural-language race preview from structured prediction data.

    Args:
        race_data: dict with keys:
            - event_name (str): e.g. "Monaco Grand Prix"
            - drivers (list[dict]): each with driver_name, team_label, prob_win
    Returns:
        Plain-English race commentary string.
    """
    pipe = _get_pipeline()
    prompt = _format_prompt(race_data)
    result = pipe(prompt, do_sample=False)
    return result[0]["generated_text"].strip()


@tool
def generate_commentary(race_prediction_json: str) -> str:
    """Generate a natural-language race preview using a local Hugging Face model.

    Args:
        race_prediction_json: JSON string with keys:
            - event_name: Full race name e.g. "Monaco Grand Prix"
            - drivers: list of {driver_name, team_label, prob_win} dicts

    Returns a one-paragraph race preview in sports journalism style.
    Useful after calling get_race_prediction to produce a human-readable summary.
    """
    try:
        race_data = json.loads(race_prediction_json)
    except json.JSONDecodeError as exc:
        return f"Invalid JSON input: {exc}"

    return generate_race_commentary(race_data)


if __name__ == "__main__":
    sample = {
        "event_name": "Monaco Grand Prix",
        "drivers": [
            {"driver_name": "Max Verstappen", "team_label": "Red Bull", "prob_win": 0.38},
            {"driver_name": "Charles Leclerc", "team_label": "Ferrari", "prob_win": 0.22},
            {"driver_name": "Lando Norris", "team_label": "McLaren", "prob_win": 0.18},
            {"driver_name": "Lewis Hamilton", "team_label": "Mercedes", "prob_win": 0.11},
            {"driver_name": "Carlos Sainz", "team_label": "Ferrari", "prob_win": 0.07},
        ],
    }
    print("Generating commentary (downloading model on first run)...\n")
    print(generate_race_commentary(sample))
