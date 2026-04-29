"""LangGraph ReAct agent: accepts a natural-language F1 query and returns a
plain-English race preview powered by the LightGBM prediction model."""
from __future__ import annotations

import sys
from datetime import date

from dotenv import load_dotenv
from langchain_groq import ChatGroq

import warnings

load_dotenv()
warnings.filterwarnings("ignore", message="create_react_agent has been moved")
from langgraph.prebuilt import create_react_agent

from agent.tools import get_race_prediction, list_available_races
from agent.hf_commentary import generate_commentary

_SYSTEM_PROMPT_TEMPLATE = """You are an expert F1 race analyst with access to a machine-learning \
prediction system. Today's date is {today}.

You have three tools:

1. list_available_races — find race IDs in the prediction database (filter by year if helpful)
2. get_race_prediction — run the LightGBM model for a given race and get win probabilities
3. generate_commentary — convert structured prediction data to a natural-language race preview using a local Hugging Face model (flan-t5-base). Call this after get_race_prediction to produce polished commentary.

Standard workflow for any race query:
- Call list_available_races (with the current year) to identify available races
- When asked about the "next" race, pick the race that has not yet occurred based on today's date
- Predictions require qualifying data. If get_race_prediction says no qualifying data exists, tell the user clearly — do NOT fall back to a different race
- Call get_race_prediction with that race_id to retrieve win probabilities
- Write a concise, engaging plain-English race preview covering:
    • The predicted winner and their probability
    • Top 3 contenders and what the numbers say
    • Any notable form patterns in the probabilities
    • A punchy one-sentence verdict

Keep responses focused — sharp analysis, not padded prose."""


def build_agent():
    """Construct the LangGraph ReAct agent."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    tools = [list_available_races, get_race_prediction, generate_commentary]
    prompt = _SYSTEM_PROMPT_TEMPLATE.format(today=date.today().isoformat())
    return create_react_agent(llm, tools, prompt=prompt)


def run_agent(query: str) -> str:
    """Run the F1 analyst agent on a natural-language query and return the response."""
    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Who are the favorites for the next F1 race?"
    print(run_agent(question))
