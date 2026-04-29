

# F1 Live Winner Predictor

An intelligent system that predicts **Formula 1 race win probabilities** in real-time based on live grid data and historical performance.  
Built with **FastAPI**, **FastF1**, **LangGraph**, and **pandas**, it provides a **CLI tool**, **web interface**, and an **AI agent** to simulate and explain results for any race from 2022 onward — including 2026.

---

## Features

- Predicts each driver’s win probability using:
  - Current grid position
  - Qualifying data
  - Historical performance of drivers & teams
- Build live CSVs for past or ongoing races using **FastF1**
- REST API and browser-based UI for easy visualization
- Automatically detects and lists races by year
- CLI support for power users and batch predictions
- **PyTorch MLP** — neural network alternative/ensemble alongside LightGBM
- **Hugging Face commentary** — local flan-t5-base model generates race previews from predictions
- **pgvector semantic search** — find historically similar races via PostgreSQL + pgvector

---

## Project Structure

```
f1-winner-predictor/
├── agent/                   # AI agent layer (LangGraph + Groq)
│   ├── agent.py             # ReAct agent entry point
│   ├── tools.py             # LangChain tools wrapping the ML pipeline
│   └── hf_commentary.py     # Hugging Face race commentary generator (flan-t5-base)
├── api/                     # FastAPI backend
│   ├── app.py               # Main API server (REST + agent endpoint)
│   └── static/              # Contains live.html frontend
├── data/
│   ├── live_inputs/         # Live-style race CSVs (2022–present)
│   ├── preds/               # Model prediction outputs
│   ├── silver/              # Cleaned historical data
│   └── ref/                 # Driver/team reference maps
├── data_pipeline/           # ETL scripts (fetch from FastF1, normalize)
├── features/                # Feature engineering
├── models/
│   ├── train_winner.py      # LightGBM training script
│   └── torch_mlp.py         # PyTorch MLP — alternative neural net model
├── vector_store/
│   └── race_store.py        # pgvector semantic race similarity search
├── registry/                # Trained model & lookup artifacts
├── scripts/                 # FastF1 data pipeline scripts
├── predict_live.py          # CLI entrypoint for predictions
├── requirements.txt         # Dependencies
└── README.md
```

---

## Installation & Setup

### Clone the project

```bash
git clone https://github.com/yourusername/f1-winner-predictor.git
cd f1-winner-predictor
```

### Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Building Live Race Data

### Easiest: sync after each race weekend

```bash
python scripts/sync.py
```

This fetches the season schedule from FastF1, skips races that haven't happened yet or already have a CSV, and downloads anything new. No arguments needed — just run it after each race weekend.

```bash
python scripts/sync.py --year 2025  # specific season
python scripts/sync.py --all        # backfill 2022–present
```

### Build one race manually

```bash
python scripts/build_live_from_fastf1.py \
  --year 2026 \
  --event-name "São Paulo Grand Prix" \
  --race-id 2026_brazil \
  --out-csv data/live_inputs/2026_brazil.csv
```

All CSVs will appear under `data/live_inputs/`.

---

## Predicting Winners (CLI)

Run predictions directly in the terminal:

```bash
python predict_live.py \
  --race-id 2026_brazil \
  --event-name "São Paulo Grand Prix" \
  --live-csv data/live_inputs/2026_brazil.csv \
  --out data/preds/2026_brazil_probs.csv \
  --top 12
```

Output:

```
2026_brazil — top 12:
raceid driver_id team_label prob_win eventname
2026_brazil 4 McLaren 46.2% São Paulo Grand Prix
...
```

---

## 🌐 Running the Web App

### Start the FastAPI backend

```bash
uvicorn api.app:app --reload --port 8000
```

### Open the frontend

Then go to:

```
http://127.0.0.1:8000/static/live.html
```

You’ll see:

- A dropdown for year & race  
- A table of driver probabilities  
- JSON data served from the FastAPI `/live_predict` endpoint

---

## Adding New Races

Whenever a new F1 race finishes:

```bash
python scripts/build_live_from_fastf1.py \
  --year 2026 \
  --event-name "New Grand Prix" \
  --race-id 2026_new \
  --out-csv data/live_inputs/2026_new.csv
```

Then refresh the app — it auto-detects new races in `data/live_inputs/`.

---

## AI Layer

The predictor includes a lightweight agentic AI layer built with **LangGraph** and the **Anthropic Claude API**.

### Architecture

```
User query (natural language)
        │
        ▼
  LangGraph ReAct Agent  ←── Claude claude-sonnet-4-6
        │
   ┌────┴────┐
   ▼         ▼
list_       get_race_
available_  prediction
races       │
            ▼
     LightGBM ensemble
     (registry/*.joblib)
            │
            ▼
   Plain-English preview
```

**Components:**

| File | Purpose |
|------|---------|
| `agent/tools.py` | Two LangChain tools wrapping the existing ML pipeline |
| `agent/agent.py` | LangGraph ReAct agent with `run_agent()` and CLI entry point |
| `api/app.py` `POST /agent/predict` | HTTP endpoint for the agent |

**Tools:**
- `list_available_races(year?)` — scans `data/live_inputs/` and returns all race IDs and names, optionally filtered by year
- `get_race_prediction(race_id, event_name)` — runs the LightGBM ensemble and returns ranked win probabilities

**Agent flow:**

Given *"Who will win the 2025 Bahrain Grand Prix?"*, the agent:
1. Calls `list_available_races(year=2025)` to locate `race_id = 2025_bahrain`
2. Calls `get_race_prediction("2025_bahrain", "Bahrain Grand Prix")`
3. Synthesizes a plain-English preview with winner odds, top contenders, and a verdict

### Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

### Usage

**CLI:**

```bash
python -m agent.agent "Who are the favorites for the 2026 Monaco Grand Prix?"
```

**API:**

Start the server, then:

```bash
curl -X POST http://127.0.0.1:8000/agent/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "Give me a race preview for the 2026 Saudi Arabian GP"}'
```

Response:

```json
{
  "answer": "Based on the model, Max Verstappen leads with a 32% win probability..."
}
```

### Tech stack

- [LangGraph](https://langchain-ai.github.io/langgraph/) — ReAct agent orchestration
- [Anthropic Claude](https://docs.anthropic.com/) (`claude-sonnet-4-6`) — LLM reasoning
- [LangChain Anthropic](https://python.langchain.com/docs/integrations/chat/anthropic/) — SDK integration

---

---

## PyTorch MLP

An MLP (`WinnerMLP`) trained on the same 18 features as LightGBM — usable standalone or blended into an ensemble.

**Train:**
```bash
python -m models.torch_mlp
# Saves: registry/winner_mlp.pt + registry/winner_mlp_scaler.joblib
```

**Inference:**
```python
from models.torch_mlp import load_model, predict_mlp
import joblib, numpy as np

model = load_model()
scaler = joblib.load("registry/winner_mlp_scaler.joblib")
X = scaler.transform(my_features_array)
probs = predict_mlp(model, X)  # shape (n_drivers,)
```

Architecture: `Linear(18→64) → BN → ReLU → Dropout → Linear(64→32) → BN → ReLU → Dropout → Linear(32→16) → ReLU → Linear(16→1)`  
Loss: `BCEWithLogitsLoss` with `pos_weight` to handle class imbalance (~1 winner per 20 drivers).

---

## Hugging Face Commentary

Generates plain-English race previews from structured prediction data using a **local** `google/flan-t5-base` model (no API key required, ~250 MB).

**Standalone:**
```bash
python -m agent.hf_commentary
# Downloads model on first run, prints a sample Monaco preview
```

**In Python:**
```python
from agent.hf_commentary import generate_race_commentary

commentary = generate_race_commentary({
    "event_name": "British Grand Prix",
    "drivers": [
        {"driver_name": "Lando Norris", "team_label": "McLaren", "prob_win": 0.31},
        {"driver_name": "Max Verstappen", "team_label": "Red Bull", "prob_win": 0.28},
    ]
})
print(commentary)
```

The `generate_commentary` LangChain tool is registered in the agent — after `get_race_prediction`, the agent can automatically call it to produce polished commentary.

---

## pgvector Semantic Race Search

Stores race performance summaries as embeddings in **PostgreSQL + pgvector**, enabling semantic similarity search.

**Prerequisites:**
```bash
# Start pgvector (Docker):
docker run -d --name pgvector \
    -e POSTGRES_PASSWORD=f1pass \
    -p 5432:5432 pgvector/pgvector:pg16

export PGVECTOR_CONNECTION_STRING="postgresql+psycopg2://postgres:f1pass@localhost:5432/f1"
```

**Demo (inserts 5 sample races + runs a query):**
```bash
python -m vector_store.race_store
python -m vector_store.race_store --query "dominant pole-to-win on a street circuit"
```

**In Python:**
```python
from vector_store.race_store import RaceVectorStore

store = RaceVectorStore()

store.add_race(
    race_id="2024_monaco",
    summary="Leclerc won from pole on home circuit. Ferrari 1-2. Dry race, minimal strategy.",
    metadata={"year": 2024, "circuit": "Monaco", "winner": "Leclerc"},
)

results = store.find_similar("Ferrari dominant street circuit win", k=3)
for r in results:
    print(r["race_id"], r["similarity_score"])
```

Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, ~80 MB).

---

##  Troubleshooting

**Problem:** `{"detail": "Not Found"}` when opening `/static/live.html`  
**Fix:** Ensure the file exists in `api/static/live.html` and that `app.py` includes:
```python
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="api/static"), name="static")
```

**Problem:** `Out of range float values are not JSON compliant`  
**Fix:** The API now replaces NaN/Inf with `None`. Reinstall updated dependencies.

**Problem:** Teams missing in output  
**Fix:** Some new drivers (2025+) aren’t in mappings. The script now uses raw team names as fallback.

---

## License & Credits

This project uses public Formula 1 timing data via the **FastF1** API.  
All F1 names and data belong to their respective owners.
