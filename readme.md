

# F1 Live Winner Predictor

An intelligent system that predicts **Formula 1 race win probabilities** in real-time based on live grid data and historical performance.  
Built with **FastAPI**, **FastF1**, and **pandas**, it provides both a **CLI tool** and a **web interface** to simulate results for any race from 2022 onward — including 2025.

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

---

## Project Structure

```
f1-winner-predictor/
├── api/                     # FastAPI backend
│   ├── app.py               # Main API server
│   └── static/              # Contains live.html frontend
├── data/
│   ├── live_inputs/         # Live-style race CSVs
│   ├── preds/               # Model prediction outputs
│   ├── silver/              # Cleaned historical data
│   └── ref/                 # Driver/team reference maps
├── registry/                # Trained model & lookup artifacts
├── scripts/                 # FastF1 data pipeline scripts
├── predict_live.py          # CLI entrypoint for predictions
├── train_winner.py          # Training script for models
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

To build CSVs for races using **FastF1**:

### Build one race manually

```bash
python scripts/build_live_from_fastf1.py \
  --year 2025 \
  --event-name "São Paulo Grand Prix" \
  --race-id 2025_brazil \
  --out-csv data/live_inputs/2025_brazil.csv
```

### Build all races for a season (e.g., 2025)

```bash
python scripts/build_2025_live_from_fastf1.py
```

All CSVs will appear under `data/live_inputs/`.

---

## Predicting Winners (CLI)

Run predictions directly in the terminal:

```bash
python predict_live.py \
  --race-id 2025_brazil \
  --event-name "São Paulo Grand Prix" \
  --live-csv data/live_inputs/2025_brazil.csv \
  --out data/preds/2025_brazil_probs.csv \
  --top 12
```

Output:

```
2025_brazil — top 12:
raceid driver_id team_label prob_win eventname
2025_brazil 4 McLaren 46.2% São Paulo Grand Prix
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
  --year 2025 \
  --event-name "New Grand Prix" \
  --race-id 2025_new \
  --out-csv data/live_inputs/2025_new.csv
```

Then refresh the app — it auto-detects new races in `data/live_inputs/`.

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
