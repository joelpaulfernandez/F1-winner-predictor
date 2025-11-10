import argparse, numpy as np, pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True, help="CSV from predict_live.py (pretty)")
ap.add_argument("--race-id", required=True)
ap.add_argument("--trials", type=int, default=100_000)
args = ap.parse_args()

df = pd.read_csv(args.preds)
sub = df[df["raceid"] == args.race_id].copy()
if sub.empty:
    raise SystemExit(f"No rows for race {args.race_id} in {args.preds}")

# Use prob_win; ensure normalized
p = sub["prob_win"].to_numpy().astype(float)
p = p / p.sum()
labels = (sub.get("driver_name") if "driver_name" in sub else sub["driver_id"].astype(str)).tolist()

# Multinomial draws: winner each trial
rng = np.random.default_rng(seed=42)
winners = rng.choice(len(labels), size=args.trials, p=p)
win_counts = np.bincount(winners, minlength=len(labels))

# Approx podium by weighted sampling without replacement heuristics:
# repeat: draw 1st w/ p, then 2nd with renormalized p (downweight winner), etc.
# (Not exact, but reasonable for a quick view.)
podium_counts = np.zeros((len(labels), 3), dtype=np.int64)
for _ in range(args.trials):
    choices = rng.choice(len(labels), size=min(3, len(labels)), replace=False, p=p)
    for rank, idx in enumerate(choices):
        podium_counts[idx, rank] += 1

out = pd.DataFrame({
    "driver": labels,
    "P(win)": win_counts / args.trials,
    "P(podium)": podium_counts.sum(axis=1) / args.trials,
    "P(P2)": podium_counts[:,1] / args.trials,
    "P(P3)": podium_counts[:,2] / args.trials,
}).sort_values("P(win)", ascending=False)

Path("reports").mkdir(exist_ok=True)
csv_out = f"reports/{args.race_id}_montecarlo.csv"
out.to_csv(csv_out, index=False)
print(f"🗂  {csv_out}")

print("\nTop 10 by P(win):")
print(out.head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))