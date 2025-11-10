import pandas as pd
import matplotlib.pyplot as plt
import sys

race_file = sys.argv[1] if len(sys.argv) > 1 else "data/preds/las_vegas_probs.csv"
df = pd.read_csv(race_file)

plt.figure(figsize=(6, 4))
plt.bar(df["team_id"] + " (" + df["driver_id"].astype(str) + ")", df["prob_win"], color="dodgerblue")
plt.title(df["eventname"].iloc[0])
plt.ylabel("Win Probability")
plt.xlabel("Driver (Team)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

out_path = race_file.replace(".csv", ".png")
plt.savefig(out_path)
plt.show()

print(f"Saved chart → {out_path}")