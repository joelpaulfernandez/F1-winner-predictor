# models/train_winner.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss
from joblib import dump
from pathlib import Path

# ---- Load features & target
fv = pd.read_parquet("data/gold/fv_preRace.parquet")
# minimal features from your schema
FEATURES = [
    # grid / quali
    "gridposition", "qualiposition",
    "quali_best_ms", "quali_gap_to_pole_ms", "quali_rank",
    # rolling form
    "driver_avg_finish_last5","driver_avg_points_last5",
    "team_avg_points_last5","team_avg_finish_last5",
    # track history
    "driver_event_avg_finish_hist3","driver_event_avg_points_hist3",
    "team_event_avg_finish_hist3","team_event_avg_points_hist3",
    # weather aggregates (historical realized; swap to forecast for live use)
    "airtempmean","tracktempmean","humidity_mean","rainprobproxy","windspeed_mean",
]
TARGET = "winner"
GROUPS = "raceid"     # ensure no race leakage

# clean
df = fv.copy()
for c in FEATURES:
    if c not in df.columns:
        df[c] = np.nan
X = df[FEATURES].astype(float).fillna(df[FEATURES].median())
y = df[TARGET].astype(int).values
groups = df[GROUPS].values

# ---- CV training (grouped by race)
gkf = GroupKFold(n_splits=5)
models, oof_pred, oof_idx = [], np.zeros(len(df)), np.zeros(len(df), dtype=bool)

for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
    dtr = lgb.Dataset(X.iloc[tr], label=y[tr])
    dva = lgb.Dataset(X.iloc[va], label=y[va])

    params = dict(
        objective="binary",
        metric="binary_logloss",
        learning_rate=0.03,
        num_leaves=63,
        min_data_in_leaf=50,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        verbose=-1,
    )

    model = lgb.train(
        params,
        dtr,
        num_boost_round=5000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
    )
    p = model.predict(X.iloc[va], num_iteration=model.best_iteration)
    oof_pred[va] = p
    oof_idx[va] = True
    models.append(model)
    print(f"fold {fold}: val logloss={log_loss(y[va], p):.4f}")

# ---- Calibrate on OOF preds
iso = IsotonicRegression(out_of_bounds="clip").fit(oof_pred[oof_idx], y[oof_idx])
cal_oof = iso.predict(oof_pred[oof_idx])

print(f"OOF logloss (raw): {log_loss(y[oof_idx], oof_pred[oof_idx]):.4f}")
print(f"OOF logloss (cal): {log_loss(y[oof_idx], cal_oof):.4f}")
print(f"OOF Brier (cal):   {brier_score_loss(y[oof_idx], cal_oof):.4f}")

# ---- Save artifacts
Path("registry").mkdir(exist_ok=True)
dump(models, "registry/winner_lgb_folds.joblib")
dump(iso, "registry/winner_isotonic.joblib")
with open("registry/winner_features.txt", "w") as f:
    f.write("\n".join(FEATURES))
print("✅ Saved model + calibrator to registry/")

# ---- Create per-race normalized predictions (useful for UI/API)
def ensemble_predict(Xdf):
    raw = np.mean([m.predict(Xdf, num_iteration=m.best_iteration) for m in models], axis=0)
    return iso.predict(raw)

pred_df = df[[GROUPS, "driver_id"]].copy()
pred_df["prob_win_raw"] = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)
pred_df["prob_win"] = iso.predict(pred_df["prob_win_raw"])
# normalize so each race sums to 1
pred_df["prob_win"] = pred_df.groupby(GROUPS)["prob_win"].transform(lambda s: s / s.sum())

out = pred_df.sort_values([GROUPS, "prob_win"], ascending=[True, False])
Path("data/preds").mkdir(parents=True, exist_ok=True)
out.to_csv("data/preds/winner_probs_by_race.csv", index=False)
print("📄 Wrote per-race predictions → data/preds/winner_probs_by_race.csv")

# quick preview
for rid, g in out.groupby(GROUPS):
    print(f"\n{rid} top 5:")
    print(g.head(5)[["driver_id", "prob_win"]].to_string(index=False))
    break  # show first race only


fi = np.mean([m.feature_importance(importance_type="gain") for m in models], axis=0)
pd.Series(fi, index=FEATURES).sort_values(ascending=False).to_csv("registry/feature_importance.csv")
print("📈 feature importance → registry/feature_importance.csv")
# ---------------------------------------------------------------------
# 6. Append readable race / driver / team labels to predictions
# ---------------------------------------------------------------------
fv = pd.read_parquet("data/gold/fv_preRace.parquet")[
    ["raceid", "driver_id", "team_id", "eventname"]
].drop_duplicates()

pretty = out.merge(fv, on=["raceid", "driver_id"], how="left") \
             .sort_values(["raceid", "prob_win"], ascending=[True, False])

pretty.to_csv("data/preds/winner_probs_pretty.csv", index=False)
print("🗂  Wrote readable predictions → data/preds/winner_probs_pretty.csv")