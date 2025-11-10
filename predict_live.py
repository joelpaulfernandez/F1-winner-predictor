
#!/usr/bin/env python3
# Robust live-race prediction with safe history rollups, model alignment, and name lookup.

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------- Paths (adjust if your layout differs) ----------
SILVER_RR_PATH = Path("data/silver/race_results.parquet")
REGISTRY_DIR   = Path("registry")

# ---------- Small utils ----------
def _softmax(z):
    z = np.asarray(z, dtype=float)
    # if all values are NaN or the array is empty, return uniform safely
    if z.size == 0 or np.isnan(z).all():
        return z + 1.0  # will be normalized by caller; keep size
    # replace remaining NaNs with the finite median (or 0.0 if still NaN)
    finite = z[np.isfinite(z)]
    fill = np.median(finite) if finite.size else 0.0
    z = np.where(np.isnan(z), fill, z)
    # standard softmax with numerical stability
    z = z - np.max(z)
    ez = np.exp(z)
    denom = ez.sum()
    if denom == 0 or not np.isfinite(denom):
        # fallback to uniform distribution
        return np.full_like(z, 1.0 / len(z))
    return ez / denom

def logistic(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))

def _get(df, *cand):
    """Return the first existing column Series from candidates (case-insensitive), else None."""
    if not hasattr(df, "columns"):
        return None
    # Build a mapping of lowercase column name -> actual column name
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for c in cand:
        # try exact match first
        if c in df.columns:
            return df[c]
        # then try case-insensitive / trimmed match
        key = str(c).strip().lower()
        if key in lower_map:
            return df[lower_map[key]]
    return None

def _parse_time_to_ms(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (float, int, np.number)):
        return float(s)
    s = str(s).strip()
    if not s:
        return np.nan
    try:
        if ":" in s:
            m, rest = s.split(":")
            sec = float(rest)
            return (float(m) * 60.0 + sec) * 1000.0
        return float(s)  # already secs/ms; we keep unit as-is
    except Exception:
        m = re.findall(r"[\d\.]+", s)
        return float(m[0]) if m else np.nan

# ---------- Model helpers ----------
def _unwrap_model(obj):
    """Return a list of model-like objects from possibly nested dicts/lists."""
    if isinstance(obj, (list, tuple)):
        out = []
        for x in obj:
            out.extend(_unwrap_model(x))
        return out
    if isinstance(obj, dict):
        for k in ("models", "estimators", "folds"):
            if k in obj:
                return _unwrap_model(obj[k])
        for k in ("model", "estimator", "clf", "best_estimator_", "sk_model", "booster"):
            if k in obj:
                return _unwrap_model(obj[k])
        return []
    return [obj]

def _maybe_wrap_xgb_booster(m):
    """Wrap raw xgboost.Booster so it exposes predict_proba/predict like sklearn."""
    try:
        import xgboost as xgb
    except Exception:
        xgb = None
    if xgb is not None and isinstance(m, xgb.Booster):
        class _BoosterWrapper:
            def __init__(self, booster):
                self.booster = booster
                self._n_features_in = None
            def predict_proba(self, X):
                dm = xgb.DMatrix(X)
                pred = self.booster.predict(dm)
                pred = np.asarray(pred).reshape(-1)
                proba_pos = pred
                proba_neg = 1.0 - proba_pos
                return np.vstack([proba_neg, proba_pos]).T
            def predict(self, X):
                return self.predict_proba(X)[:, 1]
            @property
            def n_features_in_(self):
                # best-effort (used by aligner when names are absent)
                return self._n_features_in
        return _BoosterWrapper(m)
    return m

def load_models():
    """Load one or more models and optional calibrator from registry/."""
    models = []
    if REGISTRY_DIR.exists():
        for p in sorted(REGISTRY_DIR.glob("model_*.joblib")):
            loaded = joblib.load(p)
            for m in _unwrap_model(loaded):
                models.append(_maybe_wrap_xgb_booster(m))
        single = REGISTRY_DIR / "model_xgb.joblib"
        if single.exists():
            loaded = joblib.load(single)
            for m in _unwrap_model(loaded):
                models.append(_maybe_wrap_xgb_booster(m))
        cal_path = REGISTRY_DIR / "calibrator.joblib"
        calibrator = joblib.load(cal_path) if cal_path.exists() else None
    else:
        models = []
        calibrator = None
    if not models:
        raise FileNotFoundError("No models found. Expected registry/model_*.joblib or registry/model_xgb.joblib")
    # unwrap calibrator dicts if needed
    if calibrator is not None:
        cands = _unwrap_model(calibrator) or [calibrator]
        calibrator = cands[0]
    return models, calibrator

def _expected_feature_names(model):
    """Try to discover the model's training feature names (order matters)."""
    # sklearn
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)
    # xgboost sklearn wrapper
    try:
        booster = model.get_booster()
        bn = booster.feature_names
        if bn:
            return list(bn)
    except Exception:
        pass
    return None  # count-only fallback

def _align_X_for_model(X: pd.DataFrame, model):
    """Align X to the model's expected columns (names or n_features_in_)."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    names = _expected_feature_names(model)
    if names:
        missing = [c for c in names if c not in X.columns]
        if missing:
            X = X.copy()
            for c in missing:
                X[c] = 0.0
        return X.reindex(columns=names, fill_value=0.0)
    n = getattr(model, "n_features_in_", None)
    if n is None:
        return X
    A = np.asarray(X, dtype=float)
    curr = A.shape[1]
    if curr < n:
        pad = np.zeros((A.shape[0], n - curr), dtype=float)
        A = np.concatenate([A, pad], axis=1)
    elif curr > n:
        A = A[:, :n]
    return A

def ensemble_predict(models, X):
    preds = []
    for m in models:
        Xm = _align_X_for_model(X, m)
        if hasattr(m, "predict_proba"):
            try:
                proba = np.asarray(m.predict_proba(Xm))
                p = proba[:, -1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.reshape(-1)
            except Exception:
                # fallback to predict
                p = np.asarray(m.predict(Xm)).reshape(-1)
                if p.min() < 0 or p.max() > 1:
                    p = logistic(p)
        elif hasattr(m, "predict"):
            p = np.asarray(m.predict(Xm)).reshape(-1)
            if p.min() < 0 or p.max() > 1:
                p = logistic(p)
        else:
            raise TypeError(f"Unsupported model type: {type(m)}")
        preds.append(p.astype(float))
    return np.mean(np.vstack(preds), axis=0)

# ---------- History + names ----------
def _coalesce(rr, target, *cands):
    for c in cands:
        if c in rr.columns:
            rr[target] = rr[c]
            return
    if target not in rr.columns:
        rr[target] = np.nan

def add_rollups_from_history(live: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich live rows with rolling (last-5) and event-specific (last-3) driver/team form.
    Keys: driver_id, (optional team_id), event_norm
    """
    feat = live.copy()

    # Normalize core keys on live
    feat["driver_id"] = pd.to_numeric(_get(feat, "driver_id", "DriverNumber", "driver", "DriverId"), errors="coerce").astype("Int64")
    if "team_id" in feat.columns:
        feat["team_id"] = pd.to_numeric(feat["team_id"], errors="coerce").astype("Int64")
    ev = _get(feat, "eventname", "EventName")
    feat["event_norm"] = (ev if ev is not None else "").astype(str).str.strip().str.lower()

    if not SILVER_RR_PATH.exists():
        return feat

    rr = pd.read_parquet(SILVER_RR_PATH).copy()
    _coalesce(rr, "driver_id", "driver_id", "DriverNumber", "driver", "DriverId")
    _coalesce(rr, "team_id", "team_id", "TeamId", "ConstructorId", "team")
    _coalesce(rr, "finish_position", "finish_position", "FinishPosition", "finishPos", "Position", "pos")
    _coalesce(rr, "points", "points", "Points", "pts", "Pts")
    _coalesce(rr, "eventname", "eventname", "EventName", "race_name")
    _coalesce(rr, "year", "year", "Year", "season", "Season")
    _coalesce(rr, "roundnumber", "roundnumber", "RoundNumber", "round", "Round")

    rr["driver_id"] = pd.to_numeric(rr["driver_id"], errors="coerce").astype("Int64")
    rr["team_id"] = pd.to_numeric(rr["team_id"], errors="coerce").astype("Int64")
    rr["finish_position"] = pd.to_numeric(rr["finish_position"], errors="coerce")
    rr["points"] = pd.to_numeric(rr["points"], errors="coerce")
    rr["event_norm"] = rr["eventname"].astype(str).str.strip().str.lower()

    rr = rr.sort_values(["driver_id", "year", "roundnumber"], kind="mergesort")

    # ---- rolling via transform (no FutureWarning)
    rr["driver_avg_finish_last5"] = rr.groupby("driver_id")["finish_position"] \
                                      .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    rr["driver_avg_points_last5"] = rr.groupby("driver_id")["points"] \
                                      .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    rr["team_avg_finish_last5"]   = rr.groupby("team_id")["finish_position"] \
                                      .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    rr["team_avg_points_last5"]   = rr.groupby("team_id")["points"] \
                                      .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())

    rr["driver_event_avg_points_hist3"] = rr.groupby(["driver_id", "event_norm"])["points"] \
                                            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    rr["driver_event_avg_finish_hist3"] = rr.groupby(["driver_id", "event_norm"])["finish_position"] \
                                            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    rr["team_event_avg_points_hist3"]   = rr.groupby(["team_id", "event_norm"])["points"] \
                                            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    rr["team_event_avg_finish_hist3"]   = rr.groupby(["team_id", "event_norm"])["finish_position"] \
                                            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())

    # ---- select last-known values
    last_driver_event = (
        rr.sort_values(["driver_id", "event_norm", "year", "roundnumber"], kind="mergesort")
          .groupby(["driver_id", "event_norm"], as_index=False)
          .tail(1)[["driver_id", "event_norm", "driver_event_avg_points_hist3", "driver_event_avg_finish_hist3"]]
          .drop_duplicates()
    )
    last_team_event = (
        rr.sort_values(["team_id", "event_norm", "year", "roundnumber"], kind="mergesort")
          .groupby(["team_id", "event_norm"], as_index=False)
          .tail(1)[["team_id", "event_norm", "team_event_avg_points_hist3", "team_event_avg_finish_hist3"]]
          .drop_duplicates()
    )
    last_driver = (
        rr.sort_values(["driver_id", "year", "roundnumber"], kind="mergesort")
          .groupby("driver_id", as_index=False)
          .tail(1)[["driver_id", "driver_avg_points_last5", "driver_avg_finish_last5"]]
          .drop_duplicates()
    )
    last_team = (
        rr.sort_values(["team_id", "year", "roundnumber"], kind="mergesort")
          .groupby("team_id", as_index=False)
          .tail(1)[["team_id", "team_avg_points_last5", "team_avg_finish_last5"]]
          .drop_duplicates()
    )

    # ---- merge into feat
    feat = feat.merge(last_driver, on="driver_id", how="left")
    if "team_id" in feat.columns:
        feat = feat.merge(last_team, on="team_id", how="left")
        feat = feat.merge(last_team_event, on=["team_id", "event_norm"], how="left")
    feat = feat.merge(last_driver_event, on=["driver_id", "event_norm"], how="left")

    return feat

def _normalize_names_cols(rr: pd.DataFrame) -> pd.DataFrame:
    r = rr.copy()
    # Prefer one-column full name
    full = _get(r, "FullName", "full_name", "driverName", "DriverName", "name")
    if full is not None:
        r["driver_name"] = full.astype(str)
        return r
    # Else build from first/last if present
    first = _get(r, "forename", "Forename", "firstName", "FirstName")
    last  = _get(r, "surname", "Surname", "lastName", "LastName")
    if first is not None or last is not None:
        r["driver_name"] = (first.fillna("").astype(str).str.strip() + " " +
                            last.fillna("").astype(str).str.strip()).str.strip()
    return r

def make_name_lookup(rr: pd.DataFrame) -> pd.DataFrame:
    rr2 = rr.copy()
    rr2["driver_id"] = pd.to_numeric(rr2["driver_id"], errors="coerce").astype("Int64")
    rr2 = _normalize_names_cols(rr2)
    if "driver_name" not in rr2.columns:
        rr2["driver_name"] = rr2["driver_id"].astype(str)

    team_col = None
    for c in ["team_id", "TeamName", "team_name", "Team", "Constructor"]:
        if c in rr2.columns:
            team_col = c
            break
    if team_col is None:
        team_col = "team_id"
        rr2["team_id"] = rr2.get("team_id", pd.Series(pd.NA, index=rr2.index))

    rr2 = rr2.sort_values(["year", "roundnumber"], kind="mergesort")
    last = rr2.groupby("driver_id", as_index=False).tail(1)
    keep = last[["driver_id", "driver_name", team_col]].drop_duplicates()
    keep = keep.rename(columns={team_col: "team_label"})
    keep["team_label"] = keep["team_label"].astype(str)
    return keep

def fallback_prior(feat_df: pd.DataFrame) -> np.ndarray:
    # Helper: numeric with safe median fill
    def _nz(series, default=0.0):
        v = pd.to_numeric(series, errors="coerce")
        if v.notna().any():
            m = v.median(skipna=True)
            if pd.isna(m):
                m = default
            return v.fillna(m)
        return pd.Series(default, index=series.index, dtype=float)

    s = pd.Series(0.0, index=feat_df.index, dtype=float)

    # --- Track-aware weighting: street circuits are much more grid-dependent ---
    event_col = None
    for c in ["eventname", "EventName", "race_name"]:
        if c in feat_df.columns:
            event_col = c
            break
    if event_col is not None and not feat_df.empty:
        ev0 = str(feat_df[event_col].iloc[0]).lower()
    else:
        ev0 = ""

    street_keywords = ["monaco", "singapore", "miami", "las vegas", "baku", "jeddah", "saudi", "azerbaijan"]
    is_street = any(k in ev0 for k in street_keywords)

    if is_street:
        # On tight street circuits, starting position matters a lot, but we still
        # want a realistic distribution, not a near 0/1 outcome.
        grid_w = 3.0
        quali_w = 2.0
        history_scale = 0.3
    else:
        # On normal tracks, keep grid/quali important but let form/history have
        # full influence.
        grid_w = 1.5
        quali_w = 1.2
        history_scale = 1.0

    # Grid helps (lower is better)
    if "gridposition" in feat_df.columns:
        gp = _nz(feat_df["gridposition"])
        s = s + (-gp) * grid_w

    # Quali rank helps (lower is better)
    if "quali_rank" in feat_df.columns:
        qr = _nz(feat_df["quali_rank"])
        s = s + (-qr) * quali_w

    # Recent driver/team form. If a column is entirely NaN, _nz returns zeros.
    for col, w, invert in [
        ("driver_avg_points_last5",       0.5, False),
        ("team_avg_points_last5",         0.3, False),
        ("driver_event_avg_points_hist3", 0.6, False),
        ("team_event_avg_points_hist3",   0.3, False),
        ("driver_avg_finish_last5",       0.3, True),
        ("team_avg_finish_last5",         0.2, True),
        ("driver_event_avg_finish_hist3", 0.3, True),
        ("team_event_avg_finish_hist3",   0.1, True),
    ]:
        if col in feat_df.columns:
            v = _nz(feat_df[col])
            if invert:
                v = -v
            s = s + v * (w * history_scale)

    # Convert scores to probabilities safely, with temperature to control sharpness.
    # Larger TEMP => flatter (less extreme) probabilities; smaller TEMP => sharper.
    TEMP = 6.0  # higher temperature -> less extreme probabilities
    scores = s.values.astype(float)
    p = _softmax(scores / TEMP)
    # If _softmax returned a placeholder or degenerate values, fall back to uniform
    if not np.isfinite(p).all() or np.allclose(p, 0):
        p = np.full(len(s), 1.0 / len(s))
    return p

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race-id", required=True, type=str)
    ap.add_argument("--event-name", required=True, type=str)
    ap.add_argument("--live-csv", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    # Load live CSV
    live = pd.read_csv(args.live_csv)

    # Coalesce grid/quali into canonical names
    cand_grid = _get(live, "gridposition", "grid", "grid_pos", "starting_grid", "start_grid", "start_position")
    if cand_grid is not None:
        live["gridposition"] = pd.to_numeric(cand_grid, errors="coerce")

    cand_qrank = _get(live, "quali_rank", "qualirank", "qualiposition", "qual_pos", "qualifying_position")
    if cand_qrank is not None:
        live["quali_rank"] = pd.to_numeric(cand_qrank, errors="coerce")

    # If there is no quali_rank column at all but we DO have gridposition,
    # use gridposition as quali_rank. Your live CSVs only have gridposition,
    # so this ensures quali_rank is always present for the model.
    if "quali_rank" not in live.columns and "gridposition" in live.columns:
        live["quali_rank"] = live["gridposition"]

    cand_qbest = _get(live, "quali_best_ms", "quali_best_time", "best_qual_time", "qualifying_time", "qtime")
    if cand_qbest is not None:
        live["quali_best_ms"] = pd.to_numeric(cand_qbest, errors="coerce")
        if live["quali_best_ms"].isna().any():
            live.loc[live["quali_best_ms"].isna(), "quali_best_ms"] = \
                live.loc[live["quali_best_ms"].isna(), cand_qbest.name].map(_parse_time_to_ms)

    cand_qgap = _get(live, "quali_gap_to_pole_ms", "q_gap_ms", "quali_gap", "gap_to_pole")
    if cand_qgap is not None:
        live["quali_gap_to_pole_ms"] = pd.to_numeric(cand_qgap, errors="coerce")

    live["raceid"] = args.race_id
    live["eventname"] = args.event_name

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
                feat = feat.merge(team_map, on="driver_id", how="left", suffixes=("", "_from_hist"))
                if "team_id_from_hist" in feat.columns:
                    feat["team_id"] = feat["team_id"].fillna(feat["team_id_from_hist"])
                    feat = feat.drop(columns=[c for c in ["team_id_from_hist"] if c in feat.columns])
        except Exception:
            pass
        if "team_id" not in feat.columns:
            feat["team_id"] = pd.NA

    # Build model matrix (base set; aligner will pad/trim to match model)
    BASE_FEATURES = [
        "gridposition",
        "quali_rank",
        "quali_best_ms",
        "quali_gap_to_pole_ms",
        "driver_avg_points_last5",
        "team_avg_points_last5",
        "driver_avg_finish_last5",
        "team_avg_finish_last5",
        "driver_event_avg_points_hist3",
        "team_event_avg_points_hist3",
        "driver_event_avg_finish_hist3",
        "team_event_avg_finish_hist3",
    ]
    X = feat.reindex(columns=[c for c in BASE_FEATURES if c in feat.columns]).apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())

    # Compute probabilities using the heuristic fallback prior
    # (grid position, quali, and history-based scoring) and normalize
    # within each race.
    prob_raw = fallback_prior(feat)

    out = feat[["raceid", "driver_id", "team_id", "eventname"]].copy()
    out["prob_win"] = prob_raw
    out["prob_win"] = out["prob_win"] / out.groupby("raceid")["prob_win"].transform("sum").replace(0, 1)

    # Attach names/team label
    if SILVER_RR_PATH.exists():
        rr = pd.read_parquet(SILVER_RR_PATH)
        names = make_name_lookup(rr)
        out = out.merge(names, on="driver_id", how="left")
    else:
        out["driver_name"] = out["driver_id"].astype(str)
        out["team_label"] = out["team_id"].astype(str)

    # Save + preview
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_cols = ["raceid", "driver_id", "driver_name", "team_id", "team_label", "prob_win", "eventname"]
    save_cols = [c for c in save_cols if c in out.columns]
    out_sorted = out.sort_values(["raceid", "prob_win"], ascending=[True, False])

    top = out_sorted[save_cols].head(args.top).copy()
    if "prob_win" in top.columns:
        top["prob_win"] = (top["prob_win"] * 100).round(2).astype(str) + "%"

    print(f"\n{args.race_id} — top {args.top}:")
    print(top.to_string(index=False))

    out_sorted[save_cols].to_csv(args.out, index=False) 
    print(f"\n📄 wrote {args.out}")

if __name__ == "__main__":
    main()