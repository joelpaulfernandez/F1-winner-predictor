"""PyTorch MLP: alternative winner-probability model trained on the same features as LightGBM."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

FEATURES = [
    "gridposition", "qualiposition",
    "quali_best_ms", "quali_gap_to_pole_ms", "quali_rank",
    "driver_avg_finish_last5", "driver_avg_points_last5",
    "team_avg_points_last5", "team_avg_finish_last5",
    "driver_event_avg_finish_hist3", "driver_event_avg_points_hist3",
    "team_event_avg_finish_hist3", "team_event_avg_points_hist3",
    "airtempmean", "tracktempmean", "humidity_mean",
    "rainprobproxy", "windspeed_mean",
]

MODEL_PATH = Path("registry/winner_mlp.pt")


class WinnerMLP(nn.Module):
    """Three-layer MLP for binary win-probability estimation."""

    def __init__(self, n_features: int = len(FEATURES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: Optional[str] = None,
) -> WinnerMLP:
    """Train WinnerMLP on pre-scaled feature matrix X and binary labels y."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = WinnerMLP(n_features=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # class imbalance: ~1 winner per 20 drivers
    pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}/{epochs}  loss={epoch_loss / len(dataset):.4f}")

    return model.cpu()


def predict_mlp(model: WinnerMLP, X: np.ndarray) -> np.ndarray:
    """Return sigmoid win probabilities for each row in X."""
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32)
        logits = model(x_t)
        return torch.sigmoid(logits).numpy()


def save_model(model: WinnerMLP, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved MLP weights → {path}")


def load_model(path: Path = MODEL_PATH, n_features: int = len(FEATURES)) -> WinnerMLP:
    model = WinnerMLP(n_features=n_features)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


if __name__ == "__main__":
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    fv_path = Path("data/gold/fv_preRace.parquet")
    if not fv_path.exists():
        raise SystemExit(f"Feature vector not found: {fv_path}\nRun the data pipeline first.")

    df = pd.read_parquet(fv_path)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    X_raw = df[FEATURES].astype(float).fillna(df[FEATURES].median()).values
    y = df["winner"].astype(int).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print(f"Training WinnerMLP on {len(X)} samples ({y.sum()} winners)...")
    model = train_mlp(X, y, epochs=50)

    probs = predict_mlp(model, X)
    print(f"Mean predicted prob: {probs.mean():.4f}  |  Max: {probs.max():.4f}")

    save_model(model)

    import joblib
    Path("registry").mkdir(exist_ok=True)
    joblib.dump(scaler, "registry/winner_mlp_scaler.joblib")
    print("Saved scaler → registry/winner_mlp_scaler.joblib")
