"""
Baseline LSTM for Hurricane Track and Intensity Prediction
Reads atlantic.csv and pacific.csv (HURDAT2 format), builds sequences per storm,
trains an LSTM with two heads (displacement + intensity), and evaluates results.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import warnings
import os
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASE         = os.path.dirname(os.path.abspath(__file__))
ATLANTIC_CSV = os.path.join(BASE, "csv", "atlantic.csv")
PACIFIC_CSV  = os.path.join(BASE, "csv", "pacific.csv")

SEQ_LEN      = 12       # input window (12 x 6h = 72h of history)
HORIZON      = 4        # forecast steps ahead: +6h, +12h, +18h, +24h
BATCH_SIZE   = 64
EPOCHS       = 80
LR           = 5e-4
HIDDEN_DIM   = 256
NUM_LAYERS   = 2
DROPOUT      = 0.5
TRAIN_CUTOFF = 2010     # storms <= this year = train; > this year = test
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# ─────────────────────────────────────────────
# 1.  LOAD & PARSE DATA
# ─────────────────────────────────────────────

def parse_hurdat2(filepath: str, basin: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "ID":               "storm_id",
        "Name":             "name",
        "Date":             "date",
        "Time":             "time",
        "Status":           "status",
        "Latitude":         "lat",
        "Longitude":        "lon",
        "Maximum Wind":     "max_wind",
        "Minimum Pressure": "min_pres",
    })
    df["name"]   = df["name"].str.strip()
    df["status"] = df["status"].str.strip()
    df["date"]   = df["date"].astype(str)
    df["time"]   = df["time"].astype(str)
    df["year"]   = df["date"].str[:4].astype(int)
    df["basin"]  = basin

    # Parse lat/lon strings e.g. "28.0N", "94.8W"
    df["lat"] = df["lat"].apply(lambda x: float(x[:-1]) * (1 if x[-1] == "N" else -1))
    df["lon"] = df["lon"].apply(lambda x: float(x[:-1]) * (-1 if x[-1] == "W" else 1))

    df.replace(-999, np.nan, inplace=True)
    return df


def load_data() -> pd.DataFrame:
    atl = parse_hurdat2(ATLANTIC_CSV, "ATL")
    pac = parse_hurdat2(PACIFIC_CSV,  "PAC")
    df  = pd.concat([atl, pac], ignore_index=True)

    # Replace any remaining -999 sentinel values with NaN
    df.replace(-999, np.nan, inplace=True)

    # Sort chronologically within each storm
    df["datetime"] = pd.to_datetime(
        df["date"] + df["time"].str.zfill(4),
        format="%Y%m%d%H%M",
        errors="coerce"
    )
    df.sort_values(["storm_id", "datetime"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features per storm."""
    groups = []
    for _, grp in df.groupby("storm_id", sort=False):
        grp = grp.copy().reset_index(drop=True)

        # Displacements — used as prediction targets
        grp["dlat"]  = grp["lat"].diff()
        grp["dlon"]  = grp["lon"].diff()
        grp["dwind"] = grp["max_wind"].diff()

        # Derived kinematic features
        grp["trans_speed"] = np.sqrt(grp["dlat"]**2 + grp["dlon"]**2)
        grp["heading"]     = np.arctan2(grp["dlat"], grp["dlon"])
        grp["storm_age"]   = range(len(grp))

        # Cyclical time encodings
        hour = grp["time"].astype(str).str.zfill(4).str[:2].astype(float)
        grp["sin_hour"] = np.sin(2 * np.pi * hour / 24)
        grp["cos_hour"] = np.cos(2 * np.pi * hour / 24)

        month = grp["date"].astype(str).str[4:6].astype(float)
        grp["sin_month"] = np.sin(2 * np.pi * month / 12)
        grp["cos_month"] = np.cos(2 * np.pi * month / 12)

        # Storm type and basin — strong predictors of intensity change
        status_map = {"TD": 0, "TS": 1, "HU": 2, "EX": 3, "SS": 4, "SD": 5, "LO": 6}
        grp["status_enc"] = grp["status"].map(status_map).fillna(0)
        grp["is_atl"] = (grp["basin"] == "ATL").astype(float)

        groups.append(grp)

    out = pd.concat(groups, ignore_index=True)

    # Remove rows with implausible displacements (data artifacts / date-line crossings)
    out = out[out["dlat"].abs() < 5]
    out = out[out["dlon"].abs() < 10]

    return out


# ─────────────────────────────────────────────
# 3.  SEQUENCE BUILDER
# ─────────────────────────────────────────────

INPUT_FEATURES = [
    "lat", "lon", "max_wind", "min_pres",
    "dlat", "dlon", "dwind",
    "trans_speed", "heading", "storm_age",
    "sin_hour", "cos_hour", "sin_month", "cos_month",
    "status_enc", "is_atl",
]


def build_sequences(df: pd.DataFrame, scaler: StandardScaler = None, fit_scaler: bool = False):
    X_list, y_list = [], []

    # Fill missing min_pres with per-storm median
    df["min_pres"] = df.groupby("storm_id")["min_pres"].transform(
        lambda x: x.fillna(x.median())
    )
    df[INPUT_FEATURES] = df[INPUT_FEATURES].fillna(0)

    feat_matrix = df[INPUT_FEATURES].values.astype(np.float32)

    if fit_scaler:
        scaler = StandardScaler()
        feat_matrix = scaler.fit_transform(feat_matrix)
    elif scaler is not None:
        feat_matrix = scaler.transform(feat_matrix)

    df_scaled = df.copy()
    df_scaled[INPUT_FEATURES] = feat_matrix

    for sid, grp in df_scaled.groupby("storm_id", sort=False):
        grp = grp.reset_index(drop=True)
        n   = len(grp)
        if n < SEQ_LEN + HORIZON:
            continue

        raw = df[df["storm_id"] == sid].reset_index(drop=True)

        for i in range(n - SEQ_LEN - HORIZON + 1):
            x_seq = grp[INPUT_FEATURES].values[i : i + SEQ_LEN]  # (SEQ_LEN, F)

            future_dlat     = raw["dlat"].values[i + SEQ_LEN : i + SEQ_LEN + HORIZON]
            future_dlon     = raw["dlon"].values[i + SEQ_LEN : i + SEQ_LEN + HORIZON]
            future_dwind    = raw["dwind"].values[i + SEQ_LEN : i + SEQ_LEN + HORIZON]
            future_wind_abs = raw["max_wind"].values[i + SEQ_LEN : i + SEQ_LEN + HORIZON]

            y = np.concatenate([future_dlat, future_dlon, future_dwind, future_wind_abs]).astype(np.float32)
            if np.any(np.isnan(y)):
                continue

            X_list.append(x_seq)
            y_list.append(y)

    X = np.stack(X_list)  # (N, SEQ_LEN, F)
    y = np.stack(y_list)  # (N, 3 * HORIZON)
    return X, y, scaler


# ─────────────────────────────────────────────
# 4.  DATASET & DATALOADER
# ─────────────────────────────────────────────

class HurricaneDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# 5.  MODEL
# ─────────────────────────────────────────────

class HurricaneLSTM(nn.Module):
    """
    Shared LSTM encoder -> attention pooling -> three task heads:
      - track head:     predicts dlat + dlon for each horizon step
      - intensity head: predicts dwind (delta) for each horizon step
      - wind_abs head:  predicts absolute max_wind for each horizon step
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, horizon):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout  = nn.Dropout(dropout)
        self.attn     = nn.Linear(hidden_dim, 1)      # attention scorer

        self.track_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, horizon * 2),  # dlat + dlon per step
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, horizon),       # dwind per step
        )
        self.wind_abs_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, horizon),       # absolute max_wind per step
        )

    def forward(self, x):
        out, _ = self.lstm(x)                          # (B, SEQ_LEN, H)
        scores  = self.attn(out).squeeze(-1)           # (B, SEQ_LEN)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (out * weights).sum(dim=1)           # (B, H) — attended summary
        context = self.dropout(context)

        track    = self.track_head(context)            # (B, 2*HORIZON)
        intensity = self.intensity_head(context)       # (B, HORIZON)
        wind_abs  = self.wind_abs_head(context)        # (B, HORIZON)
        return track, intensity, wind_abs


# ─────────────────────────────────────────────
# 6.  TRAINING
# ─────────────────────────────────────────────

def train(model, loader, optimizer, huber_track, huber_intensity):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        dlat     = y_batch[:, :HORIZON]
        dlon     = y_batch[:, HORIZON:2*HORIZON]
        dwind    = y_batch[:, 2*HORIZON:3*HORIZON]
        wind_abs = y_batch[:, 3*HORIZON:]
        track_target = torch.cat([dlat, dlon], dim=1)

        optimizer.zero_grad()
        track_pred, intensity_pred, wind_abs_pred = model(X_batch)

        loss_track = huber_track(track_pred, track_target)

        # Sample-weight intensity loss by magnitude of actual wind change
        dwind_mag   = dwind.abs().mean(dim=1)
        weights     = (1.0 + 3.0 * dwind_mag / (dwind_mag.max() + 1e-8)).detach()
        loss_dwind  = (weights * (intensity_pred - dwind).pow(2).mean(dim=1)).mean()

        # Auxiliary absolute wind head — downweighted to avoid dominating
        loss_wind_abs = huber_intensity(wind_abs_pred, wind_abs)

        loss = loss_track + loss_dwind + 0.1 * loss_wind_abs
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, huber_track, huber_intensity):
    model.eval()
    total_loss = 0
    all_track_pred, all_track_true   = [], []
    all_wind_pred,  all_wind_true    = [], []
    all_wind_abs_pred, all_wind_abs_true = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            dlat     = y_batch[:, :HORIZON]
            dlon     = y_batch[:, HORIZON:2*HORIZON]
            dwind    = y_batch[:, 2*HORIZON:3*HORIZON]
            wind_abs = y_batch[:, 3*HORIZON:]
            track_target = torch.cat([dlat, dlon], dim=1)

            track_pred, intensity_pred, wind_abs_pred = model(X_batch)

            dwind_mag     = dwind.abs().mean(dim=1)
            weights       = (1.0 + 3.0 * dwind_mag / (dwind_mag.max() + 1e-8)).detach()
            loss_dwind    = (weights * (intensity_pred - dwind).pow(2).mean(dim=1)).mean()
            loss          = (huber_track(track_pred, track_target)
                             + loss_dwind
                             + 0.1 * huber_intensity(wind_abs_pred, wind_abs))
            total_loss += loss.item()

            all_track_pred.append(track_pred.cpu().numpy())
            all_track_true.append(track_target.cpu().numpy())
            all_wind_pred.append(intensity_pred.cpu().numpy())
            all_wind_true.append(dwind.cpu().numpy())
            all_wind_abs_pred.append(wind_abs_pred.cpu().numpy())
            all_wind_abs_true.append(wind_abs.cpu().numpy())

    return (
        total_loss / len(loader),
        np.concatenate(all_track_pred),
        np.concatenate(all_track_true),
        np.concatenate(all_wind_pred),
        np.concatenate(all_wind_true),
        np.concatenate(all_wind_abs_pred),
        np.concatenate(all_wind_abs_true),
    )


# ─────────────────────────────────────────────
# 7.  METRICS
# ─────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * asin(sqrt(max(0, min(1, a))))


def rmse(pred, true):
    return np.sqrt(np.mean((pred - true)**2))


# ─────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────

def main():
    print(f"Using device: {DEVICE}\n")

    # Load
    print("Loading data...")
    df = load_data()
    print(f"  Total records: {len(df):,}  |  Unique storms: {df['storm_id'].nunique()}")

    # Feature engineering (includes outlier clipping)
    df = engineer_features(df)

    # Train / test split by year
    train_storms = df[df["year"] <= TRAIN_CUTOFF]["storm_id"].unique()
    test_storms  = df[df["year"] >  TRAIN_CUTOFF]["storm_id"].unique()
    df_train = df[df["storm_id"].isin(train_storms)]
    df_test  = df[df["storm_id"].isin(test_storms)]
    print(f"  Train storms: {len(train_storms)} ({df['year'].min()}-{TRAIN_CUTOFF})  "
          f"|  Test storms: {len(test_storms)} ({TRAIN_CUTOFF+1}-{df['year'].max()})\n")

    # Build sequences
    print("Building sequences...")
    X_train, y_train, scaler = build_sequences(df_train, fit_scaler=True)
    X_test,  y_test,  _      = build_sequences(df_test,  scaler=scaler)
    print(f"  Train samples: {len(X_train):,}  |  Test samples: {len(X_test):,}")
    print(f"  Input shape: {X_train.shape}  |  Target shape: {y_train.shape}\n")

    train_loader = DataLoader(HurricaneDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(HurricaneDataset(X_test,  y_test),
                              batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = HurricaneLSTM(
        input_dim=len(INPUT_FEATURES),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=HORIZON,
    ).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    optimizer    = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.7, min_lr=1e-5)
    huber_track     = nn.HuberLoss(delta=1.0)
    huber_intensity = nn.HuberLoss(delta=10.0)

    # Training loop
    train_losses, test_losses = [], []
    best_test_loss  = float("inf")
    best_state      = None
    patience_count  = 0
    EARLY_STOP      = 20   # stop if test loss hasn't improved for this many epochs

    print("Training...")
    for epoch in range(1, EPOCHS + 1):
        tr_loss     = train(model, train_loader, optimizer, huber_track, huber_intensity)
        te_loss, *_ = evaluate(model, test_loader, huber_track, huber_intensity)
        scheduler.step(te_loss)

        train_losses.append(tr_loss)
        test_losses.append(te_loss)

        if te_loss < best_test_loss:
            best_test_loss = te_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  |  Train Loss: {tr_loss:.4f}  |  "
                  f"Test Loss: {te_loss:.4f}  |  LR: {optimizer.param_groups[0]['lr']:.2e}")

        if patience_count >= EARLY_STOP:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {EARLY_STOP} epochs)")
            break

    # Evaluate best model
    model.load_state_dict(best_state)
    _, track_pred, track_true, wind_pred, wind_true, wind_abs_pred, wind_abs_true = \
        evaluate(model, test_loader, huber_track, huber_intensity)

    pred_dlat = track_pred[:, :HORIZON]
    pred_dlon = track_pred[:, HORIZON:]
    true_dlat = track_true[:, :HORIZON]
    true_dlon = track_true[:, HORIZON:]

    print("\n--- Results ---")
    horizons = [6, 12, 18, 24]
    for h_idx, h_hr in enumerate(horizons):
        w_rmse      = rmse(wind_pred[:, h_idx],     wind_true[:, h_idx])
        wabs_rmse   = rmse(wind_abs_pred[:, h_idx], wind_abs_true[:, h_idx])
        dlat_rmse   = rmse(pred_dlat[:, h_idx],     true_dlat[:, h_idx])
        dlon_rmse   = rmse(pred_dlon[:, h_idx],     true_dlon[:, h_idx])
        print(f"  +{h_hr:2d}h  |  dWind RMSE: {w_rmse:5.2f} kt  |  "
              f"Abs Wind RMSE: {wabs_rmse:5.2f} kt  |  "
              f"Track (deg): dlat={dlat_rmse:.4f}  dlon={dlon_rmse:.4f}")

    print(f"\n  Overall dWind RMSE:   {rmse(wind_pred, wind_true):.2f} kt")
    print(f"  Overall Abs Wind RMSE: {rmse(wind_abs_pred, wind_abs_true):.2f} kt")
    print("---\n")

    # Plots
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # Loss curves
    axes[0].plot(train_losses, label="Train")
    axes[0].plot(test_losses,  label="Test")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # dWind scatter (+24h)
    axes[1].scatter(wind_true[:500, -1], wind_pred[:500, -1], alpha=0.3, s=10)
    lim = max(abs(wind_true[:, -1]).max(), abs(wind_pred[:, -1]).max()) * 1.1
    axes[1].plot([-lim, lim], [-lim, lim], "r--", lw=1)
    axes[1].set_title("Wind Delta Pred vs True (+24h)")
    axes[1].set_xlabel("True Dwind (kt)")
    axes[1].set_ylabel("Pred Dwind (kt)")
    axes[1].grid(True, alpha=0.3)

    # Absolute wind scatter (+24h)
    axes[2].scatter(wind_abs_true[:500, -1], wind_abs_pred[:500, -1], alpha=0.3, s=10)
    lim2 = max(wind_abs_true[:, -1].max(), wind_abs_pred[:, -1].max()) * 1.1
    axes[2].plot([0, lim2], [0, lim2], "r--", lw=1)
    axes[2].set_title("Abs Wind Pred vs True (+24h)")
    axes[2].set_xlabel("True Wind (kt)")
    axes[2].set_ylabel("Pred Wind (kt)")
    axes[2].grid(True, alpha=0.3)

    # RMSE by horizon — all three metrics
    dwind_rmses   = [rmse(wind_pred[:, h],     wind_true[:, h])     for h in range(HORIZON)]
    abswind_rmses = [rmse(wind_abs_pred[:, h], wind_abs_true[:, h]) for h in range(HORIZON)]
    dlat_rmses    = [rmse(pred_dlat[:, h],     true_dlat[:, h])     for h in range(HORIZON)]
    dlon_rmses    = [rmse(pred_dlon[:, h],     true_dlon[:, h])     for h in range(HORIZON)]

    ax = axes[3]
    ax.plot(horizons, dwind_rmses,   marker="o", label="dWind (kt)")
    ax.plot(horizons, abswind_rmses, marker="o", label="Abs Wind (kt)")
    ax.plot(horizons, dlat_rmses,    marker="s", label="dlat (°)", linestyle="--")
    ax.plot(horizons, dlon_rmses,    marker="s", label="dlon (°)", linestyle="--")
    ax.set_title("RMSE by Forecast Horizon")
    ax.set_xlabel("Lead Time (h)")
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(BASE, "lstm_results.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")

    # Save model
    model_path = os.path.join(BASE, "hurricane_lstm.pt")
    torch.save({"model_state": best_state, "scaler": scaler}, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()