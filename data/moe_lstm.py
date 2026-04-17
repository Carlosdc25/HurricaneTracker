"""
Mixture of Experts LSTM for Hurricane Prediction — Autoregressive Decoder
Builds on base_lstm.py — reuses the data pipeline, constants, and encoder architecture.

Architecture:
  Input sequence (B, SEQ_LEN, F)
       ↓
  Shared LSTM encoder (pretrained from base_lstm)  →  h_n, c_n
       ↓
  Autoregressive decoder loop  (HORIZON steps)
  ┌──────────────────────────────────────────────────────────────────┐
  │  Decoder LSTM step t                                             │
  │    input:  [dlat, dlon, dwind, wind_abs] from step t-1 (zeros   │
  │            at t=0); hidden state initialised from encoder h_n   │
  │    output: step_ctx (B, H)                                      │
  │       ↓                        ↓                                │
  │  Gating network            3 Experts (one step each)            │
  │  [w1, w2, w3]              track(2) / dwind(1) / wind_abs(1)    │
  │       ↓                                                         │
  │  output_t = w1*E1_t + w2*E2_t + w3*E3_t  →  fed back as input  │
  └──────────────────────────────────────────────────────────────────┘
  Final: stack outputs across steps → same shapes as non-AR baseline
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Reuse data pipeline and constants from base_lstm
from base_lstm import (
    load_data, engineer_features, build_sequences,
    HurricaneDataset, INPUT_FEATURES, rmse,
    SEQ_LEN, HORIZON, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
    BATCH_SIZE, EPOCHS, LR, TRAIN_CUTOFF, DEVICE,
)

# ─────────────────────────────────────────────
# MoE CONFIG
# ─────────────────────────────────────────────

NUM_EXPERTS     = 3
LOAD_BALANCE_W  = 1.0    # weight of auxiliary load-balancing loss
PRETRAINED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hurricane_lstm.pt")


# ─────────────────────────────────────────────
# 1.  ENCODER  (shared across all experts)
# ─────────────────────────────────────────────

class HurricaneEncoder(nn.Module):
    """
    LSTM + attention pooling — identical to the encoder half of HurricaneLSTM.
    Takes (B, SEQ_LEN, F) and returns context (B, H) plus the LSTM hidden
    states h_n, c_n which seed the autoregressive decoder.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.attn    = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)                # (B, SEQ_LEN, H)
        scores  = self.attn(out).squeeze(-1)           # (B, SEQ_LEN)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (out * weights).sum(dim=1)           # (B, H)
        return self.dropout(context), h_n, c_n


# ─────────────────────────────────────────────
# 2.  EXPERT  (one set of prediction heads)
# ─────────────────────────────────────────────

class Expert(nn.Module):
    """
    One expert — three prediction heads for a SINGLE forecast step.
    Called once per autoregressive step; the decoder LSTM provides a fresh
    context vector at each step so experts can specialise per horizon.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.track_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),   # dlat + dlon for ONE step
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),   # dwind for ONE step
        )
        self.wind_abs_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),   # absolute max_wind for ONE step
        )

    def forward(self, context):
        return (
            self.track_head(context),
            self.intensity_head(context),
            self.wind_abs_head(context),
        )


# ─────────────────────────────────────────────
# 3.  GATING NETWORK
# ─────────────────────────────────────────────

class GatingNetwork(nn.Module):
    """
    Small MLP that reads the context vector and outputs a probability
    distribution over experts (softmax, sums to 1).
    Training noise encourages exploration and prevents early expert collapse.
    """
    def __init__(self, hidden_dim, num_experts):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts),
        )

    def forward(self, context, training=False):
        logits = self.gate(context)
        if training:
            # Noise encourages exploration during training, prevents early collapse
            logits = logits + torch.randn_like(logits) * 0.3
        return torch.softmax(logits, dim=-1)   # (B, K)


# ─────────────────────────────────────────────
# 4.  FULL MoE MODEL
# ─────────────────────────────────────────────

class MoEHurricaneLSTM(nn.Module):
    """
    Shared encoder → autoregressive decoder → gating + K experts per step.

    Each forecast step:
      1. Decoder LSTM advances one step using the previous prediction as input.
      2. Gating network reads the decoder's hidden state → expert weights.
      3. Each expert predicts (dlat, dlon, dwind, wind_abs) for THIS step.
      4. Weighted sum of expert outputs → prediction fed back as next decoder input.

    This produces naturally degrading RMSE across horizons (unlike the
    all-at-once baseline) because errors compound through the AR loop.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, horizon, num_experts):
        super().__init__()
        self.encoder     = HurricaneEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.gating      = GatingNetwork(hidden_dim, num_experts)
        self.experts     = nn.ModuleList([Expert(hidden_dim) for _ in range(num_experts)])
        # Decoder LSTM: input = 4 predicted features from previous step
        # [dlat, dlon, dwind, wind_abs]; initialised from encoder's final hidden state
        self.decoder_lstm = nn.LSTM(
            input_size=4,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.horizon     = horizon
        self.num_experts = num_experts

    def forward(self, x):
        B = x.size(0)
        _, h_n, c_n = self.encoder(x)          # h_n/c_n carry the storm history

        # Seed decoder from the encoder's last-layer hidden state
        h_dec = h_n[-1:].contiguous()           # (1, B, H)
        c_dec = c_n[-1:].contiguous()           # (1, B, H)

        # First decoder input: zeros — no previous prediction exists
        dec_in = torch.zeros(B, 1, 4, device=x.device)

        all_dlat, all_dlon, all_dwind, all_wind_abs = [], [], [], []
        all_gates = []

        for _ in range(self.horizon):
            dec_out, (h_dec, c_dec) = self.decoder_lstm(dec_in, (h_dec, c_dec))
            step_ctx = dec_out.squeeze(1)       # (B, H) — context for this forecast step

            gates = self.gating(step_ctx, training=self.training)  # (B, K)
            all_gates.append(gates)

            exp_track, exp_dwind, exp_wind = [], [], []
            for expert in self.experts:
                t, i, w = expert(step_ctx)
                exp_track.append(t)             # (B, 2)
                exp_dwind.append(i)             # (B, 1)
                exp_wind.append(w)              # (B, 1)

            exp_track = torch.stack(exp_track, dim=1)  # (B, K, 2)
            exp_dwind = torch.stack(exp_dwind, dim=1)  # (B, K, 1)
            exp_wind  = torch.stack(exp_wind,  dim=1)  # (B, K, 1)

            g = gates.unsqueeze(-1)
            track_step = (g * exp_track).sum(dim=1)    # (B, 2)
            dwind_step = (g * exp_dwind).sum(dim=1)    # (B, 1)
            wind_step  = (g * exp_wind ).sum(dim=1)    # (B, 1)

            all_dlat.append(track_step[:, 0])
            all_dlon.append(track_step[:, 1])
            all_dwind.append(dwind_step.squeeze(1))
            all_wind_abs.append(wind_step.squeeze(1))

            # Feed prediction back as next decoder input (autoregressive)
            dec_in = torch.cat([track_step, dwind_step, wind_step], dim=-1).unsqueeze(1)

        # Assemble into the same output format as the non-AR model so train/eval are unchanged
        track     = torch.stack(all_dlat + all_dlon, dim=1)  # (B, 2*HORIZON)
        intensity = torch.stack(all_dwind, dim=1)             # (B, HORIZON)
        wind_abs  = torch.stack(all_wind_abs, dim=1)          # (B, HORIZON)
        gates_avg = torch.stack(all_gates, dim=1).mean(dim=1) # (B, K) averaged over steps

        return track, intensity, wind_abs, gates_avg


# ─────────────────────────────────────────────
# 5.  LOAD PRETRAINED ENCODER
# ─────────────────────────────────────────────

def load_pretrained_encoder(model, path):
    """
    Copy LSTM and attention weights from a saved HurricaneLSTM checkpoint
    into the MoE encoder. Gives the model a head start instead of random init.
    """
    if not os.path.exists(path):
        print(f"  No pretrained weights found at {path}, training from scratch.")
        return model

    checkpoint   = torch.load(path, map_location=DEVICE, weights_only=False)
    saved_state  = checkpoint["model_state"]

    encoder_state = {
        k: v for k, v in saved_state.items()
        if k.startswith("lstm.") or k.startswith("attn.")
    }

    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    loaded = len(encoder_state) - len(missing)
    print(f"  Loaded {loaded} pretrained encoder weights from {path}")
    if missing:
        print(f"  Missing keys: {missing}")
    return model


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
        track_pred, intensity_pred, wind_abs_pred, gates = model(X_batch)

        loss_track    = huber_track(track_pred, track_target)
        dwind_mag     = dwind.abs().mean(dim=1)
        weights       = (1.0 + 3.0 * dwind_mag / (dwind_mag.max() + 1e-8)).detach()
        loss_dwind    = (weights * (intensity_pred - dwind).pow(2).mean(dim=1)).mean()
        loss_wind_abs = huber_intensity(wind_abs_pred, wind_abs)

        # Load balancing: penalise uneven expert usage across the batch
        avg_gate       = gates.mean(dim=0)
        load_bal_loss  = model.num_experts * (avg_gate * avg_gate).sum()

        loss = loss_track + loss_dwind + 0.1 * loss_wind_abs + LOAD_BALANCE_W * load_bal_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────
# 7.  EVALUATION
# ─────────────────────────────────────────────

def evaluate(model, loader, huber_track, huber_intensity):
    model.eval()
    total_loss = 0
    all_track_pred, all_track_true       = [], []
    all_wind_pred,  all_wind_true        = [], []
    all_wind_abs_pred, all_wind_abs_true = [], []
    all_gates                            = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            dlat     = y_batch[:, :HORIZON]
            dlon     = y_batch[:, HORIZON:2*HORIZON]
            dwind    = y_batch[:, 2*HORIZON:3*HORIZON]
            wind_abs = y_batch[:, 3*HORIZON:]
            track_target = torch.cat([dlat, dlon], dim=1)

            track_pred, intensity_pred, wind_abs_pred, gates = model(X_batch)

            dwind_mag    = dwind.abs().mean(dim=1)
            weights      = (1.0 + 3.0 * dwind_mag / (dwind_mag.max() + 1e-8)).detach()
            loss_dwind   = (weights * (intensity_pred - dwind).pow(2).mean(dim=1)).mean()
            avg_gate     = gates.mean(dim=0)
            load_bal     = model.num_experts * (avg_gate * avg_gate).sum()
            loss = (huber_track(track_pred, track_target)
                    + loss_dwind
                    + 0.1 * huber_intensity(wind_abs_pred, wind_abs)
                    + LOAD_BALANCE_W * load_bal)
            total_loss += loss.item()

            all_track_pred.append(track_pred.cpu().numpy())
            all_track_true.append(track_target.cpu().numpy())
            all_wind_pred.append(intensity_pred.cpu().numpy())
            all_wind_true.append(dwind.cpu().numpy())
            all_wind_abs_pred.append(wind_abs_pred.cpu().numpy())
            all_wind_abs_true.append(wind_abs.cpu().numpy())
            all_gates.append(gates.cpu().numpy())

    return (
        total_loss / len(loader),
        np.concatenate(all_track_pred),
        np.concatenate(all_track_true),
        np.concatenate(all_wind_pred),
        np.concatenate(all_wind_true),
        np.concatenate(all_wind_abs_pred),
        np.concatenate(all_wind_abs_true),
        np.concatenate(all_gates),          # (N, K) — for inspecting expert usage
    )


# ─────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────

def main():
    print(f"Using device: {DEVICE}\n")

    # Data
    print("Loading data...")
    df = load_data()
    df = engineer_features(df)

    train_storms = df[df["year"] <= TRAIN_CUTOFF]["storm_id"].unique()
    test_storms  = df[df["year"] >  TRAIN_CUTOFF]["storm_id"].unique()
    df_train = df[df["storm_id"].isin(train_storms)]
    df_test  = df[df["storm_id"].isin(test_storms)]
    print(f"  Train storms: {len(train_storms)}  |  Test storms: {len(test_storms)}\n")

    print("Building sequences...")
    X_train, y_train, scaler = build_sequences(df_train, fit_scaler=True)
    X_test,  y_test,  _      = build_sequences(df_test,  scaler=scaler)
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}\n")

    train_loader = DataLoader(HurricaneDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader  = DataLoader(HurricaneDataset(X_test,  y_test),
                              batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = MoEHurricaneLSTM(
        input_dim=len(INPUT_FEATURES),
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        horizon=HORIZON,
        num_experts=NUM_EXPERTS,
    ).to(DEVICE)

    print("Loading pretrained encoder weights...")
    model = load_pretrained_encoder(model, PRETRAINED_PATH)
    print(f"  Total model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    huber_track     = nn.HuberLoss(delta=1.0)
    huber_intensity = nn.HuberLoss(delta=10.0)

    train_losses, test_losses = [], []
    best_test_loss = float("inf")
    best_state     = None
    patience_count = 0
    EARLY_STOP     = 20
    WARMUP_EPOCHS  = 5 # epochs to train only experts + gating (encoder frozen)

    # Phase 1: freeze encoder, train only experts + gating
    print(f"Phase 1: training experts + gating only ({WARMUP_EPOCHS} epochs, encoder frozen)...")
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=10, factor=0.7, min_lr=1e-5)

    for epoch in range(1, WARMUP_EPOCHS + 1):
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
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{WARMUP_EPOCHS}  |  Train: {tr_loss:.4f}  |  "
                  f"Test: {te_loss:.4f}  |  LR: {optimizer.param_groups[0]['lr']:.2e}")

    # ── Phase 2: unfreeze encoder, fine-tune with differential LRs ──
    # Encoder gets 10× lower LR than decoder/experts since it's already pretrained
    print(f"\nPhase 2: fine-tuning all parameters (encoder LR={LR*0.02:.0e}, rest LR={LR*0.2:.0e})...")
    for param in model.encoder.parameters():
        param.requires_grad = True

    other_params = (list(model.decoder_lstm.parameters())
                    + list(model.gating.parameters())
                    + list(model.experts.parameters()))
    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": LR * 0.02},
        {"params": other_params,               "lr": LR * 0.2},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=10, factor=0.7, min_lr=1e-6)
    patience_count = 0

    for epoch in range(WARMUP_EPOCHS + 1, EPOCHS + 1):
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
        if (epoch - WARMUP_EPOCHS) % 10 == 0 or epoch == WARMUP_EPOCHS + 1:
            enc_lr   = optimizer.param_groups[0]['lr']
            other_lr = optimizer.param_groups[1]['lr']
            print(f"  Epoch {epoch:3d}/{EPOCHS}  |  Train: {tr_loss:.4f}  |  "
                  f"Test: {te_loss:.4f}  |  LR enc: {enc_lr:.2e}  other: {other_lr:.2e}")
        if patience_count >= EARLY_STOP:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # Final evaluation
    model.load_state_dict(best_state)
    (_, track_pred, track_true,
     wind_pred, wind_true,
     wind_abs_pred, wind_abs_true,
     gates) = evaluate(model, test_loader, huber_track, huber_intensity)

    pred_dlat = track_pred[:, :HORIZON]
    pred_dlon = track_pred[:, HORIZON:]
    true_dlat = track_true[:, :HORIZON]
    true_dlon = track_true[:, HORIZON:]

    print("\n--- Results ---")
    horizons = [6, 12, 18, 24]
    for h_idx, h_hr in enumerate(horizons):
        print(f"  +{h_hr:2d}h  |  dWind RMSE: {rmse(wind_pred[:, h_idx], wind_true[:, h_idx]):5.2f} kt  |  "
              f"Abs Wind RMSE: {rmse(wind_abs_pred[:, h_idx], wind_abs_true[:, h_idx]):5.2f} kt  |  "
              f"Track (deg): dlat={rmse(pred_dlat[:, h_idx], true_dlat[:, h_idx]):.4f}  "
              f"dlon={rmse(pred_dlon[:, h_idx], true_dlon[:, h_idx]):.4f}")

    print(f"\n  Overall dWind RMSE:    {rmse(wind_pred, wind_true):.2f} kt")
    print(f"  Overall Abs Wind RMSE: {rmse(wind_abs_pred, wind_abs_true):.2f} kt")

    # Expert usage — how often each expert is dominant
    dominant_expert = gates.argmax(axis=1)
    print("\n--- Expert Usage ---")
    for k in range(NUM_EXPERTS):
        pct = (dominant_expert == k).mean() * 100
        avg_weight = gates[:, k].mean() * 100
        print(f"  Expert {k+1}: dominant {pct:5.1f}% of samples  |  avg gate weight {avg_weight:.1f}%")
    print("---\n")

    # Plots
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].plot(train_losses, label="Train")
    axes[0].plot(test_losses,  label="Test")
    axes[0].axvline(WARMUP_EPOCHS - 1, color="gray", linestyle="--", alpha=0.5, label="Phase 2 start")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(wind_true[:500, -1], wind_pred[:500, -1], alpha=0.3, s=10)
    lim = max(abs(wind_true[:, -1]).max(), abs(wind_pred[:, -1]).max()) * 1.1
    axes[1].plot([-lim, lim], [-lim, lim], "r--", lw=1)
    axes[1].set_title("Wind Delta Pred vs True (+24h)")
    axes[1].set_xlabel("True Dwind (kt)")
    axes[1].set_ylabel("Pred Dwind (kt)")
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(range(1, NUM_EXPERTS + 1),
                [(dominant_expert == k).mean() * 100 for k in range(NUM_EXPERTS)],
                color="steelblue")
    axes[2].set_title("Expert Usage (% dominant)")
    axes[2].set_xlabel("Expert")
    axes[2].set_ylabel("% of samples")
    axes[2].grid(True, alpha=0.3, axis="y")

    dwind_rmses   = [rmse(wind_pred[:, h], wind_true[:, h]) for h in range(HORIZON)]
    abswind_rmses = [rmse(wind_abs_pred[:, h], wind_abs_true[:, h]) for h in range(HORIZON)]
    axes[3].plot(horizons, dwind_rmses,   marker="o", label="dWind (kt)")
    axes[3].plot(horizons, abswind_rmses, marker="o", label="Abs Wind (kt)")
    axes[3].set_title("RMSE by Forecast Horizon")
    axes[3].set_xlabel("Lead Time (h)")
    axes[3].set_ylabel("RMSE (kt)")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_results.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hurricane_moe.pt")
    torch.save({"model_state": best_state, "scaler": scaler}, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
