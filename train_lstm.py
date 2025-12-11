from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader

from src.config import DATA_RAW, MODELS_DIR, SEQ_LEN, TRAIN_FRAC, VAL_FRAC
from src.data_utils import (
    load_and_engineer,
    build_sequences,
    train_val_test_split,
    scale_sequences,
)
from src.lstm_model import BankNiftyLSTM


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()
    epoch_loss = 0.0
    all_y_true = []
    all_y_pred = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)  # (batch,1)

        optimizer.zero_grad()
        probs = model(X_batch)                    # (batch,1)
        loss = criterion(probs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * X_batch.size(0)

        preds = (probs.detach().cpu().numpy() >= 0.5).astype(int)
        all_y_true.extend(y_batch.cpu().numpy().flatten())
        all_y_pred.extend(preds.flatten())

    avg_loss = epoch_loss / len(loader.dataset)
    acc = accuracy_score(all_y_true, all_y_pred)
    return avg_loss, acc


def eval_epoch(
    model,
    loader,
    criterion,
    device,
):
    model.eval()
    epoch_loss = 0.0
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            probs = model(X_batch)
            loss = criterion(probs, y_batch)

            epoch_loss += loss.item() * X_batch.size(0)

            preds = (probs.cpu().numpy() >= 0.5).astype(int)
            all_y_true.extend(y_batch.cpu().numpy().flatten())
            all_y_pred.extend(preds.flatten())

    avg_loss = epoch_loss / len(loader.dataset)
    acc = accuracy_score(all_y_true, all_y_pred)
    return avg_loss, acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load + engineer features
    print("Loading and engineering features...")
    df = load_and_engineer(DATA_RAW)

    feature_cols = [
        "close",
        "return",
        "hl_range",
        "ret_ma_short",
        "ret_ma_long",
        "ret_vol_short",
        "ret_vol_long",
    ]
    print(f"Feature columns: {feature_cols}")
    print(f"Total engineered rows: {len(df)}")

    # 2) Build sequences with stride (step) to manage size
    STEP = 3  # use every 3rd window → faster on 1M rows
    print(f"Building sequences with seq_len={SEQ_LEN}, step={STEP} ...")
    X, y, dt_arr = build_sequences(df, seq_len=SEQ_LEN, feature_cols=feature_cols, step=STEP)
    print(f"Sequences shape: {X.shape}, Targets shape: {y.shape}")

    # 3) Split train/val/test
    splits = train_val_test_split(X, y, dt_arr, TRAIN_FRAC, VAL_FRAC)

    # 4) Scale
    scaled_splits, scaler = scale_sequences(splits)

    train_ds = SequenceDataset(
        scaled_splits["train"]["X"], scaled_splits["train"]["y"]
    )
    val_ds = SequenceDataset(
        scaled_splits["val"]["X"], scaled_splits["val"]["y"]
    )
    test_ds = SequenceDataset(
        scaled_splits["test"]["X"], scaled_splits["test"]["y"]
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # 5) Model, loss, optimizer
    input_dim = scaled_splits["train"]["X"].shape[-1]
    model = BankNiftyLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2)
    model.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 6) Training loop
    n_epochs = 50  # ⬅️ updated from 10 to 50
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "seq_len": SEQ_LEN,
                "feature_cols": feature_cols,
                "scaler": scaler,
            }

    # 7) Evaluate on test set with best model
    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}")

    # 8) Save best model
    out_path = MODELS_DIR / "banknifty_lstm.pt"
    torch.save(best_state, out_path)
    print(f"Saved best LSTM model to {out_path}")


if __name__ == "__main__":
    main()
