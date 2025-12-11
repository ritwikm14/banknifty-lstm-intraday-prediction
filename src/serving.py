# src/serving.py

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from src.lstm_model import BankNiftyLSTM


def load_lstm_artifact(model_path: Path | str) -> Dict:
    """
    Load the trained BankNifty LSTM checkpoint that was saved in train_lstm.py.

    Expected keys in the checkpoint:
        - "model_state_dict"
        - "input_dim"
        - "seq_len"
        - "feature_cols"
        - "scaler"
    """
    model_path = Path(model_path)

    # Important: weights_only=False so we can unpickle the sklearn scaler
    checkpoint = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )

    expected_keys = {"model_state_dict", "input_dim", "seq_len", "feature_cols", "scaler"}
    missing = expected_keys - set(checkpoint.keys())
    if missing:
        raise KeyError(
            f"Checkpoint at {model_path} is missing keys: {missing}. "
            f"Found keys: {list(checkpoint.keys())}"
        )

    input_dim: int = checkpoint["input_dim"]
    seq_len: int = checkpoint["seq_len"]
    feature_cols = checkpoint["feature_cols"]
    scaler = checkpoint["scaler"]

    model = BankNiftyLSTM(input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "input_dim": input_dim,
        "seq_len": seq_len,
        "feature_cols": feature_cols,
        "scaler": scaler,
    }


@torch.no_grad()
def make_single_prediction(
    df_full: pd.DataFrame,
    artifact: Dict,
    end_index: Optional[int] = None,
) -> Dict:
    """
    Make a single intraday prediction using the latest LSTM window.

    Args:
        df_full: engineered DataFrame with 'datetime', 'close', 'return', etc.
        artifact: dict returned by load_lstm_artifact(...)
        end_index: index in df_full to end the LSTM window at.
                   If None or 0 ⇒ use the latest available window (n - 2).

    Returns:
        JSON-serializable dict with:
        - instrument, end_datetime, prob_up, label_up
        - window_summary{...}
    """
    model = artifact["model"]
    seq_len: int = artifact["seq_len"]
    feature_cols = artifact["feature_cols"]
    scaler = artifact["scaler"]

    # Ensure time-ordered
    df_sorted = df_full.sort_values("datetime").reset_index(drop=True)
    n = len(df_sorted)

    # Choose end index
    if end_index is None or end_index == 0:
        # n-1 is last bar, but target is based on next bar, so stop at n-2
        end_idx = n - 2
    else:
        end_idx = int(end_index)
        if end_idx < seq_len - 1 or end_idx >= n - 1:
            raise ValueError(
                f"end_index {end_idx} is out of valid range "
                f"[{seq_len - 1}, {n - 2}] for seq_len={seq_len} and n={n}"
            )

    # Extract the window [end_idx - seq_len + 1 : end_idx]
    window_df = df_sorted.iloc[end_idx - seq_len + 1 : end_idx + 1].copy()

    # Scale features using training-time scaler
    feat_mat = window_df[feature_cols].values.astype("float32")
    feat_scaled = scaler.transform(feat_mat)
    x = torch.from_numpy(feat_scaled).unsqueeze(0)  # (1, seq_len, input_dim)

    prob_up = float(model(x).item())
    label_up = int(prob_up >= 0.5)

    # Build window summary for dashboard / explanation
    start_dt = window_df["datetime"].iloc[0]
    end_dt = window_df["datetime"].iloc[-1]
    start_price = float(window_df["close"].iloc[0])
    end_price = float(window_df["close"].iloc[-1])
    price_change_pct = (end_price - start_price) / start_price * 100.0

    avg_ret = float(window_df["return"].mean())
    vol_ret = float(window_df["return"].std())

    return {
        "instrument": "BANKNIFTY",
        "end_datetime": str(end_dt),
        "prob_up": round(prob_up, 4),
        "label_up": label_up,
        "window_summary": {
            "instrument": "BANKNIFTY",
            "window_len": int(seq_len),
            "start_datetime": str(start_dt),
            "end_datetime": str(end_dt),
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "price_change_pct": round(price_change_pct, 4),
            "avg_return": round(avg_ret, 8) if not np.isnan(avg_ret) else None,
            "vol_return": round(vol_ret, 8) if not np.isnan(vol_ret) else None,
            "end_index": int(end_idx),
        },
    }
