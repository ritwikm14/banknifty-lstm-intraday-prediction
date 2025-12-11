from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import DATA_RAW, SEQ_LEN, INSTRUMENT


def load_and_engineer(xls_path: Path | str = DATA_RAW) -> pd.DataFrame:
    """
    Load intraday BANKNIFTY data from Excel and engineer features + target.

    Expected columns in Excel:
        instrument, date, time, open, high, low, close, datetime (optional)

    Target:
        up_next_bar = 1 if next close > current close else 0.
    """
    xls_path = Path(xls_path)

    # Read Excel file
    # If engine issues: pip install openpyxl
    df = pd.read_excel(xls_path)

    # Build datetime column if needed
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        if {"date", "time"}.issubset(df.columns):
            df["datetime"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str)
            )
        else:
            raise ValueError("Need 'datetime' or 'date'+'time' columns in Excel file.")

    # Normalize column names we care about (open/high/low/close already ok)
    rename_map = {}
    for col in ["instrument", "open", "high", "low", "close"]:
        if col in df.columns:
            rename_map[col] = col
    df = df.rename(columns=rename_map)

    # Filter to our instrument (BANKNIFTY)
    df = df[df["instrument"] == INSTRUMENT].copy()

    # Sort by time
    df = df.sort_values(["instrument", "datetime"]).reset_index(drop=True)

    def add_features(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()

        # Close-to-close returns
        g["return"] = g["close"].pct_change()

        # Intrabar high-low range
        g["hl_range"] = (g["high"] - g["low"]) / g["close"]

        # Rolling mean & volatility of returns
        g["ret_ma_short"] = g["return"].rolling(window=10, min_periods=10).mean()
        g["ret_ma_long"] = g["return"].rolling(window=30, min_periods=30).mean()
        g["ret_vol_short"] = g["return"].rolling(window=10, min_periods=10).std()
        g["ret_vol_long"] = g["return"].rolling(window=30, min_periods=30).std()

        # Target: next bar close vs current
        g["future_close"] = g["close"].shift(-1)
        g["up_next_bar"] = (g["future_close"] > g["close"]).astype("float").astype("Int64")

        return g

    df_feat = df.groupby("instrument", group_keys=False).apply(add_features)

    feature_cols = [
        "close",
        "return",
        "hl_range",
        "ret_ma_short",
        "ret_ma_long",
        "ret_vol_short",
        "ret_vol_long",
    ]

    # Drop rows where we don't have all features or target
    df_feat = df_feat.dropna(subset=feature_cols + ["up_next_bar"]).reset_index(drop=True)

    return df_feat


def build_sequences(
    df: pd.DataFrame,
    seq_len: int,
    feature_cols: List[str],
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a time-ordered DataFrame into LSTM sequences.

    Args:
        df: DataFrame with engineered features + 'up_next_bar' and 'datetime'.
        seq_len: number of time steps per sequence (e.g. 30).
        step: stride between consecutive windows (e.g. 1 uses every bar,
              3 uses every 3rd bar → smaller dataset, faster training).

    Returns:
        X: (num_samples, seq_len, num_features)
        y: (num_samples,)
        end_datetimes: (num_samples,) array of datetime for each sample's last bar
    """
    df = df.sort_values("datetime").reset_index(drop=True)

    feature_array = df[feature_cols].values.astype("float32")
    target_array = df["up_next_bar"].values.astype(int)
    datetimes = df["datetime"].values

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    dt_list: List[np.datetime64] = []

    n = len(df)

    # We need seq_len bars ending at index i, where i <= n-2 because target uses close_{i+1}
    for i in range(seq_len - 1, n - 1, step):
        seq = feature_array[i - seq_len + 1 : i + 1]
        y_val = target_array[i]
        dt_end = datetimes[i]

        if seq.shape[0] != seq_len:
            continue

        X_list.append(seq)
        y_list.append(y_val)
        dt_list.append(dt_end)

    X = np.stack(X_list)  # (samples, seq_len, num_features)
    y = np.array(y_list)
    dt_arr = np.array(dt_list)

    return X, y, dt_arr


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    dt_arr: np.ndarray,
    train_frac: float,
    val_frac: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Temporal split based on index (data already sorted by time).
    """
    n = X.shape[0]
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    splits = {
        "train": {
            "X": X[:train_end],
            "y": y[:train_end],
            "dt": dt_arr[:train_end],
        },
        "val": {
            "X": X[train_end:val_end],
            "y": y[train_end:val_end],
            "dt": dt_arr[train_end:val_end],
        },
        "test": {
            "X": X[val_end:],
            "y": y[val_end:],
            "dt": dt_arr[val_end:],
        },
    }
    return splits


def scale_sequences(
    splits: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], StandardScaler]:
    """
    Fit a StandardScaler on the TRAIN features (flattened across time),
    then apply to train/val/test.
    """
    scaler = StandardScaler()

    X_train = splits["train"]["X"]
    n_train, seq_len, num_feat = X_train.shape

    # Flatten to (n_train * seq_len, num_feat)
    X_train_flat = X_train.reshape(-1, num_feat)
    scaler.fit(X_train_flat)

    scaled_splits: Dict[str, Dict[str, np.ndarray]] = {}

    for split_name, data in splits.items():
        X = data["X"]
        n_s, s_len, n_feat = X.shape
        X_flat = X.reshape(-1, n_feat)
        X_scaled_flat = scaler.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(n_s, s_len, n_feat)

        scaled_splits[split_name] = {
            "X": X_scaled,
            "y": data["y"],
            "dt": data["dt"],
        }

    return scaled_splits, scaler
