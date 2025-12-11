# app/dashboard.py

import sys
from pathlib import Path

# --- Make sure Python can find the top-level `src` package ---
ROOT = Path(__file__).resolve().parents[1]  # .../BankNiftyLSTM
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.config import DATA_RAW, INSTRUMENT, SEQ_LEN
from src.data_utils import load_and_engineer

API_URL = "http://127.0.0.1:8000/predict_intraday_lstm"


# ---------- Data loading (local, for plotting only) ----------

@st.cache_data(show_spinner="Loading BANKNIFTY data & features…")
def load_local_features() -> pd.DataFrame:
    df_feat = load_and_engineer(DATA_RAW)
    df_feat["datetime"] = pd.to_datetime(df_feat["datetime"])
    return df_feat


df_feat = load_local_features()

# The very last index in df is NOT valid for prediction – we need at least 1 bar
# AFTER the window to label UP/DOWN. So use max_index - 1 as the latest index.
raw_max_index = int(df_feat.index.max())
valid_max_index = max(0, raw_max_index - 1)


# ---------- Streamlit UI ----------

st.set_page_config(
    page_title="BankNIFTY LSTM Intraday Demo",
    layout="wide",
)

# Sidebar – just like your old UI
st.sidebar.header("Prediction settings")
end_index = st.sidebar.number_input(
    "LSTM window end index",
    min_value=0,
    max_value=valid_max_index,
    value=valid_max_index,      # default to latest valid window
    step=1,
    help="Use an index from the engineered dataset. 0 = earliest, max = latest valid.",
)

if st.sidebar.button("Get prediction"):
    # ---------- Call FastAPI backend ----------
    payload = {"end_index": int(end_index)}
    try:
        resp = requests.post(API_URL, json=payload, timeout=30)
    except Exception as e:
        st.error(f"Error calling backend: {e}")
        st.stop()

    if resp.status_code != 200:
        st.error(f"Backend error {resp.status_code}: {resp.text}")
        st.stop()

    data = resp.json()

    # ---------- Top header ----------
    st.title("📈 BankNIFTY LSTM Intraday Dashboard")
    st.caption(
        "This dashboard calls the FastAPI service on "
        "http://127.0.0.1:8000/predict_intraday_lstm, plots the last 30 minutes "
        "of BANKNIFTY price, and shows the model’s view."
    )

    prob_up = float(data.get("prob_up", 0.0))
    label_up = int(data.get("label_up", 0))
    ws = data.get("window_summary", {}) or {}

    # Direction from label
    direction = "UP" if label_up == 1 else "DOWN"

    price_change_pct = float(ws.get("price_change_pct", 0.0))
    vol_return = float(ws.get("vol_return", 0.0))
    window_len = int(ws.get("window_len", SEQ_LEN))

    # ---------- Metrics row ----------
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Direction (model label)", direction)
    with m2:
        st.metric("Prob(next bar UP)", f"{prob_up * 100:.2f} %")
    with m3:
        st.metric("Window price change", f"{price_change_pct:+.2f} %")
    with m4:
        st.metric("Volatility (σ of returns)", f"{vol_return:.5f}")

    # Small line describing window index + span
    start_dt_raw = ws.get("start_datetime")
    end_dt_raw = ws.get("end_datetime")

    window_idx = int(ws.get("end_index", end_index))
    if start_dt_raw and end_dt_raw:
        start_dt = pd.to_datetime(start_dt_raw)
        end_dt = pd.to_datetime(end_dt_raw)
        st.caption(
            f"Window {window_idx} | {INSTRUMENT} from "
            f"{start_dt:%Y-%m-%d %H:%M} to {end_dt:%Y-%m-%d %H:%M}"
        )
    else:
        start_dt = end_dt = None

    # ---------- Line chart of last 30 mins ----------
    st.subheader("Price over last 30 minutes before prediction")

    if start_dt is None or end_dt is None:
        st.warning("No window price data returned from the backend.")
        window_df = None
    else:
        mask = (df_feat["datetime"] >= start_dt) & (df_feat["datetime"] <= end_dt)
        window_df = df_feat.loc[mask].copy()

        if window_df.empty:
            st.warning("No matching rows in local data for this window.")
            window_df = None
        else:
            window_df["time_str"] = window_df["datetime"].dt.strftime("%H:%M")

            x_vals = list(range(len(window_df)))
            fig, ax = plt.subplots(figsize=(10, 4))

            ax.plot(x_vals, window_df["close"], marker="o")
            ax.axvline(x_vals[-1], linestyle="--", color="gray", alpha=0.8)

            ax.set_title(
                f"{INSTRUMENT} – last {window_len} minutes before {end_dt:%Y-%m-%d %H:%M}"
            )
            ax.set_xlabel("Time (HH:MM)")
            ax.set_ylabel("Close price")

            ax.set_xticks(x_vals)
            ax.set_xticklabels(window_df["time_str"], rotation=45, ha="right")

            plt.tight_layout()
            st.pyplot(fig)

    # ---------- Model explanation ----------
    st.subheader("Model explanation")
    explanation = data.get("llm_explanation") or "No explanation returned."
    st.write(explanation)
    st.caption(
        "⚠️ This is an educational demo only and **not** investment or trading advice."
    )

    # ---------- Expanders ----------
    with st.expander("Show raw API response (debug)", expanded=False):
        st.json(data)

    if window_df is not None:
        with st.expander("Show underlying 30-minute window data", expanded=False):
            display_cols = [
                "datetime",
                "time_str",
                "open",
                "high",
                "low",
                "close",
            ]
            display_cols = [c for c in display_cols if c in window_df.columns]
            st.dataframe(window_df[display_cols], use_container_width=True)

else:
    st.title("📈 BankNIFTY LSTM Intraday Dashboard")
    st.info("Set **LSTM window end index** in the sidebar and click **Get prediction** to begin.")
