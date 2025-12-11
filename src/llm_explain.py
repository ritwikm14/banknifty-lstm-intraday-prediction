from __future__ import annotations

from typing import Dict, Any, Optional

import pandas as pd


def _classify_probability(prob_up: float) -> str:
    """
    Map probability of an UP move into a simple qualitative view.
    """
    if prob_up >= 0.65:
        return "strongly bullish"
    elif prob_up >= 0.55:
        return "mildly bullish"
    elif prob_up > 0.45:
        return "roughly balanced between up and down"
    elif prob_up > 0.35:
        return "mildly bearish"
    else:
        return "strongly bearish"


def _describe_trend(price_change_pct: Optional[float]) -> str:
    if price_change_pct is None:
        return "Price action over the window was roughly flat."

    pct = float(price_change_pct)
    if pct > 0.30:
        return f"Price has been in a clear intraday up-move, gaining about {pct:.2f}% over the window."
    elif pct > 0.05:
        return f"Price has drifted higher, up roughly {pct:.2f}% over the window."
    elif pct > -0.05:
        return "Price action over the window has been basically sideways, with only minor net change."
    elif pct > -0.30:
        return f"Price has eased lower, down around {pct:.2f}% over the window."
    else:
        return f"Price has been in a clear intraday down-move, losing about {pct:.2f}% over the window."


def _describe_volatility(vol_return: Optional[float]) -> str:
    if vol_return is None:
        return "Volatility is measured using the standard deviation of 1-minute returns."

    v = float(vol_return)
    # Very rough bands tuned for intraday index data
    if v < 0.00015:
        return f"Intraday volatility has been **low**, with a 1-minute return standard deviation near {v:.5f}."
    elif v < 0.00035:
        return f"Intraday volatility has been **normal**, with a 1-minute return standard deviation near {v:.5f}."
    else:
        return f"Intraday volatility has been **elevated**, with a 1-minute return standard deviation around {v:.5f}."


def explain_lstm_prediction(pred: Dict[str, Any]) -> str:
    """
    Build a deterministic, human-readable explanation from the LSTM prediction
    and window summary. This does not call any external LLM – it is purely
    rule-based so that it is stable and reproducible.

    Expected pred structure (as returned by make_single_prediction):
      - instrument: str
      - prob_up: float in [0,1]
      - label_up: int (0/1)
      - window_summary: dict with:
          * start_datetime, end_datetime (str)
          * start_price, end_price (float)
          * price_change_pct, avg_return, vol_return, end_index
    """
    instrument = pred.get("instrument", "BANKNIFTY")
    prob_up = float(pred.get("prob_up", 0.0))
    label_up = int(pred.get("label_up", 0))
    ws = pred.get("window_summary", {}) or {}

    start_dt = ws.get("start_datetime")
    end_dt = ws.get("end_datetime")
    start_price = ws.get("start_price")
    end_price = ws.get("end_price")
    price_change_pct = ws.get("price_change_pct")
    avg_ret = ws.get("avg_return")
    vol_ret = ws.get("vol_return")

    # Parse times once for nicer display
    try:
        if start_dt and end_dt:
            s = pd.to_datetime(start_dt)
            e = pd.to_datetime(end_dt)
            # e.g. "2013-01-01 from 09:47 to 10:16"
            if s.date() == e.date():
                window_text = (
                    f"on {s.date().isoformat()} from {s.strftime('%H:%M')} to {e.strftime('%H:%M')}"
                )
            else:
                window_text = f"from {s} to {e}"
        else:
            window_text = "over the selected 30-minute window"
    except Exception:
        window_text = "over the selected 30-minute window"

    direction_word = "UP" if prob_up >= 0.5 else "DOWN"
    qualitative_view = _classify_probability(prob_up)
    trend_sentence = _describe_trend(price_change_pct)
    vol_sentence = _describe_volatility(vol_ret)

    core_line = (
        f"For {instrument}, {window_text}, the LSTM model estimates a "
        f"{prob_up * 100:.2f}% probability that the **next 1-minute candle** "
        f"will close higher than the current one (base case: **{direction_word}**). "
        f"This corresponds to a **{qualitative_view}** short-term signal."
    )

    if (
        start_price is not None
        and end_price is not None
        and price_change_pct is not None
    ):
        price_line = (
            f"During this window, price moved from {start_price:.2f} to {end_price:.2f}, "
            f"a net change of about {float(price_change_pct):.2f}%."
        )
    else:
        price_line = "Price change information for the window is unavailable."

    if avg_ret is not None:
        avg_line = (
            f"The average 1-minute return over the window was approximately {float(avg_ret):.5f}."
        )
    else:
        avg_line = ""

    disclaimer = (
        "This is a **purely statistical, very short-horizon signal**. It does not take into "
        "account order-book depth, news, macro events, or higher-timeframe structure. "
        "Use it only as an educational example, **not** as trading or investment advice."
    )

    parts = [core_line, trend_sentence, price_line, vol_sentence, avg_line, disclaimer]
    explanation = " ".join(p for p in parts if p)

    return explanation
