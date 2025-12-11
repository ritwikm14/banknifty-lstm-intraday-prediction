# api/main.py

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from src.config import DATA_RAW, MODELS_DIR
from src.data_utils import load_and_engineer
from src.serving import load_lstm_artifact, make_single_prediction
from src.llm_explain import explain_lstm_prediction

load_dotenv()

app = FastAPI(title="BankNIFTY LSTM Intraday API", version="0.1.0")

# Load data + model once at startup
df_full = load_and_engineer(DATA_RAW)
checkpoint_path = MODELS_DIR / "banknifty_lstm.pt"
artifact = load_lstm_artifact(checkpoint_path)


class PredictRequest(BaseModel):
    # 0 or None → latest window
    end_index: Optional[int] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_intraday_lstm")
def predict_intraday_lstm(req: PredictRequest):
    """
    Predict using LSTM on intraday BANKNIFTY data.

    If end_index is None or 0, use the latest available window.
    """
    try:
        pred = make_single_prediction(
            df_full=df_full,
            artifact=artifact,
            end_index=req.end_index,
        )
    except Exception as e:
        # Expose precise error in Swagger UI instead of generic 500
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}",
        )

    explanation = explain_lstm_prediction(pred)
    pred["llm_explanation"] = explanation
    return pred
