from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Your BANKNIFTY file lives directly on D:\
DATA_RAW = Path(r"D:\BANKNIFTY_2013_CLEAN.xlsx")

DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# LSTM sequence + split settings
SEQ_LEN = 30          # 30-minute window
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15       # test = 1 - TRAIN_FRAC - VAL_FRAC

INSTRUMENT = "BANKNIFTY"
