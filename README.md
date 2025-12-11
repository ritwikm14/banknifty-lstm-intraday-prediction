# BANKNIFTY LSTM – Intraday Price Direction Prediction

This repository implements a complete machine-learning pipeline for preparing high-frequency BANKNIFTY 1-minute OHLC data, engineering quantitative features, converting them into sliding-window sequences, training an LSTM model, and evaluating its predictive performance on next-minute price direction.

The goal is not to build a trading strategy, but to demonstrate how intraday  
financial data behaves when structured for supervised learning — and to analyze why deep learning models frequently converge to random-like accuracy on ultrashort-horizon predictions.

---

## 1. Problem Overview

The task is to predict whether the **next 1-minute candle** closes **up (1)** or **down (0)** relative to the current candle.

This is a classical but notoriously difficult problem because:

- Microstructure noise dominates at 1-minute resolution  
- OHLC-only features contain very weak directional signal  
- Bid–ask bounce causes randomness in small candle movements  
- Market efficiency makes short-horizon prediction extremely limited  

This project shows what a complete deep-learning pipeline looks like for such  
data — and why accuracy often converges near 50%.

---

## 2. High-Level Architecture

The ML workflow includes:

### **A. Data Processing**
- Load raw OHLC data  
- Clean timestamps and missing values  

### **B. Feature Engineering**
- Returns  
- High–Low range  
- Moving averages  
- Realized volatility  

### **C. Sequence Construction**
- Convert time-series into fixed-length LSTM windows  
- Use stride (`STEP=3`) to manage dataset size  

### **D. Train/Validation/Test Split**
- Time-based split (no shuffling to avoid leakage)

### **E. Scaling**
- Fit scaler only on training set  
- Apply to val/test sets  

### **F. LSTM Model Training**
- 2-layer LSTM  
- Hidden size 64  
- Dropout 0.2  
- Binary classification (sigmoid output)

### **G. Evaluation & Saving**
- Evaluate on hold-out test set  
- Save best checkpoint under `models/`

#### **Architecture Diagram**

```text
             Raw OHLC Data
                    |
                    v
        Feature Engineering (returns, vol, MAs)
                    |
                    v
        Sliding Window Sequences (LSTM-ready)
                    |
                    v
       Train / Validation / Test Split (time-based)
                    |
                    v
            Scaling (train → val/test)
                    |
                    v
                LSTM Model
                    |
                    v
        Evaluation + Model Checkpoint Saving
```

---

## 3. Tech Stack

- **Python 3.10**

- **PyTorch** for LSTM model

- **NumPy / Pandas** for preprocessing

- **scikit-learn** for scaling & metrics

- **dotenv** for API keys (optional)

- **Conda** for environment management

---

## 4. Project Structure

```text
BankNiftyLSTM/
│
├── data/                     # Raw OHLC data (ignored in Git)
├── models/                   # Saved LSTM model checkpoints
│
├── src/
│   ├── config.py             # Global paths and project parameters
│   ├── data_utils.py         # Feature engineering, scaling, sequence builder
│   ├── lstm_model.py         # PyTorch LSTM architecture
│   ├── serving.py            # (Optional) model inference utilities
│   └── llm_explain.py        # (Optional) LLM-based explanations
│
├── train_lstm.py             # Main end-to-end training pipeline
├── .gitignore
└── README.md


```

---

## 5. Feature Engineering

The following quantitative features are generated from raw OHLC candles:

- Close price

- Log returns

- High–Low range

- Short-term return moving average

- Long-term return moving average

- Short-term realized volatility

- Long-term realized volatility

These are common features used across quant finance and academic literature for
intraday price modeling.

After sequence creation, all features are standard-scaled.

---

## 6. Installation

### **6.1 Create Conda Environment**
```bash
conda create -n banknifty python=3.10
conda activate banknifty
```

### **6.2 Install Dependencies**
```bash
pip install numpy pandas torch scikit-learn python-dotenv
```
(Optional)
```bash
pip freeze > requirements.txt
```
---

## 7. Running the Training Script

Execute:
```bash
python train_lstm.py
```
This command performs:

- Loading and preprocessing of BANKNIFTY OHLC data

- Feature engineering

- Sliding-window sequence construction

- Train/validation/test splitting

- Feature scaling

- LSTM model training

- Validation and test evaluation

- Saving the best model checkpoint under models/

The script serves as the single entry point for the entire ML pipeline.

---

## 8. Performance Summary

Observed performance on typical BANKNIFTY intraday datasets:

- **Validation Accuracy**: ~50%
- **Test Accuracy**: ~49%

These values align with well-established research showing that:

Market microstructure, latency uncertainty, and limited information density all
contribute to this outcome.

---

## 9. Interpretation

This project illustrates the limitations of using deep learning for
ultrashort-horizon financial forecasting.

**Key insights:**

- 1-minute candle direction is dominated by noise

- LSTMs have difficulty capturing predictive structure at this horizon

- More complex features are required for meaningful predictive lift

**Quantitative models in production typically rely on:**

- Order-book imbalance

- Volume and liquidity metrics

- Queue position changes

- Volatility & regime indicators

- Microstructure features

Thus, this project is best viewed as:

*A demonstration of the ML pipeline, not a trading strategy.*

---

## 10. Possible Extensions

For more realistic market modeling, you could add:

**A. Additional Feature Sets**

- VWAP

- Market depth

- Tick-rule buy/sell pressure

- Spread / slippage metrics

**B. Regime Detection**

- Volatility buckets

- Trend/non-trend regimes

- Autocorrelation regimes

**C. Alternative Architectures**

- GRUs

- Temporal CNNs

- Transformers

- DeepLOB-style models

**D. LLM-Based Interpretation**

Use llm_explain.py to generate natural-language summaries for model outputs.

**E. Online Learning**

Update the model as new market data streams in.
