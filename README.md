# BANKNIFTY LSTM — Intraday Price Direction Prediction

This repository implements a complete machine learning pipeline for predicting the next-minute price direction of BANKNIFTY using 1-minute OHLC data. The project demonstrates how to structure high-frequency financial time-series for sequence modeling using an LSTM classifier. The focus is analytical rather than predictive performance.

---

## 1. Overview

The pipeline performs the following steps:

1. Load and preprocess 1-minute OHLC data  
2. Engineer features relevant to intraday movement  
3. Generate fixed-length sliding windows for sequence modeling  
4. Split data into train, validation, and test sets  
5. Normalize features  
6. Train an LSTM classifier  
7. Evaluate performance and save the best model  

The objective is to evaluate whether short-horizon directional signal exists in price-based features.

---

## 2. Project Structure

BankNiftyLSTM/
│
├── data/ # Raw data (ignored in version control)
├── models/ # Saved model checkpoints
│
├── src/
│ ├── config.py # Configuration parameters and paths
│ ├── data_utils.py # Data loading, feature engineering, sequence creation
│ ├── lstm_model.py # PyTorch LSTM model
│ ├── serving.py # Optional inference utilities
│ └── llm_explain.py # Optional LLM-based explanation module
│
├── train_lstm.py # Main training script
├── .gitignore
└── README.md

yaml
Copy code

This structure follows standard machine-learning project organization for clarity and extensibility.

---

## 3. Feature Engineering

The following features are derived from raw OHLC data:

- Close price  
- Return  
- High–Low range  
- Short-term return moving average  
- Long-term return moving average  
- Short-term realized volatility  
- Long-term realized volatility  

All features are normalized after sequence generation.

---

## 4. Model Architecture

The model is a two-layer LSTM with:

- Hidden dimension: 64  
- Number of layers: 2  
- Dropout: 0.2  
- Sigmoid output for binary direction prediction  

Loss function: Binary Cross-Entropy  
Optimizer: Adam  

The model predicts whether the next candle closes higher than the current one.

---

## 5. Installation

### 5.1 Create environment
```bash
conda create -n banknifty python=3.10
conda activate banknifty
5.2 Install dependencies
bash
Copy code
pip install numpy pandas torch scikit-learn python-dotenv
(If needed, a requirements.txt can be generated.)

6. Running the Training Script
Execute:

bash
Copy code
python train_lstm.py
This runs the full pipeline: feature engineering, sequence creation, model training, validation, test evaluation, and saving the best checkpoint under models/.

7. Performance Summary
Typical observed results:

Validation Accuracy: ~50%

Test Accuracy: ~49%

These results are consistent with empirical evidence that next-minute direction forecasting using only price-based features is statistically indistinguishable from randomness due to market microstructure noise.

8. Interpretation
The model serves as an analytical demonstration rather than a predictive trading system.
The results highlight:

The challenge of extracting signal from high-frequency financial data

The limitations of LSTMs for ultra-short-horizon forecasting

The importance of richer features (order-book data, volume imbalance, etc.) in real-world quantitative research
