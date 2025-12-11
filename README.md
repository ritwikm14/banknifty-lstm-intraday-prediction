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
pip install numpy pandas torch scikit-learn python-dotenv


(If required, a requirements.txt file can be provided.)

6. Running the Training Script

Run the training pipeline using:

python train_lstm.py


This executes the complete workflow including:

Feature engineering

Sliding-window sequence generation

Train/validation/test splitting

Feature scaling

LSTM training

Evaluation on the test set

Saving the best model checkpoint under models/

7. Performance Summary

Expected performance on typical intraday datasets:

Validation Accuracy: ~50%

Test Accuracy: ~49%

This outcome aligns with established findings that next-minute direction prediction using only OHLC-based features is statistically close to random, due to the inherent noise and unpredictability of high-frequency markets.

8. Interpretation

This project is intended as an analytical and educational demonstration rather than a predictive trading model.

Key observations:

Extracting meaningful signal from 1-minute price data is extremely challenging.

LSTMs struggle with ultra-short-horizon forecasting where noise dominates.

Real-world quantitative forecasting typically requires additional data, such as:

Order-book depth and imbalance

Trade volume and liquidity metrics

Volatility regimes

Market microstructure signals

The results reinforce the limitations of deep learning models in high-frequency financial prediction tasks.

The importance of richer features (order-book data, volume imbalance, etc.) in real-world quantitative research
