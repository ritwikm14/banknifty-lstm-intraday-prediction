# BANKNIFTY LSTM — Intraday Price Direction Prediction

This repository implements a complete machine-learning pipeline for predicting the next-minute price direction of BANKNIFTY using 1-minute OHLC data.  
The objective is to structure high-frequency financial data for sequence modeling and evaluate LSTM performance under short-horizon market noise.

---

## 1. Overview

The pipeline includes:

- Data loading and preprocessing  
- Feature engineering  
- Sliding-window sequence generation  
- Train/validation/test splitting  
- Feature scaling  
- LSTM model training  
- Evaluation on hold-out data  
- Saving the best model checkpoint  

The focus is on analytical evaluation rather than trading performance.

---

## 2. Project Structure

BankNiftyLSTM/
│
├── data/ # Raw data (ignored in version control)
├── models/ # Saved LSTM checkpoints
│
├── src/
│ ├── config.py # Configuration parameters and paths
│ ├── data_utils.py # Data processing, feature engineering, sequence creation
│ ├── lstm_model.py # PyTorch LSTM architecture
│ ├── serving.py # Optional model inference utilities
│ └── llm_explain.py # Optional LLM-based explanation module
│
├── train_lstm.py # Main training script
├── .gitignore
└── README.md



This structure follows standard ML project conventions for clarity and maintainability.

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

All features are normalized following sequence creation.

---

## 4. LSTM Model

The architecture consists of:

- Two LSTM layers  
- Hidden dimension: 64  
- Dropout: 0.2  
- Sigmoid output for binary classification  

**Loss Function:** Binary Cross-Entropy  
**Optimizer:** Adam  

The model predicts whether the next candle closes higher than the current one.

---

## 5. Installation

### 5.1 Create environment

```bash
conda create -n banknifty python=3.10
conda activate banknifty
```
## 5.2 Install dependencies

```bash
pip install numpy pandas torch scikit-learn python-dotenv
```
---

## 6. Running the Training Script

Execute the full training pipeline:

```bash
python train_lstm.py
This command performs:Loading and preprocessing of BANKNIFTY OHLC dataFeature engineeringSliding-window sequence constructionTrain/validation/test splittingFeature scalingLSTM model trainingValidation and test evaluationSaving the best model checkpoint to the models/ directoryThe script serves as the single entry point for the entire workflow.7. Performance SummaryObserved results on typical intraday datasets:Validation Accuracy: $\sim 50\%$Test Accuracy: $\sim 49\%$These results match empirical findings showing that next-minute price direction based solely on OHLC-derived features behaves like a random process due to market microstructure noise.8. InterpretationThis project is intended as an analytical machine learning exercise rather than a predictive trading system.Key takeaways:Extracting directional signal from 1-minute OHLC data is extremely challenging.LSTMs struggle with ultra-short-horizon forecasting where noise dominates.Practical quantitative models typically incorporate additional signals such as:Order-book imbalanceVolume and liquidity metricsVolatility regimesMarket microstructure featuresThe results highlight the limitations of deep learning models in high-frequency financial prediction tasks.
The results highlight the limitations of deep learning models in high-frequency financial prediction tasks.
