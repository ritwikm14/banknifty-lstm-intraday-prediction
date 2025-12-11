This repository contains a fully implemented deep learning pipeline built to evaluate whether a Long Short-Term Memory (LSTM) neural network can predict the next-minute price direction of BANKNIFTY using 1-minute OHLC data.
The project follows a research-oriented workflow similar to what quantitative analysts and ML engineers use in financial time-series modeling.

The primary objective is signal extraction, noise analysis, and performance evaluation, not trading strategy generation.

1. Project Overview

Financial markets at the 1-minute frequency exhibit extremely low signal-to-noise ratios.
This project investigates whether engineered features combined with an LSTM can produce any predictive advantage over randomness.

The workflow includes:

Data ingestion and feature engineering

Sliding-window sequence construction

Train/validation/test splitting

Scaling and normalization

LSTM training and evaluation

Saving best model weights

The final results demonstrate that 1-minute direction forecasting behaves as a near-random process, consistent with microstructure theory.

2. Key Features
Modeling & Data Pipeline

Uses 1-minute OHLC data

Engineers price-based and volatility-based features

Generates sequential windows for LSTM input

Splits data chronologically (train/val/test)

Conducts model training with BCE loss and accuracy metrics

Saves best-performing model checkpoint

Technical Highlights

Two-layer LSTM

Hidden dimension = 64

Dropout = 0.2

Binary classification (up/down)

Sliding window generation with configurable sequence length

3. Project Structure

The repository is structured for clarity, modularity, and scalability:

BankNiftyLSTM/
│
├── api/                       # Optional REST API endpoints for future extensions
│   └── main.py
│
├── app/                       # Optional UI or dashboard components
│   └── dashboard.py
│
├── data/                      # Raw data directory (ignored in git)
│
├── models/                    # Saved LSTM model checkpoints (.pt files)
│
├── src/                       # Core ML and data processing modules
│   ├── config.py              # Paths, hyperparameters, constants
│   ├── data_utils.py          # Data loading, feature engineering, sequence creation
│   ├── lstm_model.py          # PyTorch LSTM architecture
│   ├── serving.py             # Optional model inference utilities
│   └── llm_explain.py         # Optional LLM-based natural language explanation
│
├── train_lstm.py              # Main training script (entry point)
│
├── .gitignore                 # Excludes sensitive files (e.g., .env, models/)
│
└── README.md                  # Project documentation


This structure follows standard machine learning project organization and enables easy extensibility.

4. Installation and Setup
4.1 Create a Conda environment
conda create -n banknifty python=3.10
conda activate banknifty

4.2 Install required libraries

(If you want, I can generate a requirements.txt)

pip install numpy pandas torch scikit-learn python-dotenv

5. API Keys (Optional)

If you use LLM-based explanation features in llm_explain.py, create a .env file:

OPENAI_API_KEY=your_key_here


This file is ignored by Git and remains local.

6. Running the Training Script

Run LSTM training:

python train_lstm.py


The script will:

Load raw OHLC data

Engineer features

Build sequential windows

Split into train/validation/test

Normalize data

Train the LSTM model

Evaluate test accuracy

Save the best model under models/

7. Results Summary

Observed performance (typical output):

Validation Accuracy: ~50.3%

Test Accuracy: ~49.7%

Behavior: No statistically significant edge over randomness

This aligns with established findings:
short-horizon (1-minute) direction prediction is generally not predictable using price-based features alone.

8. Interpretation and Practical Insights

1-minute candles are dominated by noise and microstructure effects

LSTMs cannot extract stable directional signals from such data

The exercise is valuable academically because it demonstrates:

Limitations of deep learning in high-frequency finance

Impact of noise and volatility on predictive modeling

Importance of feature quality and data granularity

This project is therefore best positioned as an exploratory analysis and ML workflow demonstration.

9. Possible Extensions

Future directions that may improve predictive performance:

Transformer or Temporal Convolution (TCN) architectures

Market microstructure features (order imbalance, depth, liquidity)

Regime detection (volatility clustering, trend phases)

Multi-horizon prediction instead of next-minute direction

Ensemble or hybrid models

Probabilistic forecasting instead of binary classification
