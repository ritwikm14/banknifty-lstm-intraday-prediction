# src/lstm_model.py

import torch
import torch.nn as nn


class BankNiftyLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, 1) probability in [0,1]
        """
        output, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]          # (batch, hidden_dim)
        logits = self.fc(last_hidden)  # (batch, 1)
        prob = self.sigmoid(logits)    # (batch, 1)
        return prob


class LSTMClassifier(BankNiftyLSTM):
    """
    Thin alias so serving / training code can refer to `LSTMClassifier`
    while internally using the same architecture as BankNiftyLSTM.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__(
            input_dim=input_size,
            hidden_dim=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
