import torch
import torch.nn as nn


class ForecastingLSTM(nn.Module):
    # lstm for volatility forecasting
    # inputs: log-returns + 3 har lags (daily, weekly, monthly) = 4 features
    # outputs: scalar logvariance prediction
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, dropout=0.2):
        super(ForecastingLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # batch_first=True, input shape is (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # take last timestep -> scalar
        return self.fc(out[:, -1, :]).squeeze(-1)  


class ForecastingGRU(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, dropout=0.2):
        super(ForecastingGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :]).squeeze(-1)