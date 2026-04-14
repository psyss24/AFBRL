import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class VolatilityDataset(Dataset):
    def __init__(self, 
                 data_file='Data/cleaned_volatility_data.csv', 
                 benchmark_file='Data/garch_forecasts.csv', 
                 symbol=None, 
                 window_size=22, 
                 mode='train', 
                 train_start_idx=None,
                 train_end_idx=None,
                 test_start_idx=None,
                 test_end_idx=None,
                 norm_mean=None,
                 norm_std=None):
        
        self.window_size = window_size
        self.mode = mode
        
        df_clean = pd.read_csv(data_file)
        df_garch = pd.read_csv(benchmark_file)
        
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], utc=True)
        df_garch['Date'] = pd.to_datetime(df_garch['Date'], utc=True)
        
        if symbol:
            df_clean = df_clean[df_clean['Symbol'] == symbol]
            df_garch = df_garch[df_garch['Symbol'] == symbol]
            
        merged = pd.merge(df_clean, df_garch, on=['Date', 'Symbol'], how='inner')
        merged.sort_values('Date', inplace=True)
        
        # conv garch to logvariance 
        if 'GARCH_vol' in merged.columns:
            merged['log_garch_variance'] = np.log(merged['GARCH_vol'] ** 2)
        
        # targets
        self.target_col = 'log_rv5_ss'            
        self.eval_col = 'log_rk_parzen'
        self.benchmark_col = 'log_garch_variance'
        
        # feature engineering
        target_series = merged[self.target_col]
        merged['har_d'] = target_series.shift(1)
        merged['har_w'] = target_series.shift(1).rolling(window=5).mean()
        merged['har_m'] = target_series.shift(1).rolling(window=22).mean()
        merged.dropna(inplace=True)
        
        feature_cols = ['log_returns', 'har_d', 'har_w', 'har_m']
        
        if mode == 'train':
            if train_start_idx is None: train_start_idx = 0
            if train_end_idx is None: train_end_idx = len(merged)

            self.data = merged.iloc[train_start_idx:train_end_idx].copy()

            features = self.data[feature_cols].values
            self.mean = np.mean(features, axis=0)
            self.std = np.std(features, axis=0) + 1e-8

        else: 
            if test_start_idx is None or test_end_idx is None:
                raise ValueError("Test mode requires test_start_idx and test_end_idx")
            if norm_mean is None or norm_std is None:
                raise ValueError("Test mode requires norm_mean and norm_std from training window")

            # incl lookback buffer so the first prediction has a full window of history
            buffer_start = max(0, test_start_idx - window_size)
            self.data = merged.iloc[buffer_start:test_end_idx].copy()

            self.mean = norm_mean
            self.std = norm_std
            features = self.data[feature_cols].values

        # normalise
        features_norm = (features - self.mean) / self.std
        self.X = torch.tensor(features_norm, dtype=torch.float32)
        self.y_train = torch.tensor(self.data[self.target_col].values, dtype=torch.float32)
        self.y_eval = torch.tensor(self.data[self.eval_col].values, dtype=torch.float32)
        self.y_bench = torch.tensor(self.data[self.benchmark_col].values, dtype=torch.float32)
        self.dates = pd.to_datetime(self.data['Date']).astype('int64').values

    def __len__(self):
        # length is features minus the first window
        return len(self.X) - self.window_size

    def __getitem__(self, idx):
        # x_window uses the previous window_size steps
        x_window = self.X[idx : idx + self.window_size]
        
        # targets at end of window
        y_train = self.y_train[idx + self.window_size]
        y_eval = self.y_eval[idx + self.window_size]
        y_bench = self.y_bench[idx + self.window_size]
        date = self.dates[idx + self.window_size]
        
        return x_window, y_train, y_eval, y_bench, date