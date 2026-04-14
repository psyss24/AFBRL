import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os

from dataset import VolatilityDataset
from models import ForecastingLSTM, ForecastingGRU
from losses import QLIKELoss, OriginalAFBRLLoss
from afbrl import AFBRLLoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EPOCHS = 30
WINDOW_SIZE = 22
SAVE_DIR = 'results/raw'

# expanding window config: rougly 2yr initial train, retrain + predict monthly
INITIAL_TRAIN_SIZE = 500
TEST_WINDOW_SIZE = 22
RETRAIN_FREQUENCY = 22


def get_all_symbols(filepath='Data/cleaned_volatility_data.csv'):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return []
    df = pd.read_csv(filepath)
    symbols = sorted(df['Symbol'].unique().tolist())
    return symbols


def train_and_predict_expanding_window():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    all_symbols = get_all_symbols()
    if not all_symbols:
        print("no symbols found. check data path.")
        return

    architectures = [
        ('LSTM', ForecastingLSTM),
        ('GRU', ForecastingGRU)
    ]

    loss_functions = [
        ('MSE', nn.MSELoss()),
        ('QLIKE', QLIKELoss().to(DEVICE)),
        ('AFBRL-Original', OriginalAFBRLLoss(strict_sigma=0.15, loose_sigma=0.3).to(DEVICE)),
        ('AFBRL', AFBRLLoss(strict_sigma=0.15, loose_sigma=0.3, recovery_weight=1.0).to(DEVICE))
    ]

    all_predictions = []

    for symbol in all_symbols:
        print(f"\n{'='*60}\nPROCESSING {symbol}\n{'='*60}")

        try:
            # load full series just to get total length
            temp_ds = VolatilityDataset(symbol=symbol, window_size=WINDOW_SIZE, mode='train')
            total_len = len(temp_ds.data)
        except Exception as e:
            print(f"Skipping {symbol}: {e}")
            continue

        window_count = 0
        for train_end in range(INITIAL_TRAIN_SIZE, total_len, RETRAIN_FREQUENCY):
            test_start = train_end
            test_end = min(train_end + TEST_WINDOW_SIZE, total_len)

            if test_end - test_start < TEST_WINDOW_SIZE: break

            print(f"  Window {window_count}: Train[0:{train_end}] -> Test[{test_start}:{test_end}]")

            # data prep 
            try:
                train_ds = VolatilityDataset(
                    symbol=symbol, window_size=WINDOW_SIZE, mode='train',
                    train_start_idx=0, train_end_idx=train_end
                )
                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

                # capture norm stats from train window to apply to test
                train_mean = train_ds.mean
                train_std = train_ds.std

                test_ds = VolatilityDataset(
                    symbol=symbol, window_size=WINDOW_SIZE, mode='test',
                    test_start_idx=test_start, test_end_idx=test_end,
                    norm_mean=train_mean, norm_std=train_std
                )
                test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

            except Exception as e:
                print(f"    data error: {e}")
                continue

            # train 
            for arch_name, ModelClass in architectures:
                for loss_name, criterion in loss_functions:

                    model = ModelClass(input_dim=4, hidden_dim=64, num_layers=2, dropout=0.2).to(DEVICE)
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    model.train()

                    for epoch in range(EPOCHS):
                        for x, y_train, _, y_bench, _ in train_loader:
                            x, y_train, y_bench = x.to(DEVICE), y_train.to(DEVICE), y_bench.to(DEVICE)

                            optimizer.zero_grad()
                            y_pred = model(x)

                            # afbrl variants need the garch benchmark as 3rd arg
                            if 'AFBRL' in loss_name:
                                loss = criterion(y_pred, y_train, y_bench)
                            else:
                                loss = criterion(y_pred, y_train)

                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()

                    # inference 
                    model.eval()
                    with torch.no_grad():
                        for x_test, _, y_eval, _, date in test_loader:
                            x_test = x_test.to(DEVICE)

                            # outputs and targets are in log-variance space
                            pred_log = model(x_test).item()
                            actual_log_rk = y_eval.item()

                            all_predictions.append({
                                'Symbol': symbol,
                                'Window': window_count,
                                'Model': f"{arch_name}-{loss_name}",
                                'Date': date.item(),
                                'Actual_Log_RK': actual_log_rk,
                                'Predicted_Log_Vol': pred_log,
                                'Predicted_Vol': np.sqrt(np.exp(pred_log)),   # convert to linear vol
                                'Actual_RK': np.sqrt(np.exp(actual_log_rk))  # convert to linear vol
                            })

                    del model

            window_count += 1

    pd.DataFrame(all_predictions).to_csv(f"{SAVE_DIR}/dl_predictions_expanding_window.csv", index=False)
    print("DONE.")


if __name__ == "__main__":
    train_and_predict_expanding_window()