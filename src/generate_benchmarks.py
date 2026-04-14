import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.linear_model import LinearRegression
import os
import warnings

DATA_FILE = '../Data/cleaned_volatility_data.csv'
OUTPUT_FILE = '../Data/garch_forecasts.csv'

warnings.simplefilter('ignore')

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Missing {DATA_FILE}")
    
    df = pd.read_csv(DATA_FILE)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df = df.sort_values(['Symbol', 'Date'])

    # calc log returns if missing (required for garch)
    if 'log_return' not in df.columns and 'close_price' in df.columns:
        df['log_return'] = df.groupby('Symbol')['close_price'].apply(
            lambda x: np.log(x) - np.log(x.shift(1))
        ).reset_index(level=0, drop=True)

    df['log_return'] = df['log_return'].fillna(0)  # first row per symbol is nan
    return df

def fit_har_model(sub_df, target_col='rv5_ss'):
    # har-rv: RV_next ~ c + b_d*RV_d + b_w*RV_w + b_m*RV_m
    # returns predicted volatility (sqrt of variance)

    # fallback to any rv/vol column if target missing
    if target_col not in sub_df.columns:
        candidates = [c for c in sub_df.columns if 'rv' in c or 'vol' in c]
        target_col = candidates[0] if candidates else None
        if not target_col:
            return pd.Series(np.nan, index=sub_df.index)

    # build daily, weekly, monthly rv lags
    df_har = sub_df[[target_col]].copy()
    df_har['RV_d'] = df_har[target_col].shift(1)
    df_har['RV_w'] = df_har[target_col].rolling(window=5).mean().shift(1)
    df_har['RV_m'] = df_har[target_col].rolling(window=22).mean().shift(1)

    valid_data = df_har.dropna()
    if len(valid_data) < 50:
        return pd.Series(np.nan, index=sub_df.index)

    X = valid_data[['RV_d', 'RV_w', 'RV_m']]
    y = valid_data[target_col]

    try:
        reg = LinearRegression()
        reg.fit(X, y)
        preds_variance = reg.predict(X)

        # clip negatives then sqrt to get volatility
        preds_variance = np.maximum(preds_variance, 0)
        preds_volatility = np.sqrt(preds_variance)

        # align predictions back to original index
        full_series = pd.Series(np.nan, index=sub_df.index)
        full_series.loc[valid_data.index] = preds_volatility
        return full_series

    except Exception:
        return pd.Series(np.nan, index=sub_df.index)

def generate_benchmarks():
    df = load_data()
    all_results = []
    all_symbols = df['Symbol'].unique()

    print(f"fitting benchmarks for {len(all_symbols)} symbols...")

    for i, symbol in enumerate(all_symbols):
        print(f"[{i+1}/{len(all_symbols)}] {symbol}...", end=" ", flush=True)

        sub_df = df[df['Symbol'] == symbol].copy()

        try:
            # scale returns *100 for garch numerical stability
            returns = sub_df['log_return'] * 100

            # fit garch family, conditional_volatility is sigma, divide back by 100
            res_garch = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal').fit(disp='off')
            sub_df['GARCH_vol'] = res_garch.conditional_volatility / 100.0

            res_egarch = arch_model(returns, vol='EGARCH', p=1, q=1, dist='Normal').fit(disp='off')
            sub_df['EGARCH_vol'] = res_egarch.conditional_volatility / 100.0

            res_gjr = arch_model(returns, vol='Garch', p=1, o=1, q=1, dist='Normal').fit(disp='off')
            sub_df['GJR_vol'] = res_gjr.conditional_volatility / 100.0

            sub_df['HAR_vol'] = fit_har_model(sub_df, target_col='rv5_ss')

            output_cols = ['Date', 'Symbol', 'GARCH_vol', 'EGARCH_vol', 'GJR_vol', 'HAR_vol']
            all_results.append(sub_df[output_cols])
            print("done.")

        except Exception as e:
            print(f"failed ({e})")
            # pad with nans so downstream merges don't drop this symbol
            for c in ['GARCH_vol', 'EGARCH_vol', 'GJR_vol', 'HAR_vol']:
                sub_df[c] = np.nan
            all_results.append(sub_df[['Date', 'Symbol', 'GARCH_vol', 'EGARCH_vol', 'GJR_vol', 'HAR_vol']])

    pd.concat(all_results).to_csv(OUTPUT_FILE, index=False)
    print(f"\nsaved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_benchmarks()