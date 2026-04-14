import pandas as pd
import numpy as np
import os

def clean_data(input_file, output_file):
    print(f"reading {input_file}...")

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"file not found: {input_file}")
        return

    # fix timestamps and sort
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df.sort_values(by=['Symbol', 'Date'], inplace=True)

    # log returns: r_t = ln(P_t) - ln(P_{t-1}), required for garch
    df['log_returns'] = df.groupby('Symbol')['close_price'].transform(
        lambda x: np.log(x) - np.log(x.shift(1))
    )

    # log-transform rv5_ss and rk_parzen; drop non-positive values to avoid -inf
    initial_count = len(df)
    valid_vol_mask = (df['rv5_ss'] > 0) & (df['rk_parzen'] > 0)
    df = df[valid_vol_mask].copy()

    df['log_rv5_ss'] = np.log(df['rv5_ss'])
    df['log_rk_parzen'] = np.log(df['rk_parzen'])

    # drop nans and infs (first day return is nan, occasional inf from bad prices)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['log_returns', 'log_rv5_ss', 'log_rk_parzen'], inplace=True)

    dropped_count = initial_count - len(df)
    if dropped_count > 0:
        print(f"dropped {dropped_count} rows (nans, infs, or zero vol)")

    # select output columns
    output_columns = [
        'Date', 'Symbol', 
        'close_price', 
        'rv5_ss', 'rk_parzen',       
        'log_returns',               
        'log_rv5_ss', 'log_rk_parzen' 
    ]
    final_df = df[output_columns]

    final_df.to_csv(output_file, index=False)
    print(f"saved to {output_file} — {len(final_df)} rows, {final_df['Symbol'].nunique()} symbols.")

if __name__ == "__main__":
    clean_data('oxfordmanrealizedvolatilityindices.csv', 'cleaned_volatility_data.csv')