import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def main():
    df = pd.read_csv('../../results/raw/dl_predictions_expanding_window.csv')
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    
    spx = df[df['Symbol'] == '.SPX'].copy()
    period = spx[(spx['Date'] >= '2011-07-01') & (spx['Date'] <= '2011-11-30')].copy()
    
    period['Actual_Vol'] = np.sqrt(np.exp(period['Actual_Log_RK'])) * np.sqrt(252)
    
    actuals = period[period['Model'] == 'LSTM-MSE'][['Date', 'Actual_Vol']].drop_duplicates()
    lstm_mse = period[period['Model'] == 'LSTM-MSE'][['Date', 'Predicted_Log_Vol']].copy()
    lstm_afbrl = period[period['Model'] == 'LSTM-AFBRL'][['Date', 'Predicted_Log_Vol']].copy()
    
    lstm_mse['Predicted_Vol'] = np.sqrt(np.exp(lstm_mse['Predicted_Log_Vol'])) * np.sqrt(252)
    lstm_afbrl['Predicted_Vol'] = np.sqrt(np.exp(lstm_afbrl['Predicted_Log_Vol'])) * np.sqrt(252)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    ax.plot(actuals['Date'], actuals['Actual_Vol'], color='black', linewidth=2, label='Realised Volatility')
    ax.plot(lstm_afbrl['Date'], lstm_afbrl['Predicted_Vol'], color='#1f77b4', linewidth=1.8, label='AFBRL')
    ax.plot(lstm_mse['Date'], lstm_mse['Predicted_Vol'], color='#ff7f0e', linewidth=1.8, label='MSE', linestyle='--')
    
    ax.set_xlim(actuals['Date'].min(), actuals['Date'].max())
    ax.set_ylabel('Annualised Volatility', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    ax.legend(loc='upper left', fontsize=11, frameon=False)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../results/plots/2011_forecast.png', bbox_inches='tight', dpi=300)
    print("saved to results/plots/2011_forecast.png")

if __name__ == "__main__":
    main()
