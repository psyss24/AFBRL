import pandas as pd
import numpy as np
import scipy.stats as stats
import os


GARCH_FORECASTS_FILE = '../Data/garch_forecasts.csv'
CLEANED_DATA_FILE = '../Data/cleaned_volatility_data.csv'
OUTPUT_DIR = '../results/econometric'

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

def diebold_mariano_test(y_true, y_pred_a, y_pred_b, metric='MSE'):
    e_a = y_true - y_pred_a
    e_b = y_true - y_pred_b
    
    if metric == 'MSE':
        d = e_a**2 - e_b**2
    elif metric == 'MAE':
        d = np.abs(e_a) - np.abs(e_b)
    elif metric == 'QLIKE':
        var_true = y_true**2
        var_a = y_pred_a**2 + 1e-8
        var_b = y_pred_b**2 + 1e-8
        loss_a = (var_true/var_a) - np.log(var_true/var_a) - 1
        loss_b = (var_true/var_b) - np.log(var_true/var_b) - 1
        d = loss_a - loss_b

    mean_d = np.mean(d)
    gamma_0 = np.var(d)
    if gamma_0 == 0: 
        return 0.0, 1.0
    
    dm_stat = mean_d / np.sqrt((gamma_0 / len(d)))
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value

def calculate_metrics(df):
    real = df['Actual_RK'].values
    pred = df['Predicted_Vol'].values
    
    mse = np.mean((real - pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(real - pred))

    ratio = (real**2) / (pred**2 + 1e-8)
    qlike = np.mean(ratio - np.log(ratio) - 1)

    under_rate = np.mean((pred - real) < 0) * 100
    
    return pd.Series({
        'RMSE': rmse,
        'MAE': mae,
        'QLIKE': qlike,
        'UnderPred_Rate': under_rate,
        'Count': len(df)
    })

def main():
    
    try:
        garch_df = pd.read_csv(GARCH_FORECASTS_FILE)
        clean_df = pd.read_csv(CLEANED_DATA_FILE)
    except FileNotFoundError as e:
        print(f"could not find file: {e}")
        return
    
    garch_df['Date'] = pd.to_datetime(garch_df['Date'])
    clean_df['Date'] = pd.to_datetime(clean_df['Date'])
    
    merged = garch_df.merge(
        clean_df[['Date', 'Symbol', 'rk_parzen']], 
        on=['Date', 'Symbol'], 
        how='inner'
    )
    
    merged['Actual_RK'] = np.sqrt(merged['rk_parzen'])
    
    models_to_eval = ['GARCH_vol', 'EGARCH_vol', 'GJR_vol', 'HAR_vol']
    
    dfs_list = []
    for model_col in models_to_eval:
        if model_col not in merged.columns:
            continue
            
        temp = merged[['Date', 'Symbol', 'Actual_RK', model_col]].copy()
        temp = temp.dropna(subset=[model_col])
        temp['Model'] = model_col.replace('_vol', '').upper()
        temp['Predicted_Vol'] = temp[model_col]
        temp = temp[['Date', 'Symbol', 'Model', 'Actual_RK', 'Predicted_Vol']]
        dfs_list.append(temp)
    
    if not dfs_list:
        print("no valid forecasts")
        return
    
    eval_df = pd.concat(dfs_list, ignore_index=True)
    
    print("\neconometric model performance")
    summary = eval_df.groupby(['Model']).apply(calculate_metrics, include_groups=False).reset_index()
    print(summary[['Model', 'RMSE', 'MAE', 'QLIKE', 'UnderPred_Rate']].sort_values('QLIKE'))
    summary.to_csv(f"{OUTPUT_DIR}/summary_metrics.csv", index=False)
    
    print("\ndiebold mariano tests")
    df_pivot = eval_df.pivot_table(index=['Symbol', 'Date'], columns='Model', values='Predicted_Vol')
    df_actual = eval_df.pivot_table(index=['Symbol', 'Date'], columns='Model', values='Actual_RK').iloc[:, 0]
    
    dm_results = []
    unique_models = df_pivot.columns.tolist()
    benchmark_name = 'GARCH'
    
    if benchmark_name in unique_models:
        y_true_all = df_actual.values
        y_bench_all = df_pivot[benchmark_name].values
        
        for model in unique_models:
            if model == benchmark_name:
                continue
            
            y_curr_all = df_pivot[model].values
            
            mask = ~np.isnan(y_bench_all) & ~np.isnan(y_curr_all) & ~np.isnan(y_true_all)
            if np.sum(mask) == 0: 
                continue
            
            y_t = y_true_all[mask]
            y_b = y_bench_all[mask]
            y_c = y_curr_all[mask]
            
            dm_mse, p_mse = diebold_mariano_test(y_t, y_b, y_c, metric='MSE')
            dm_qlike, p_qlike = diebold_mariano_test(y_t, y_b, y_c, metric='QLIKE')
            
            dm_results.append({
                'Comparison': f"{benchmark_name} vs {model}",
                'Metric': 'MSE',
                'DM_Stat': dm_mse,
                'P_Value': p_mse,
                'Significant': 'YES' if p_mse < 0.05 else 'NO'
            })
            
            dm_results.append({
                'Comparison': f"{benchmark_name} vs {model}",
                'Metric': 'QLIKE',
                'DM_Stat': dm_qlike,
                'P_Value': p_qlike,
                'Significant': 'YES' if p_qlike < 0.05 else 'NO'
            })
    
    dm_df = pd.DataFrame(dm_results)
    if not dm_df.empty:
        print("\n", dm_df)
        dm_df.to_csv(f"{OUTPUT_DIR}/dm_test_results.csv", index=False)
    
    by_symbol = eval_df.groupby(['Symbol', 'Model']).apply(calculate_metrics, include_groups=False).reset_index()
    by_symbol.to_csv(f"{OUTPUT_DIR}/detailed_metrics_by_symbol.csv", index=False)
    
    print(f"\ntables saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
