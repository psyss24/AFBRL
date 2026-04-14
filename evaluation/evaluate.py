import pandas as pd
import numpy as np
import scipy.stats as stats
import os


RESULTS_FILE = 'results/raw/dl_predictions_expanding_window.csv'
OUTPUT_DIR = 'results/deep_learning'

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def diebold_mariano_test(y_true, y_pred_a, y_pred_b, metric='MSE'):
    # diebold mariano test: negative stat = model a (benchmark) better, positive = model b better
    e_a = y_true - y_pred_a
    e_b = y_true - y_pred_b
    
    if metric == 'MSE':
        d = e_a**2 - e_b**2
    elif metric == 'MAE':
        d = np.abs(e_a) - np.abs(e_b)
    elif metric == 'QLIKE':
        # qlike: y/h - log(y/h) - 1
        var_true = y_true**2
        var_a = y_pred_a**2 + 1e-8
        var_b = y_pred_b**2 + 1e-8
        loss_a = (var_true/var_a) - np.log(var_true/var_a) - 1
        loss_b = (var_true/var_b) - np.log(var_true/var_b) - 1
        d = loss_a - loss_b

    # dm stat
    mean_d = np.mean(d)
    gamma_0 = np.var(d)
    if gamma_0 == 0: return 0.0, 1.0
    
    dm_stat = mean_d / np.sqrt((gamma_0 / len(d)))
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value

def calculate_metrics(df):
    real = df['Actual_RK'].values
    pred = df['Predicted_Vol'].values
    
    mse = np.mean((real - pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(real - pred))

    # qlike
    ratio = (real**2) / (pred**2 + 1e-8)
    qlike = np.mean(ratio - np.log(ratio) - 1)

    # underprediction rate
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
        df = pd.read_csv(RESULTS_FILE)
    except FileNotFoundError:
        print(f"could not find {RESULTS_FILE}. run training first.")
        return

    # convert logvol to linear
    df['Predicted_Vol'] = np.sqrt(np.exp(df['Predicted_Log_Vol']))
    df['Actual_RK'] = np.sqrt(np.exp(df['Actual_Log_RK']))
    
    # aggregate summary
    summary = df.groupby(['Model']).apply(calculate_metrics, include_groups=False).reset_index()
    print("\n performance summary")
    print(summary[['Model', 'RMSE', 'QLIKE', 'UnderPred_Rate']].sort_values('Model'))
    summary.to_csv(f"{OUTPUT_DIR}/summary_metrics.csv", index=False)
    
    # diebold mariano tests
    print("\n dm testing")

    # pivot
    df_pivot = df.pivot_table(index=['Symbol', 'Date'], columns='Model', values='Predicted_Vol')
    df_actual = df.pivot_table(index=['Symbol', 'Date'], columns='Model', values='Actual_RK').iloc[:, 0]

    dm_results = []
    unique_models = df_pivot.columns.tolist()

    # identify architectures from model names
    architectures = set([m.split('-')[0] for m in unique_models])
    
    for arch in architectures:
        benchmark_name = f"{arch}-MSE"
        
        if benchmark_name not in unique_models:
            print(f"benchmark {benchmark_name} not found. skipping {arch} tests.")
            continue


        
        y_true_all = df_actual.values
        y_bench_all = df_pivot[benchmark_name].values
        
        for model in unique_models:
            # only compare within same architecture
            if not model.startswith(arch) or model == benchmark_name:
                continue

            y_curr_all = df_pivot[model].values

            # mask nans
            mask = ~np.isnan(y_bench_all) & ~np.isnan(y_curr_all) & ~np.isnan(y_true_all)
            if np.sum(mask) == 0: continue
            
            y_t = y_true_all[mask]
            y_b = y_bench_all[mask]
            y_c = y_curr_all[mask]
            
            # mse
            dm_mse, p_mse = diebold_mariano_test(y_t, y_b, y_c, metric='MSE')

            # qlike
            dm_qlike, p_qlike = diebold_mariano_test(y_t, y_b, y_c, metric='QLIKE')
            
            dm_results.append({
                'Architecture': arch,
                'Comparison': f"{benchmark_name} vs {model}",
                'Metric': 'MSE',
                'DM_Stat': dm_mse,
                'P_Value': p_mse,
                'Significant': 'YES' if p_mse < 0.05 else 'NO'
            })
            
            dm_results.append({
                'Architecture': arch,
                'Comparison': f"{benchmark_name} vs {model}",
                'Metric': 'QLIKE',
                'DM_Stat': dm_qlike,
                'P_Value': p_qlike,
                'Significant': 'YES' if p_qlike < 0.05 else 'NO'
            })
            
    dm_df = pd.DataFrame(dm_results)
    print("\n", dm_df)
    dm_df.to_csv(f"{OUTPUT_DIR}/dm_test_results.csv", index=False)
    
    # detailed symbol breakdown
    by_symbol = df.groupby(['Symbol', 'Model']).apply(calculate_metrics, include_groups=False).reset_index()
    by_symbol.to_csv(f"{OUTPUT_DIR}/detailed_metrics_by_symbol.csv", index=False)

    print(f"\n tables saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()