# AFBRL

It compares deep learning models trained with multiple loss functions (including AFBRL, its original formulation without the recovery term, aswell as qlike and mse) along with classical econometric benchmarks (GARCH family + HAR). 


## What This Project Does

- Cleans and transforms realised volatility data.
- Builds econometric benchmark forecasts (GARCH, EGARCH, GJR-GARCH, HAR-RV).
- Trains LSTM and GRU models using expanding window.
- Compute RMSE, MAE, QLIKE, underprediction rate and Diebold Mariano tests.

## Repository Layout

- `Data/`
  - `oxfordmanrealizedvolatilityindices.csv`: bundled raw Oxford-Man realised volatility input file.
  - `data_cleaner.py`: raw data cleaning and feature creation.
  - `cleaned_volatility_data.csv`: cleaned dataset used by training and benchmarks.
  - `garch_forecasts.csv`: econometric forecast output.
- `src/`
  - `dataset.py`: sequence dataset construction and normalization.
  - `models.py`: LSTM and GRU forecasting models.
  - `losses.py`: QLIKE and original AFBRL loss implementations.
  - `afbrl.py`: improved AFBRL loss with recovery term.
  - `generate_benchmarks.py`: fits econometric models.
  - `train_expanding_window.py`: deep learning training and prediction pipeline.
  - `plots/plot.py`, `plots/plot_2011.py`: plotting scripts.
- `evaluation/`
  - `evaluate.py`: deep learning model evaluation.
  - `evaluate_econometric.py`: econometric benchmark evaluation.
- `results/`
  - `raw/`: raw deep learning predictions.
  - `deep_learning/`: deep learning summary and DM test tables.
  - `econometric/`: econometric summary and DM test tables.
  - `plots/`: generated figures.

## Python and Dependencies

Recommended Python: 3.10+

Install dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib torch arch
```


## Run Order

Important: some scripts use relative paths that depend on where you run them from. Use the commands exactly as shown.

1) Clean raw data

The project already includes the raw Oxford-Man CSV at `Data/oxfordmanrealizedvolatilityindices.csv`.
Run the cleaner to generate `Data/cleaned_volatility_data.csv`:

```bash
cd Data
python data_cleaner.py
cd ..
```

2) Generate econometric benchmark forecasts

```bash
cd src
python generate_benchmarks.py
cd ..
```

3) Train deep learning models with expanding window forecasting

Run from repository root:

```bash
python src/train_expanding_window.py
```

4) Evaluate deep learning outputs

Run from repository root:

```bash
python evaluation/evaluate.py
```

5) Evaluate econometric outputs

```bash
cd evaluation
python evaluate_econometric.py
cd ..
```

6) Generate plots

```bash
cd src/plots
python plot.py
python plot_2011.py
cd ../..
```

## Main Modeling Setup

- Architectures: LSTM and GRU.
- Input features per timestep:
  - log returns
  - HAR daily lag
  - HAR weekly lag
  - HAR monthly lag
- Training target: `log_rv5_ss`.
- Evaluation target: `log_rk_parzen`.
- Losses used:
  - MSE
  - QLIKE
  - AFBRL-Original
  - AFBRL

## Reproducibility Notes

- Training uses CPU if CUDA is unavailable.
- Econometric fitting can take time across many symbols.
- Relative path assumptions are currently script-specific; run locations above avoid path errors.