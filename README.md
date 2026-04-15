# AFBRL

It compares deep learning models trained with multiple loss functions (including AFBRL, its original formulation without the recovery term, aswell as qlike and mse) along with classical econometric benchmarks (GARCH family + HAR). 


## What This Project Does

- Cleans and transforms realised volatility data.
- Builds econometric benchmark forecasts (GARCH, EGARCH, GJR-GARCH, HAR-RV).
- Trains LSTM and GRU models using expanding window.
- Compute RMSE, MAE, QLIKE, underprediction rate and Diebold Mariano tests.
## The AFBRL Derivation Overview

Standard symmetric loss functions like Mean Squared Error (MSE) produce unbounded quadratic penalty growth during heavy-tailed structural breaks (e.g., market crashes), which can destabilize recurrent weight matrices. Conversely, purely bounded loss functions suffer from gradient saturation (vanishing gradients) during these same extreme events. 

The **Asymmetric Fuzzy Bounded Recovery Loss (AFBRL)** solves this through three core mechanisms:

**1. Bounded Relative Error ($e_t$)** Forecast errors are scaled relative to an econometric benchmark (we use GARCH(1,1)) and mapped to a strict [0, 1] interval. This prevents the initial error magnitude from exploding:

$$
e_t = \frac{|y_t - \hat{y}_t|}{|y_t - \hat{y}_t| + |y_t - y_t^*|}
$$

**2. Asymmetric Fuzzy Gaussian Kernel** ($\mu$($e_t$)) The crisp relative error is mapped to a membership degree using a Gaussian kernel. To penalise the dangerous underprediction of volatility (risk underestimation), the Gaussian width ($\sigma$) is adaptive: it uses a narrower $\sigma_{strict}$ for underpredictions and a wider $\sigma_{loose}$ for overpredictions:

$$
\mu(e_t) = \exp\left(-\frac{e_t^2}{2\sigma_t^2}\right)
$$

$$
\sigma_t = \begin{cases} \sigma_{strict} & \text{if } \hat{y}_t < y_t \\ \sigma_{loose} & \text{if } \hat{y}_t \ge y_t \end{cases}
$$

**3. Dynamically Gated Linear Recovery ($\mathcal{R}_t$)** To solve the vanishing gradient problem during massive market crashes, AFBRL introduces a linear recovery term. As the fuzzy membership approaches zero during a catastrophic tail event, a smooth activation gate opens, applying a constant linear gradient. 

$$
\mathcal{R}_t = \lambda \cdot (1 - \mu(e_t)) \cdot |y_t - \hat{y}_t|
$$

**Final Objective Function** Combining the bounded fuzzy penalty with the gated linear recovery, the final objective function minimised during training is:

$$
L_{AFBRL} = \frac{1}{N} \sum_{t=1}^N \left[ \left(1 - \exp\left(-\frac{e_t^2}{2\sigma_t^2}\right)\right) + \lambda \cdot \left(1 - \exp\left(-\frac{e_t^2}{2\sigma_t^2}\right)\right) \cdot |y_t - \hat{y}_t| \right]
$$

## Repository Layout

- `Data/`
  - `oxfordmanrealizedvolatilityindices.csv`: bundled raw Oxford-Man realised volatility input file.
  - `data_cleaner.py`: raw data cleaning and feature creation.
  - `cleaned_volatility_data.csv`: cleaned dataset used by training and benchmarks.
  - `garch_forecasts.csv`: econometric forecast output.
- `src/`
  - `dataset.py`: sequence dataset construction and normalisation.
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
