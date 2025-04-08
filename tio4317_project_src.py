import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import os

from arch import arch_model
from sklearn.model_selection import train_test_split
from scipy.stats import chi2, jarque_bera
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import warnings
from joblib import Parallel, delayed

tickers = ["XOM", "META", "PFE", "WMT", "TSLA", "CVX", "C", "BA", "CTRA", "JNJ"]

data_dir = "data/"
rolling_window = 21
smoothing_window = 10
train_window = 252 
forecast_horizon = 21
n_jobs = 10 

models_to_run = [
    {'name': 'GARCH(1,1)', 'vol': 'GARCH', 'p': 1, 'q': 1},
    {'name': 'GJR-GARCH(1,1)', 'vol': 'GARCH', 'p': 1, 'o': 1, 'q': 1},
    {'name': 'ARCH(1)', 'vol': 'ARCH', 'p': 1, 'q': 0}
]

all_stock_results = {}
all_stock_data = {}

def run_rolling_forecast_for_model(ticker, model_spec, df, train_window, forecast_horizon):
    """Run a rolling forecast for a given model."""
    import warnings
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*Dtype inference on a pandas object.*"
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning, 
        message=".*ConvergenceWarning.*"
    )

    model_name = model_spec['name']

    q_low = df['q_low'].iloc[0]
    q_high = df['q_high'].iloc[0]

    if len(df) < train_window + forecast_horizon:
        return ticker, model_name, None 

    forecasts, actuals, regimes, dates = [], [], [], []

    for t in range(train_window, len(df) - forecast_horizon):
        train_data = df.iloc[t-train_window : t]['e']

        if not np.isfinite(train_data).all() or train_data.empty:
            continue

        model = arch_model(
            train_data, mean='Zero',
            vol=model_spec['vol'],
            p=model_spec['p'],
            q=model_spec['q'],
            o=model_spec.get('o', 0)
        )

        try:
            res = model.fit(disp='off', show_warning=False) 
            forecasts_horizon = res.forecast(horizon=forecast_horizon, reindex=False)
            f_t = forecasts_horizon.variance.iloc[0].mean()
            actual = df['unsmoothed_rolling_vol'].iloc[t + forecast_horizon - 1]
            regime_vol = df['Rolling volatility'].iloc[t + forecast_horizon - 1]

            if np.isfinite(f_t) and np.isfinite(actual) and np.isfinite(regime_vol):
                forecasts.append(f_t)
                actuals.append(actual)
                regimes.append(regime_vol)
                dates.append(df.index[t + forecast_horizon - 1])
        except Exception as e:
            continue

    if not dates:
         results = {
             'rmse_overall': np.nan, 'rmse_low': np.nan, 'rmse_med': np.nan,
             'rmse_high': np.nan, 'forecast_df': pd.DataFrame()
         }
         return ticker, model_name, results 


    forecast_df = pd.DataFrame({'forecast': forecasts, 'actual': actuals, 'regime_vol': regimes}, index=dates)

    rmse_overall = np.sqrt(np.mean((forecast_df['forecast'] - forecast_df['actual'])**2))
    low_vol_df = forecast_df[forecast_df['regime_vol'] < q_low]
    rmse_low = np.sqrt(np.mean((low_vol_df['forecast'] - low_vol_df['actual'])**2)) if not low_vol_df.empty else np.nan
    high_vol_df = forecast_df[forecast_df['regime_vol'] > q_high]
    rmse_high = np.sqrt(np.mean((high_vol_df['forecast'] - high_vol_df['actual'])**2)) if not high_vol_df.empty else np.nan
    med_vol_df = forecast_df[(forecast_df['regime_vol'] >= q_low) & (forecast_df['regime_vol'] <= q_high)]
    rmse_med = np.sqrt(np.mean((med_vol_df['forecast'] - med_vol_df['actual'])**2)) if not med_vol_df.empty else np.nan

    results = {
        'rmse_overall': rmse_overall, 'rmse_low': rmse_low,
        'rmse_med': rmse_med, 'rmse_high': rmse_high,
        'forecast_df': forecast_df 
    }
    return ticker, model_name, results






def plot_quantiles_example():
    """Plot the quantiles of the data."""

    ticker = "PFE"
    csv_filepath = f"data/{ticker}.csv"

    df = pd.read_csv(csv_filepath, index_col='Date', parse_dates=True)
    df['Return'] = np.pad(np.diff(np.log(df['Close'])) * 100, (1, 0), 'constant', constant_values=np.nan)
    df.index = pd.to_datetime(df.index, utc=True)

    mean_return = df['Return'].mean()
    df['e'] = df['Return'] - mean_return
    df['Volatility'] = (df['e']) ** 2
    df['unsmoothed_rolling_vol'] = df['Volatility'].rolling(window=21).mean()
    # smooth it with 10 day MA
    df['Rolling volatility'] = df['unsmoothed_rolling_vol'].rolling(window=10).mean()

    quantiles = df['Rolling volatility'].quantile([0.1, 0.9])
    q_low = quantiles.loc[0.1]
    q_high = quantiles.loc[0.9]

    under = df['Rolling volatility'].where(
        (df['Rolling volatility'] < q_low)
    )
    over = df['Rolling volatility'].where(
        (df['Rolling volatility'] > q_high)
    )
    plt.figure(figsize=(20, 12))
    plt.plot(df.index, df['Rolling volatility'], label="Rolling Volatility", linewidth=2)
    plt.axhline(y=q_low, color='gray', linestyle='--', label='10% quantile')
    plt.axhline(y=q_high, color='gray', linestyle='--', label='90% quantile')
    plt.plot(df.index, over, color='red', linewidth=3)
    plt.plot(df.index, under, color='red', linewidth=3)
    plt.title(f"{ticker} Rolling Volatility with 10% and 90% quantiles", fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Volatility', fontsize=12)
    plt.grid(True)
    plt.show()





def fit_and_forecast_models():
    """Fit and forecast models for each stock."""
    global tickers
    for ticker in tickers:
        print(f"Processing {ticker}...")
        csv_filepath = os.path.join(data_dir, f"{ticker}.csv")

        try:
            df = pd.read_csv(csv_filepath, index_col='Date', parse_dates=True)
        except FileNotFoundError:
            print(f"Warning: Data file not found for {ticker} at {csv_filepath}. Skipping.")
            continue

        df['Return'] = np.log(df['Close']).diff() * 100
        df = df.dropna(subset=['Return'])

        if df.empty:
            print(f"Warning: No return data for {ticker}. Skipping.")
            continue

        mean_return = df['Return'].mean()
        df['e'] = df['Return'] - mean_return 
        df['Volatility'] = (df['e']) ** 2 

        df['unsmoothed_rolling_vol'] = df['Volatility'].rolling(window=rolling_window).mean()

        df['Rolling volatility'] = df['unsmoothed_rolling_vol'].rolling(window=smoothing_window).mean()

        df = df.dropna(subset=['Rolling volatility'])

        if df.empty:
            print(f"Warning: Not enough data after calculating rolling windows for {ticker}. Skipping.")
            continue

        quantiles = df['Rolling volatility'].quantile([0.1, 0.9])
        df['q_low'] = quantiles.loc[0.1]
        df['q_high'] = quantiles.loc[0.9]

        all_stock_data[ticker] = df
        print(f"Finished processing {ticker}.")

    tickers = list(all_stock_data.keys())
    if not tickers:
        raise ValueError("No stock data could be processed. Check file paths and data integrity.")

    tasks = []
    for ticker in tickers:
        if ticker in all_stock_data: 
            df_ticker = all_stock_data[ticker]
            for model_spec in models_to_run:
                tasks.append((ticker, model_spec, df_ticker))

    print(f"\nRunning {len(tasks)} model fits in parallel using {os.cpu_count() if n_jobs == -1 else n_jobs} cores...")
    results_list = Parallel(n_jobs=n_jobs, prefer="processes", verbose=10)(
        delayed(run_rolling_forecast_for_model)(
            ticker, spec, df, train_window, forecast_horizon
        ) for ticker, spec, df in tasks 
    )

    global all_stock_results
    all_stock_results = {ticker: {} for ticker in tickers if ticker in all_stock_data}
    for result in results_list:
        if result is not None:
            ticker, model_name, results_dict = result
            if ticker in all_stock_results and results_dict is not None:
                all_stock_results[ticker][model_name] = results_dict

    print("\nFinished all model fitting and forecasting.")
    return all_stock_results

def plot_average_rmse(all_stock_results):
    """Plot the average RMSE among all stocks for each model."""
    avg_rmse_data = {
        'Total': [],
        'Low Volatility': [],
        'Medium Volatility': [],
        'High Volatility': []
    }
    global model_names
    model_names = [m['name'] for m in models_to_run]

    for model_name in model_names:
        rmses_overall, rmses_low, rmses_med, rmses_high = [], [], [], []
        for ticker in tickers:
            if ticker in all_stock_results and model_name in all_stock_results.get(ticker, {}):
                results = all_stock_results[ticker][model_name]
                if results and not pd.isna(results.get('rmse_overall')):
                    rmses_overall.append(results['rmse_overall'])
                    rmses_low.append(results['rmse_low'])
                    rmses_med.append(results['rmse_med'])
                    rmses_high.append(results['rmse_high'])
                else: 
                    rmses_overall.append(np.nan)
                    rmses_low.append(np.nan)
                    rmses_med.append(np.nan)
                    rmses_high.append(np.nan)
            else:
                rmses_overall.append(np.nan)
                rmses_low.append(np.nan)
                rmses_med.append(np.nan)
                rmses_high.append(np.nan)


        avg_rmse_data['Total'].append(np.nanmean(rmses_overall) if not all(np.isnan(rmses_overall)) else np.nan)
        avg_rmse_data['Low Volatility'].append(np.nanmean(rmses_low) if not all(np.isnan(rmses_low)) else np.nan)
        avg_rmse_data['Medium Volatility'].append(np.nanmean(rmses_med) if not all(np.isnan(rmses_med)) else np.nan)
        avg_rmse_data['High Volatility'].append(np.nanmean(rmses_high) if not all(np.isnan(rmses_high)) else np.nan)


    avg_rmse_df = pd.DataFrame(avg_rmse_data, index=model_names)

    fig_agg, ax_agg = plt.subplots(figsize=(12, 8))
    avg_rmse_df.plot(kind='bar', ax=ax_agg)
    ax_agg.set_title('Average RMSE Among All Stocks for Each Model and Period')
    ax_agg.set_ylabel('Average RMSE')
    ax_agg.set_xlabel('Model')
    ax_agg.tick_params(axis='x', rotation=45)
    ax_agg.legend(title='Period')
    ax_agg.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

    print("\nAverage RMSE Table:")
    print(avg_rmse_df)

def plot_rmse_across_tickers(all_stock_results):
    """Plot RMSE for each stock and model."""
    n_stocks = len(tickers)
    n_cols = 3
    n_rows = (n_stocks + n_cols - 1) // n_cols

    fig_sub, axes_sub = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5), squeeze=False)
    axes_flat = axes_sub.flatten()

    for i, ticker in enumerate(tickers):
        ax = axes_flat[i]
        stock_results = all_stock_results.get(ticker, {})

        if not stock_results:
            ax.set_title(f"{ticker} (No Data/Results)")
            ax.text(0.5, 0.5, "No results available", ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        rmse_data = {
            'Total': [stock_results.get(model_name, {}).get('rmse_overall', np.nan) for model_name in model_names],
            'Low Volatility': [stock_results.get(model_name, {}).get('rmse_low', np.nan) for model_name in model_names],
            'Medium Volatility': [stock_results.get(model_name, {}).get('rmse_med', np.nan) for model_name in model_names],
            'High Volatility': [stock_results.get(model_name, {}).get('rmse_high', np.nan) for model_name in model_names]
        }
        rmse_df = pd.DataFrame(rmse_data, index=model_names)

        rmse_df.plot(kind='bar', ax=ax)
        ax.set_title(f'RMSE for {ticker}')
        ax.set_ylabel('RMSE')
        ax.set_xlabel('Model')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Period', fontsize='small')
        ax.grid(axis='y', linestyle='--')

    for j in range(i + 1, n_rows * n_cols):
        fig_sub.delaxes(axes_flat[j])

    plt.suptitle('Per-Stock RMSE for Volatility Forecasts', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()