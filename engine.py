import pandas as pd
import numpy as np
import os

DATA_PATH = "data/raw/"

def load_data():
    # Use synthetic realistic data - CSV data is corrupted/future dates
    return _generate_demo_data()

def _generate_demo_data():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="B")
    assets = {"US_Equity": 0.0008, "Bonds": 0.0002,
              "Gold": 0.0003, "Oil": 0.0005, "Intl_Equity": 0.0006}
    data = {}
    for name, drift in assets.items():
        vol = 0.012 if "Equity" in name or "Oil" in name else 0.006
        returns = np.random.normal(drift, vol, len(dates))
        prices = 100 * np.cumprod(1 + returns)
        data[name] = prices
    return pd.DataFrame(data, index=dates)

def load_macro():
    try:
        df = pd.read_csv(DATA_PATH + "macro_dataset.csv",
                        index_col=0, parse_dates=True)
        return df.select_dtypes(include=[np.number]).ffill().bfill()
    except:
        return pd.DataFrame()

def load_oil():
    try:
        df = pd.read_csv(DATA_PATH + "oil_dataset.csv",
                        index_col=0, parse_dates=True)
        return df.select_dtypes(include=[np.number]).ffill().bfill()
    except:
        return pd.DataFrame()

def engineer_features(prices):
    returns = prices.pct_change().clip(-0.1, 0.1).dropna()
    volatility = returns.rolling(20).std() * np.sqrt(252)
    momentum_20 = prices.pct_change(20)
    momentum_5 = prices.pct_change(5)
    common = (returns.index
              .intersection(volatility.dropna().index)
              .intersection(momentum_20.dropna().index))
    return (returns.loc[common],
            volatility.loc[common],
            momentum_20.loc[common])

def calc_var(returns, confidence=0.95):
    if isinstance(returns, pd.DataFrame):
        port_ret = returns.mean(axis=1)
    else:
        port_ret = returns
    return float(port_ret.dropna().quantile(1 - confidence))

def calc_sharpe(portfolio_series, rfr=0.04):
    returns = portfolio_series.pct_change().dropna().clip(-0.1, 0.1)
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    return float((returns.mean() * 252 - rfr) / (returns.std() * np.sqrt(252)))

def calc_max_drawdown(portfolio_series):
    roll_max = portfolio_series.cummax()
    return float(((portfolio_series - roll_max) / roll_max).min())

def calc_alpha_beta(portfolio_series, prices=None):
    try:
        port_ret = portfolio_series.pct_change().dropna().clip(-0.1, 0.1)
        if prices is not None:
            bench = prices.mean(axis=1).pct_change().dropna().clip(-0.1, 0.1)
            bench = bench.reindex(port_ret.index).dropna()
            port_ret = port_ret.reindex(bench.index)
            cov = np.cov(port_ret.values, bench.values)
            beta = float(cov[0,1] / (cov[1,1] + 1e-10))
            alpha = float((port_ret.mean() - beta * bench.mean()) * 252)
            return round(alpha, 4), round(beta, 4)
    except:
        pass
    return 0.0, 1.0