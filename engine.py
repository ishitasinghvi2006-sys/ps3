import pandas as pd
import numpy as np
import os

DATA_PATH = "data/raw/"

def load_data():
    df = pd.read_csv(DATA_PATH + "multi_asset_dataset.csv", 
                     index_col=0, parse_dates=True)
    # Keep only actual price columns (positive values, not returns)
    numeric = df.select_dtypes(include=[np.number])
    # Price columns have large values, return columns are small (-1 to 1)
    price_cols = [c for c in numeric.columns 
                  if numeric[c].median() > 1 and 'return' not in c.lower()]
    if len(price_cols) == 0:
        price_cols = numeric.columns[:5].tolist()
    prices = numeric[price_cols].ffill().bfill()
    prices = prices.loc[:, prices.std() > 0]
    return prices

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
    returns = prices.pct_change()
    # Remove extreme return rows (data errors)
    returns = returns.clip(-0.05, 0.05)
    returns = returns.dropna()
    volatility = returns.rolling(20).std() * np.sqrt(252)
    momentum = prices.pct_change(20)
    common = returns.index.intersection(
                volatility.dropna().index).intersection(
                momentum.dropna().index)
    return (returns.loc[common], 
            volatility.loc[common], 
            momentum.loc[common])

def calc_var(returns, confidence=0.95):
    if isinstance(returns, pd.DataFrame):
        port_ret = returns.mean(axis=1)
    else:
        port_ret = returns
    port_ret = port_ret.dropna()
    return float(port_ret.quantile(1 - confidence))

def calc_sharpe(portfolio_series, rfr=0.02):
    returns = portfolio_series.pct_change().dropna()
    returns = returns.clip(-0.05, 0.05)
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    excess = returns.mean() * 252 - rfr
    return float(excess / (returns.std() * np.sqrt(252)))

def calc_max_drawdown(portfolio_series):
    roll_max = portfolio_series.cummax()
    drawdown = (portfolio_series - roll_max) / roll_max
    return float(drawdown.min())

def calc_alpha_beta(portfolio_series, prices=None):
    try:
        port_ret = portfolio_series.pct_change().dropna().clip(-0.05, 0.05)
        if prices is not None and len(prices.columns) > 0:
            bench = prices.iloc[:, 0].pct_change().dropna().clip(-0.05, 0.05)
            bench = bench.reindex(port_ret.index).dropna()
            port_ret = port_ret.reindex(bench.index)
            if len(bench) < 10:
                return 0.0, 1.0
            cov = np.cov(port_ret.values, bench.values)
            beta = float(cov[0, 1] / (cov[1, 1] + 1e-10))
            alpha = float((port_ret.mean() - beta * bench.mean()) * 252)
            return round(alpha, 4), round(beta, 4)
    except:
        pass
    return 0.0, 1.0