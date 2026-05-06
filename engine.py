import pandas as pd
import numpy as np
import os

DATA_PATH = "data/raw/"

def load_data():
    multi = pd.read_csv(DATA_PATH + "multi_asset_dataset.csv", index_col=0, parse_dates=True)
    
    # Keep only numeric price columns
    multi = multi.select_dtypes(include=[np.number])
    multi = multi.ffill().bfill().dropna(axis=1, thresh=int(len(multi)*0.7))
    return multi

def load_macro():
    macro = pd.read_csv(DATA_PATH + "macro_dataset.csv", index_col=0, parse_dates=True)
    return macro.select_dtypes(include=[np.number]).ffill().bfill()

def load_oil():
    oil = pd.read_csv(DATA_PATH + "oil_dataset.csv", index_col=0, parse_dates=True)
    return oil.select_dtypes(include=[np.number]).ffill().bfill()

def engineer_features(prices):
    returns = prices.pct_change().dropna()
    volatility = returns.rolling(20).std() * np.sqrt(252)
    momentum = prices / prices.shift(20) - 1
    return returns, volatility.dropna(), momentum.dropna()

def calc_var(returns, confidence=0.95):
    if isinstance(returns, pd.DataFrame):
        port_ret = returns.mean(axis=1)
    else:
        port_ret = returns
    return float(port_ret.quantile(1 - confidence))

def calc_sharpe(portfolio_series, rfr=0.04):
    returns = portfolio_series.pct_change().dropna()
    excess = returns.mean() * 252 - rfr
    return float(excess / (returns.std() * np.sqrt(252)))

def calc_max_drawdown(portfolio_series):
    roll_max = portfolio_series.cummax()
    drawdown = (portfolio_series - roll_max) / roll_max
    return float(drawdown.min())

def calc_alpha_beta(portfolio_series, benchmark_col=None, prices=None):
    port_ret = portfolio_series.pct_change().dropna()
    if prices is not None:
        bench = prices.iloc[:, 0].pct_change().dropna()
        bench = bench.reindex(port_ret.index).dropna()
        port_ret = port_ret.reindex(bench.index)
        cov = np.cov(port_ret, bench)
        beta = cov[0, 1] / cov[1, 1]
        alpha = (port_ret.mean() - beta * bench.mean()) * 252
        return round(alpha, 4), round(beta, 4)
    return 0.0, 1.0