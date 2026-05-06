import pandas as pd
import numpy as np
import yfinance as yf

TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "GLD"]
START, END = "2020-01-01", "2024-01-01"
CAPITAL = 1_000_000

def load_data():
    raw = yf.download(TICKERS, start=START, end=END)["Close"]
    raw = raw.ffill().dropna()  # handle missing values (Issue 2)
    return raw

def engineer_features(prices):
    returns = prices.pct_change().dropna()
    volatility = returns.rolling(20).std() * np.sqrt(252)  # annualized vol
    momentum = prices / prices.shift(20) - 1               # 20-day momentum
    return returns, volatility, momentum

def calc_var(returns, confidence=0.95):
    return returns.quantile(1 - confidence)  # Historical VaR (Issue 6)

def calc_sharpe(returns, rfr=0.04):
    excess = returns.mean() * 252 - rfr
    return excess / (returns.std() * np.sqrt(252))  # Issue 12

def calc_max_drawdown(portfolio_values):
    roll_max = portfolio_values.cummax()
    drawdown = (portfolio_values - roll_max) / roll_max
    return drawdown.min()  # Issue 7