import pandas as pd
import numpy as np
<<<<<<< HEAD
=======

def generate_signals(prices, momentum, volatility):
    signals = pd.DataFrame(index=prices.index, columns=prices.columns, data=0)
    signals[momentum > 0.05] = 1   # BUY if 5%+ momentum
    signals[momentum < -0.05] = -1  # SELL
    return signals  # Issue 8
>>>>>>> b95e9f2 (fix: add missing imports to strategy)

def generate_signals(prices, momentum, volatility):
    signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    # Momentum strategy with volatility filter
    high_mom = momentum > 0.03
    low_vol = volatility < volatility.quantile(0.7, axis=0)
    signals[high_mom & low_vol] = 1    # BUY
    signals[momentum < -0.03] = -1    # SELL
    return signals

def size_positions(signals, volatility, capital, max_pct=0.25):
    inv_vol = 1 / (volatility.clip(lower=1e-6))
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    weights = weights.clip(upper=max_pct)
    buy_signals = signals.replace(-1, 0)
    return weights * buy_signals

def apply_costs(trade_value, slippage=0.001, commission=0.0005):
    return trade_value * (1 - slippage - commission)

def backtest(prices, signals, weights, capital):
    portfolio_value = [capital]
    returns = prices.pct_change().fillna(0)
    aligned_weights = weights.reindex(returns.index).fillna(0)
    
    for i in range(1, len(prices)):
        w = aligned_weights.iloc[i-1]
        r = returns.iloc[i]
        daily_return = (w * r).sum()
        # Apply costs on signal changes
        signal_change = (signals.iloc[i] != signals.iloc[i-1]).sum()
        cost = 0.0015 * signal_change / len(prices.columns)
        capital *= (1 + daily_return - cost)
        portfolio_value.append(capital)
    
    return pd.Series(portfolio_value, index=prices.index)

def rebalance_check(weights, target_weights, threshold=0.05):
    drift = (weights - target_weights).abs()
    return drift.max() > threshold