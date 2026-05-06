import pandas as pd
import numpy as np

def generate_signals(prices, momentum, volatility):
    signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    sma20 = prices.rolling(20).mean()
    sma50 = prices.rolling(50).mean()
    # Mean reversion + trend following hybrid
    trend_up = sma20 > sma50
    mom_positive = momentum > 0.01
    mom_negative = momentum < -0.02
    signals[trend_up & mom_positive] = 1   # BUY
    signals[~trend_up & mom_negative] = -1  # SELL
    return signals.fillna(0)

def size_positions(signals, volatility, capital, max_pct=0.35):
    buy = signals.copy()
    buy[buy < 0] = 0
    # Inverse volatility weighting
    inv_vol = 1.0 / (volatility.clip(lower=0.01))
    inv_vol = inv_vol * buy  # only weight assets we're buying
    row_sum = inv_vol.sum(axis=1).replace(0, 1)
    weights = inv_vol.div(row_sum, axis=0).clip(upper=max_pct)
    return weights.fillna(0)

def apply_costs(trade_value, slippage=0.001, commission=0.0005):
    return trade_value * (1 - slippage - commission)

def backtest(prices, signals, weights, capital):
    returns = prices.pct_change().clip(-0.1, 0.1).fillna(0)
    aligned = weights.reindex(returns.index).fillna(0)
    portfolio_value = []
    cur = float(capital)
    prev_w = pd.Series(0, index=prices.columns)
    for i in range(len(prices)):
        if i == 0:
            portfolio_value.append(cur)
            continue
        w = aligned.iloc[i-1]
        r = returns.iloc[i]
        daily_ret = float((w * r).sum())
        # Transaction cost on rebalancing
        turnover = float((w - prev_w).abs().sum())
        cost = 0.001 * turnover
        cur = max(cur * (1 + daily_ret - cost), 1.0)
        portfolio_value.append(cur)
        prev_w = w
    return pd.Series(portfolio_value, index=prices.index)

def rebalance_check(weights, target_weights, threshold=0.05):
    try:
        return bool((weights - target_weights).abs().max() > threshold)
    except:
        return False