import pandas as pd
import numpy as np

def generate_signals(prices, momentum, volatility):
    signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    # Simple moving average crossover + momentum
    sma20 = prices.rolling(20).mean()
    sma50 = prices.rolling(50).mean()
    # BUY: price above SMA20, positive momentum
    buy = (prices > sma20) & (momentum > 0.01)
    # SELL: price below SMA20, negative momentum  
    sell = (prices < sma20) & (momentum < -0.01)
    signals[buy] = 1
    signals[sell] = -1
    return signals.fillna(0)

def size_positions(signals, volatility, capital, max_pct=0.3):
    buy_signals = signals.copy()
    buy_signals[buy_signals < 0] = 0
    # Equal weight among buy signals
    n_buys = buy_signals.sum(axis=1).replace(0, 1)
    weights = buy_signals.div(n_buys, axis=0)
    weights = weights.clip(upper=max_pct)
    return weights

def apply_costs(trade_value, slippage=0.001, commission=0.0005):
    return trade_value * (1 - slippage - commission)

def backtest(prices, signals, weights, capital):
    portfolio_value = [capital]
    returns = prices.pct_change().clip(-0.05, 0.05).fillna(0)
    aligned = weights.reindex(returns.index).fillna(0)
    cur_capital = float(capital)
    for i in range(1, len(prices)):
        w = aligned.iloc[i-1].values
        r = returns.iloc[i].values
        daily_return = float(np.dot(w, r))
        # Transaction cost
        if i > 1:
            trades = np.abs(aligned.iloc[i] - aligned.iloc[i-1]).sum()
            daily_return -= 0.001 * float(trades)
        cur_capital *= (1 + daily_return)
        cur_capital = max(cur_capital, 1)  # prevent going to zero
        portfolio_value.append(cur_capital)
    return pd.Series(portfolio_value, index=prices.index)

def rebalance_check(weights, target_weights, threshold=0.05):
    try:
        drift = (weights - target_weights).abs()
        return bool(drift.max() > threshold)
    except:
        return False