def generate_signals(prices, momentum, volatility):
    signals = pd.DataFrame(index=prices.index, columns=prices.columns, data=0)
    signals[momentum > 0.05] = 1   # BUY if 5%+ momentum
    signals[momentum < -0.05] = -1  # SELL
    return signals  # Issue 8

def size_positions(signals, volatility, capital, max_pct=0.2):
    # Risk-aware: lower vol → bigger position (Issue 9)
    inv_vol = 1 / (volatility + 1e-6)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    weights = weights.clip(upper=max_pct)
    return weights * signals.replace(-1, 0)

def apply_costs(trade_value, slippage=0.001, commission=0.0005):
    return trade_value * (1 - slippage - commission)  # Issue 10

def backtest(prices, signals, weights, capital):
    portfolio_value = [capital]
    returns = prices.pct_change().fillna(0)
    for i in range(1, len(prices)):
        daily_return = (weights.iloc[i-1] * returns.iloc[i]).sum()
        capital *= (1 + daily_return)
        portfolio_value.append(capital)
    return pd.Series(portfolio_value, index=prices.index)