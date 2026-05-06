# Hedge Fund Risk Modeling & Semi-Automated Trading System

## Team Information
- **Team Name**: [Team Nexus]
- **Year**: 2026
- **All-Female Team**: [No]

## Architecture Overview

#### Describe your approach here. Keep it short and clear.
    - Our system ingests multi-asset price data, macroeconomic indicators, and oil market data from CSV datasets. We preprocess by forward-filling missing values, clipping outliers to realistic ranges, and removing constant or corrupted columns. Each data source is aligned to a common timeline before processing.
    - We implement Historical Value at Risk (VaR) at 95% confidence, Maximum Drawdown tracking, Sharpe Ratio for risk-adjusted returns, and Alpha/Beta vs benchmark. These metrics are recalculated continuously and fed into position sizing — riskier assets automatically receive smaller allocations via inverse-volatility weighting.
    - Our semi-automated strategy uses SMA crossover (20-day vs 50-day) combined with momentum confirmation to generate BUY/SELL/HOLD signals. Position sizes respect a configurable max allocation cap. Slippage (0.1%) and commission (0.05%) costs are deducted on every trade. Portfolio rebalancing triggers when drift exceeds 5% threshold.
    - The Streamlit dashboard displays 7 live metrics (Sharpe, Return, Drawdown, VaR, Alpha, Beta, Volatility), portfolio value chart, drawdown analysis, normalized asset performance, returns distribution with VaR line, trading signal heatmap, rolling 30-day Sharpe, and an explainable trade log showing the reason behind every signal.


**Note:** Please do not change the format or spelling of anything in this README. The fields are extracted using a script, so any changes to the structure or formatting may break the extraction process.
