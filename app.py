import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Hedge Fund Risk System", layout="wide", page_icon="🏦")

st.title("🏦 Hedge Fund Risk Modeling System")
st.markdown("*Semi-Automated Trading & Risk Dashboard*")

# Sidebar
st.sidebar.header("⚙️ Portfolio Controls")
capital = st.sidebar.number_input("Initial Capital ($)", value=1_000_000, step=100_000)
confidence = st.sidebar.slider("VaR Confidence Level", 0.90, 0.99, 0.95)
st.sidebar.markdown("---")
st.sidebar.info("System analyzes multi-asset portfolio with real-time risk metrics")

# Try to import engine (Person 1's work)
try:
    from engine import load_data, engineer_features, calc_sharpe, calc_max_drawdown, calc_var
    from strategy import generate_signals, size_positions, backtest

    with st.spinner("Loading market data..."):
        prices = load_data()
        returns, vol, momentum = engineer_features(prices)
        signals = generate_signals(prices, momentum, vol)
        weights = size_positions(signals, vol, capital)
        portfolio = backtest(prices, signals, weights, capital)

    port_returns = portfolio.pct_change().dropna()
    sharpe = calc_sharpe(port_returns)
    drawdown = calc_max_drawdown(portfolio)
    var = calc_var(port_returns, confidence)
    final_val = portfolio.iloc[-1]
    total_return = (final_val - capital) / capital * 100

    # ── Metrics Row ──
    st.subheader("📊 Key Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📈 Sharpe Ratio", f"{sharpe:.2f}", delta="Higher is better")
    col2.metric("📉 Max Drawdown", f"{drawdown*100:.1f}%", delta_color="inverse")
    col3.metric("⚠️ VaR (95%)", f"{var*100:.2f}%", delta_color="inverse")
    col4.metric("💰 Final Value", f"${final_val:,.0f}")
    col5.metric("📊 Total Return", f"{total_return:.1f}%")

    st.markdown("---")

    # ── Portfolio Value Chart ──
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📈 Portfolio Value Over Time")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=portfolio.index, y=portfolio.values,
            fill='tozeroy', name="Portfolio Value",
            line=dict(color='#00C851', width=2)
        ))
        fig1.update_layout(
            xaxis_title="Date", yaxis_title="Portfolio Value ($)",
            template="plotly_dark", height=350
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        st.subheader("📉 Drawdown Over Time")
        roll_max = portfolio.cummax()
        dd_series = (portfolio - roll_max) / roll_max * 100
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dd_series.index, y=dd_series.values,
            fill='tozeroy', name="Drawdown %",
            line=dict(color='#FF4444', width=2)
        ))
        fig2.update_layout(
            xaxis_title="Date", yaxis_title="Drawdown (%)",
            template="plotly_dark", height=350
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Daily Returns Histogram ──
    st.subheader("📊 Daily Returns Distribution")
    fig3 = px.histogram(
        port_returns, nbins=50, title="Daily Returns Distribution",
        color_discrete_sequence=["#00C851"], template="plotly_dark"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Signal Log ──
    st.subheader("📋 Explainable Signal Log (Latest 20 Days)")
    st.dataframe(signals.tail(20).style.applymap(
        lambda v: 'background-color: #1a4a1a' if v == 1
        else ('background-color: #4a1a1a' if v == -1 else '')
    ))

    # ── Asset Prices ──
    st.subheader("🏷️ Asset Price History")
    fig4 = go.Figure()
    for col in prices.columns:
        fig4.add_trace(go.Scatter(x=prices.index, y=prices[col], name=col))
    fig4.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig4, use_container_width=True)

except ImportError as e:
    st.warning(f"Waiting for backend engine... ({e})")
    st.info("Person 1 needs to complete engine.py and strategy.py")