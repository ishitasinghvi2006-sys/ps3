import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="HedgeAI - Risk Modeling System",
    layout="wide", page_icon="🏦",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0a0e1a; }
[data-testid="stSidebar"] { background: #0f1629; border-right: 1px solid #1e3a5f; }
.metric-box { background: linear-gradient(135deg, #0f1629, #1a2744);
  border: 1px solid #1e3a5f; border-radius: 12px; padding: 20px;
  text-align: center; margin: 4px; }
.metric-val { font-size: 28px; font-weight: 700; margin: 8px 0; }
.metric-lbl { font-size: 12px; color: #8892a4; text-transform: uppercase; letter-spacing: 1px; }
.positive { color: #00d084; }
.negative { color: #ff4757; }
.neutral  { color: #ffa502; }
.section-header { background: linear-gradient(90deg, #1e3a5f, transparent);
  padding: 10px 16px; border-left: 3px solid #00d084;
  border-radius: 4px; margin: 16px 0 8px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
st.sidebar.markdown("## ⚙️ Portfolio Controls")
capital = st.sidebar.number_input("Initial Capital ($)", value=1_000_000, step=100_000)
confidence = st.sidebar.slider("VaR Confidence", 0.90, 0.99, 0.95)
max_pos = st.sidebar.slider("Max Position Size (%)", 10, 50, 35) / 100
st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Data Sources")
st.sidebar.success("✅ Multi-Asset Prices")
st.sidebar.success("✅ Macro Indicators")
st.sidebar.success("✅ Oil Market Data")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Strategy")
st.sidebar.info("SMA Crossover + Momentum\nInverse-Vol Position Sizing\nSlippage & Cost Simulation")

# ── Load ──
from engine import (load_data, load_macro, load_oil, engineer_features,
                    calc_sharpe, calc_max_drawdown, calc_var, calc_alpha_beta)
from strategy import generate_signals, size_positions, backtest

st.markdown("<h1 style='color:#fff;margin-bottom:0'>🏦 HedgeAI Risk Modeling System</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#8892a4;margin-top:4px'>Semi-Automated Trading & Risk Analysis Dashboard</p>", unsafe_allow_html=True)

with st.spinner("⚡ Running backtest engine..."):
    prices = load_data()
    macro  = load_macro()
    oil    = load_oil()
    returns, vol, momentum = engineer_features(prices)
    common = returns.index.intersection(vol.index).intersection(momentum.index)
    returns  = returns.loc[common]
    vol      = vol.loc[common]
    momentum = momentum.loc[common]
    prices_a = prices.loc[common]
    signals  = generate_signals(prices_a, momentum, vol)
    weights  = size_positions(signals, vol, capital, max_pct=max_pos)
    portfolio = backtest(prices_a, signals, weights, capital)

port_ret  = portfolio.pct_change().dropna()
sharpe    = calc_sharpe(portfolio)
drawdown  = calc_max_drawdown(portfolio)
var_val   = calc_var(returns, confidence)
final_val = portfolio.iloc[-1]
total_ret = (final_val - capital) / capital * 100
alpha, beta = calc_alpha_beta(portfolio, prices=prices_a)
ann_vol   = port_ret.std() * np.sqrt(252) * 100
ann_ret   = port_ret.mean() * 252 * 100
n_trades  = int((signals.diff().abs().sum().sum()) / 2)

# ── Metrics ──
st.markdown("<div class='section-header'><b style='color:#fff'>📊 Key Performance Metrics</b></div>", unsafe_allow_html=True)

def color_class(val, good_positive=True):
    if good_positive:
        return "positive" if val > 0 else "negative"
    else:
        return "negative" if val > 0 else "positive"

cols = st.columns(7)
metrics = [
    ("Sharpe Ratio", f"{sharpe:.2f}", sharpe > 0),
    ("Total Return", f"{total_ret:.1f}%", total_ret > 0),
    ("Max Drawdown", f"{drawdown*100:.1f}%", False),
    (f"VaR {int(confidence*100)}%", f"{var_val*100:.2f}%", False),
    ("Alpha", f"{alpha:.3f}", alpha > 0),
    ("Beta", f"{beta:.2f}", None),
    ("Ann. Vol", f"{ann_vol:.1f}%", None),
]
for col, (lbl, val, good) in zip(cols, metrics):
    if good is None:
        css = "neutral"
    elif good:
        css = "positive"
    else:
        css = "negative"
    col.markdown(f"""
    <div class='metric-box'>
      <div class='metric-lbl'>{lbl}</div>
      <div class='metric-val {css}'>{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 1: Portfolio + Drawdown ──
col1, col2 = st.columns(2)
with col1:
    st.markdown("<div class='section-header'><b style='color:#fff'>📈 Portfolio Value</b></div>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio.index, y=portfolio.values,
        fill='tozeroy', fillcolor='rgba(0,208,132,0.1)',
        line=dict(color='#00d084', width=2), name="Portfolio"
    ))
    fig.add_hline(y=capital, line_dash="dash",
                  line_color="#ffa502", annotation_text="Initial Capital",
                  annotation_font_color="#ffa502")
    fig.update_layout(
        template="plotly_dark", height=300, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,22,41,0.8)', margin=dict(t=10,b=40,l=60,r=20),
        yaxis_title="Value ($)", xaxis_title="Date"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("<div class='section-header'><b style='color:#fff'>📉 Drawdown Analysis</b></div>", unsafe_allow_html=True)
    roll_max = portfolio.cummax()
    dd = (portfolio - roll_max) / roll_max * 100
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill='tozeroy', fillcolor='rgba(255,71,87,0.15)',
        line=dict(color='#ff4757', width=1.5), name="Drawdown %"
    ))
    fig2.update_layout(
        template="plotly_dark", height=300, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,22,41,0.8)', margin=dict(t=10,b=40,l=60,r=20),
        yaxis_title="Drawdown (%)", xaxis_title="Date"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Normalized Prices + Returns Dist ──
col3, col4 = st.columns(2)
with col3:
    st.markdown("<div class='section-header'><b style='color:#fff'>🏷️ Asset Performance (Normalized)</b></div>", unsafe_allow_html=True)
    norm = prices_a / prices_a.iloc[0] * 100
    fig3 = go.Figure()
    colors = ['#00d084','#ffa502','#00b4d8','#ff4757','#a855f7','#f59e0b']
    for i, col in enumerate(norm.columns[:6]):
        fig3.add_trace(go.Scatter(
            x=norm.index, y=norm[col],
            name=str(col), line=dict(width=1.5, color=colors[i % len(colors)])
        ))
    fig3.update_layout(
        template="plotly_dark", height=300, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,22,41,0.8)', margin=dict(t=10,b=40,l=60,r=20)
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.markdown("<div class='section-header'><b style='color:#fff'>📊 Returns Distribution</b></div>", unsafe_allow_html=True)
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(
        x=port_ret.values, nbinsx=60,
        marker_color='#00b4d8', opacity=0.8, name="Returns"
    ))
    fig4.add_vline(x=float(port_ret.mean()), line_dash="dash",
                   line_color="#00d084", annotation_text="Mean",
                   annotation_font_color="#00d084")
    fig4.add_vline(x=var_val, line_dash="dash",
                   line_color="#ff4757", annotation_text="VaR",
                   annotation_font_color="#ff4757")
    fig4.update_layout(
        template="plotly_dark", height=300, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,22,41,0.8)', margin=dict(t=10,b=40,l=60,r=20)
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── Row 3: Signal Heatmap + Rolling Sharpe ──
col5, col6 = st.columns(2)
with col5:
    st.markdown("<div class='section-header'><b style='color:#fff'>🎯 Trading Signals Heatmap</b></div>", unsafe_allow_html=True)
    sig_tail = signals.tail(40).T
    fig5 = go.Figure(data=go.Heatmap(
        z=sig_tail.values,
        x=[str(d)[:10] for d in sig_tail.columns],
        y=[str(c) for c in sig_tail.index],
        colorscale=[[0,'#ff4757'],[0.5,'#1a2744'],[1,'#00d084']],
        zmin=-1, zmax=1, showscale=True,
        colorbar=dict(tickvals=[-1,0,1], ticktext=['SELL','HOLD','BUY'])
    ))
    fig5.update_layout(
        template="plotly_dark", height=300, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,22,41,0.8)', margin=dict(t=10,b=60,l=80,r=20)
    )
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    st.markdown("<div class='section-header'><b style='color:#fff'>📐 Rolling Sharpe Ratio (30-day)</b></div>", unsafe_allow_html=True)
    rolling_s = (port_ret.rolling(30).mean() /
                 port_ret.rolling(30).std()) * np.sqrt(252)
    fig6 = go.Figure()
    fig6.add_hrect(y0=1, y1=rolling_s.max()+0.5,
                   fillcolor="rgba(0,208,132,0.05)", line_width=0)
    fig6.add_hrect(y0=rolling_s.min()-0.5, y1=0,
                   fillcolor="rgba(255,71,87,0.05)", line_width=0)
    fig6.add_trace(go.Scatter(
        x=rolling_s.index, y=rolling_s.values,
        line=dict(color='#ffa502', width=2), name="Rolling Sharpe"
    ))
    fig6.add_hline(y=1.0, line_dash="dash", line_color="#00d084",
                   annotation_text="Target (1.0)", annotation_font_color="#00d084")
    fig6.add_hline(y=0, line_dash="dash", line_color="#ff4757", line_width=0.5)
    fig6.update_layout(
        template="plotly_dark", height=300, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,22,41,0.8)', margin=dict(t=10,b=40,l=60,r=20)
    )
    st.plotly_chart(fig6, use_container_width=True)

# ── Macro & Oil ──
if not macro.empty or not oil.empty:
    st.markdown("<div class='section-header'><b style='color:#fff'>🌍 Macro & Oil Context</b></div>", unsafe_allow_html=True)
    col7, col8 = st.columns(2)
    with col7:
        if not macro.empty:
            fig7 = go.Figure()
            colors2 = ['#00d084','#ffa502','#00b4d8']
            for i, c in enumerate(macro.columns[:3]):
                norm_col = (macro[c] - macro[c].mean()) / (macro[c].std() + 1e-10)
                fig7.add_trace(go.Scatter(
                    x=macro.index, y=norm_col,
                    name=str(c), line=dict(width=1.5, color=colors2[i])
                ))
            fig7.update_layout(
                template="plotly_dark", height=250,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,22,41,0.8)',
                margin=dict(t=10,b=40,l=60,r=20),
                title=dict(text="Macro Indicators (Normalized)", font=dict(color="#8892a4", size=12))
            )
            st.plotly_chart(fig7, use_container_width=True)
    with col8:
        if not oil.empty:
            fig8 = go.Figure()
            for i, c in enumerate(oil.columns[:2]):
                fig8.add_trace(go.Scatter(
                    x=oil.index, y=oil[c],
                    name=str(c), line=dict(width=1.5, color=colors2[i])
                ))
            fig8.update_layout(
                template="plotly_dark", height=250,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,22,41,0.8)',
                margin=dict(t=10,b=40,l=60,r=20),
                title=dict(text="Oil Price History", font=dict(color="#8892a4", size=12))
            )
            st.plotly_chart(fig8, use_container_width=True)

# ── Trade Log ──
st.markdown("<div class='section-header'><b style='color:#fff'>📋 Explainable Trade Log</b></div>", unsafe_allow_html=True)
log = []
for i in range(max(1, len(signals)-15), len(signals)):
    date = signals.index[i]
    buys  = int((signals.iloc[i] == 1).sum())
    sells = int((signals.iloc[i] == -1).sum())
    holds = int((signals.iloc[i] == 0).sum())
    pval  = portfolio.iloc[i]
    ddval = dd.iloc[i]
    action = "🟢 BUY" if buys > sells else ("🔴 SELL" if sells > buys else "⚪ HOLD")
    log.append({
        "Date": str(date)[:10], "Action": action,
        "BUY signals": buys, "SELL signals": sells, "HOLD": holds,
        "Portfolio Value": f"${pval:,.0f}", "Drawdown": f"{ddval:.1f}%",
        "Reason": f"SMA crossover + momentum {'↑' if buys > sells else '↓'}"
    })
st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)

# ── System Stats ──
st.markdown("<div class='section-header'><b style='color:#fff'>⚙️ System Statistics</b></div>", unsafe_allow_html=True)
sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Assets Tracked", len(prices_a.columns))
sc2.metric("Trading Days", len(portfolio))
sc3.metric("Total Trades", n_trades)
sc4.metric("Data Points", f"{len(prices_a) * len(prices_a.columns):,}")

# ── Architecture ──
st.markdown("<div class='section-header'><b style='color:#fff'>🏗️ System Architecture</b></div>", unsafe_allow_html=True)
a1, a2, a3, a4 = st.columns(4)
a1.markdown("**📥 Data Pipeline**\n\nMulti-asset ingestion · Missing value imputation · Outlier clipping · Schema validation")
a2.markdown("**🧮 Risk Engine**\n\nVaR (Historical) · Max Drawdown · Sharpe Ratio · Alpha/Beta vs benchmark")
a3.markdown("**🎯 Signal Engine**\n\nSMA crossover · Momentum filter · Inv-vol sizing · Slippage simulation")
a4.markdown("**📊 Dashboard**\n\nReal-time metrics · Signal heatmap · Rolling Sharpe · Explainable trade log")