import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Portfolio Builder")

st.title("💼 Portfolio Builder")

# ---------- Light theme styling ----------
st.markdown("""
<style>
    .stApp { background-color: #ffffff; color: #000000; }
    h1,h2,h3,h4,h5,h6,p,span,div,label { color: #000000 !important; }
</style>
""", unsafe_allow_html=True)

# ---------- Session state ----------
if "portfolio_tickers" not in st.session_state:
    st.session_state.portfolio_tickers = ["AAPL", "MSFT"]
if "portfolio_weights" not in st.session_state:
    st.session_state.portfolio_weights = [0.5, 0.5]

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Portfolio Setup")

    new_ticker = st.text_input("Add Stock", "")
    if st.button("Add Stock"):
        t = new_ticker.strip().upper()
        if t and t not in st.session_state.portfolio_tickers:
            st.session_state.portfolio_tickers.append(t)
            st.session_state.portfolio_weights.append(1.0)
        else:
            st.warning("Invalid or duplicate ticker.")

    if st.button("Clear Portfolio"):
        st.session_state.portfolio_tickers = []
        st.session_state.portfolio_weights = []

    for i, t in enumerate(st.session_state.portfolio_tickers):
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.markdown(f"**{t}**")
        with c2:
            st.session_state.portfolio_weights[i] = st.number_input(
                f"Weight {t}", min_value=0.0, max_value=1.0,
                value=float(st.session_state.portfolio_weights[i]), key=f"w_{t}_{i}"
            )
        with c3:
            if st.button("❌", key=f"del_{t}_{i}"):
                st.session_state.portfolio_tickers.pop(i)
                st.session_state.portfolio_weights.pop(i)
                st.experimental_rerun()

    normalize = st.checkbox("Normalize Weights", True)
    if normalize and sum(st.session_state.portfolio_weights) > 0:
        total = sum(st.session_state.portfolio_weights)
        st.session_state.portfolio_weights = [w / total for w in st.session_state.portfolio_weights]

    horizon = st.selectbox("Select Time Horizon", ["1M", "3M", "6M", "1Y", "5Y", "YTD", "Max"], index=3)

def get_period(h):
    return {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y", "YTD": "ytd", "Max": "max"}[h]

with col2:
    if len(st.session_state.portfolio_tickers) == 0:
        st.info("Add tickers to build your portfolio.")
    else:
        try:
            data = yf.download(st.session_state.portfolio_tickers, period=get_period(horizon))["Adj Close"]
            weights = np.array(st.session_state.portfolio_weights)
            if normalize:
                weights = weights / weights.sum()

            daily_returns = data.pct_change().fillna(0)
            portfolio_return = (daily_returns * weights).sum(axis=1)
            portfolio_cum = (np.cumprod(1 + portfolio_return) - 1) * 100
            cum_returns = (data / data.iloc[0] - 1) * 100

            fig = go.Figure()
            for t in cum_returns.columns:
                fig.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns[t], mode="lines", name=t))
            fig.add_trace(go.Scatter(x=cum_returns.index, y=portfolio_cum, mode="lines", name="Portfolio",
                                     line=dict(color="red", width=3)))
            fig.update_layout(title="Portfolio vs Individual Stocks", xaxis_title="Date",
                              yaxis_title="Cumulative Return (%)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            def sharpe_ratio(series):
                rets = series.pct_change().dropna()
                return np.sqrt(252) * rets.mean() / rets.std() if rets.std() != 0 else np.nan

            stats = []
            for t in data.columns:
                cum = (data[t].iloc[-1] / data[t].iloc[0]) - 1
                stats.append([t, f"{weights[list(data.columns).index(t)]:.2%}",
                              f"{cum*100:.2f}%", f"{sharpe_ratio(data[t]):.2f}"])

            port_cum = portfolio_cum.iloc[-1] / 100
            port_sharpe = np.sqrt(252) * portfolio_return.mean() / portfolio_return.std()
            stats.append(["Portfolio", "100%", f"{port_cum*100:.2f}%", f"{port_sharpe:.2f}"])

            st.subheader("Performance Summary")
            df = pd.DataFrame(stats, columns=["Ticker", "Weight", "Cumulative Return", "Sharpe Ratio"])
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading data: {e}")
