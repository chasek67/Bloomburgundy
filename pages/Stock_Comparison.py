# pages/Stock_Comparison.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Stock Comparison")

st.markdown("""
<style>
    .stApp { background-color: #0e0e0e; color: #e6e6e6; }
    h1,h2,h3,label,p,span,div { color: #e6e6e6 !important; }
    .stTextInput>div>div>input { background-color: #1a1a1a; color: #e6e6e6; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Stock Comparison")

col1, col2 = st.columns([2, 1])

with col1:
    tickers_input = st.text_input("Enter tickers separated by commas", "AAPL,MSFT,GOOGL")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    if tickers:
        data = yf.download(tickers, period="1y")["Adj Close"]
        fig = go.Figure()
        for t in tickers:
            fig.add_trace(go.Scatter(x=data.index, y=data[t], mode="lines", name=t))
        fig.update_layout(title="Stock Prices", xaxis_title="Date", yaxis_title="Price", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        cum_returns = (data / data.iloc[0] - 1) * 100
        fig2 = go.Figure()
        for t in tickers:
            fig2.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns[t], mode="lines", name=t))
        fig2.update_layout(title="Cumulative Returns (%)", xaxis_title="Date", yaxis_title="Cumulative Return (%)",
                           template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.header("Company Info")
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            st.markdown(f"""
            <div style="background:#1a1a1a;padding:10px;border-radius:8px;margin-bottom:8px;">
            <h3 style="color:#ffffff;">{t}</h3>
            <p>Market Cap: {info.get('marketCap')}</p>
            <p>Trailing P/E: {info.get('trailingPE')}</p>
            <p>Forward P/E: {info.get('forwardPE')}</p>
            <p>Dividend Yield: {info.get('dividendYield')}</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.warning(f"Could not load info for {t}")

