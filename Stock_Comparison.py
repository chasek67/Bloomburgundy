import streamlit as st
import yfinance as yf
import numpy as np
import plotly.express as px

st.title("📈 Stock Comparison")

tickers = st.multiselect("Choose tickers:", ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"], default=["AAPL","MSFT"])

if tickers:
    data = yf.download(tickers, period="1y")["Adj Close"]

    # Normal scale plot
    st.subheader("Raw Prices")
    st.line_chart(data)

    # Log scale
    st.subheader("Log-Scaled Prices")
    log_data = np.log(data / data.iloc[0])  # log returns
    st.line_chart(log_data)
