import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

st.title("📊 Stock Comparison")

tickers = st.text_input("Enter tickers separated by commas", "AAPL,MSFT,GOOGL")
tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

if tickers:
    raw = yf.download(tickers, period="1y")

    if raw.empty:
        st.error("No data found. Please check your tickers.")
    else:
        # Handle single vs multiple tickers
        if isinstance(raw.columns, pd.MultiIndex):
            data = raw["Adj Close"]
        else:
            data = raw[["Adj Close"]]

        st.line_chart(data)
