import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

st.title("📈 Market Overview")

# Example ticker input
ticker = st.text_input("Enter a ticker (e.g. AAPL, MSFT)", "AAPL")

if ticker:
    df = yf.download(ticker, period="1y")

    if df.empty:
        st.error(f"No data found for ticker: {ticker}")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode="lines", name=ticker))
        st.plotly_chart(fig, use_container_width=True)
