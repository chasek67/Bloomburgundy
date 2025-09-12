import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.title("Bloomberg Dashboard")

# Input ticker
ticker = st.text_input("Enter ticker symbol", "MSFT")

if ticker:
    # Download data
    df = yf.download(ticker, period="1y")

    # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)

    # Check if 'Adj Close' exists
    if "Adj Close" not in df.columns:
        st.error(f"'Adj Close' column not found in {ticker} data!")
    else:
        # Plot adjusted close
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode="lines", name=ticker))
        fig.update_layout(title=f"{ticker} Adjusted Close Price", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)
