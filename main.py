import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.title("Bloomberg Dashboard")

# Input ticker
ticker = st.text_input("Enter ticker symbol", "MSFT")

if ticker:
    try:
        # Download data
        df = yf.download(ticker, period="1y")

        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(1)

        # Prefer 'Adj Close', fallback to 'Close'
        if "Adj Close" in df.columns:
            price_col = "Adj Close"
        elif "Close" in df.columns:
            price_col = "Close"
            st.warning(f"'Adj Close' not found for {ticker}, using 'Close' instead.")
        else:
            st.error(f"No suitable price column found for {ticker}.")
            st.stop()

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[price_col], mode="lines", name=ticker))
        fig.update_layout(title=f"{ticker} {price_col} Price", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error downloading data for {ticker}: {e}")
