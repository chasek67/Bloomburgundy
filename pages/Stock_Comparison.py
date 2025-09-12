import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

st.title("Stock Comparison")

# Input tickers as comma-separated
tickers_input = st.text_input("Enter tickers separated by commas", "AAPL,MSFT,GOOGL")
tickers = [t.strip().upper() for t in tickers_input.split(",")]

if tickers:
    try:
        # Download multiple tickers
        raw = yf.download(tickers, period="1y", group_by='ticker', auto_adjust=False)

        # Flatten MultiIndex columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [' '.join(col).strip() for col in raw.columns.values]

        # Extract adjusted close columns for all tickers
        adj_close_cols = [col for col in raw.columns if 'Adj Close' in col]
        if not adj_close_cols:
            # Fallback to 'Close' if no Adj Close
            adj_close_cols = [col for col in raw.columns if 'Close' in col]
            st.warning("No 'Adj Close' found, using 'Close' instead.")

        data = raw[adj_close_cols]

        # Rename columns to just the ticker names
        data.columns = [col.split()[0] for col in data.columns]

        # Plot
        fig = go.Figure()
        for ticker in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[ticker], mode="lines", name=ticker))
        fig.update_layout(title="Stock Comparison", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error downloading or plotting data: {e}")
