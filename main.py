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
        df = yf.download(ticker, period="1y", group_by='ticker', auto_adjust=False)

        # Flatten columns if MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(col).strip() for col in df.columns.values]

        # Try to find an adjusted close column
        price_col = None
        for col in df.columns:
            if 'Adj Close' in col:
                price_col = col
                break
        if not price_col:
            for col in df.columns:
                if 'Close' in col:
                    price_col = col
                    st.warning(f"'Adj Close' not found for {ticker}, using '{price_col}' instead.")
                    break

        if not price_col:
            st.error(f"No suitable price column found for {ticker}. Columns: {df.columns}")
            st.stop()

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[price_col], mode="lines", name=ticker))
        fig.update_layout(title=f"{ticker} {price_col} Price", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error downloading data for {ticker}: {e}")
