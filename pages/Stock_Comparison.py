import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Dark theme
st.set_page_config(layout="wide", page_title="Stock Comparison")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e0e0e;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #1a1a1a;
        color: white;
        border: 1px solid #444;
    }
    .stSelectbox>div>div>div>span {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚀 Stock Comparison Dashboard")

col1, col2 = st.columns([2, 1])

# ------------------- LEFT COLUMN -------------------
with col1:
    tickers_input = st.text_input("Enter tickers separated by commas", "AAPL,MSFT,GOOGL")
    tickers = [t.strip().upper() for t in tickers_input.split(",")]

    if tickers:
        try:
            raw = yf.download(tickers, period="1y", group_by='ticker', auto_adjust=False)

            # Flatten MultiIndex columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [' '.join(col).strip() for col in raw.columns.values]

            # Extract adjusted close columns
            adj_close_cols = [col for col in raw.columns if 'Adj Close' in col]
            if not adj_close_cols:
                adj_close_cols = [col for col in raw.columns if 'Close' in col]
                st.warning("No 'Adj Close' found, using 'Close' instead.")

            data = raw[adj_close_cols]
            data.columns = [col.split()[0] for col in data.columns]

            # --- Chart 1: Normal Prices ---
            fig1 = go.Figure()
            for ticker in data.columns:
                fig1.add_trace(go.Scatter(x=data.index, y=data[ticker], mode="lines", name=ticker))
            fig1.update_layout(
                title="Stock Prices",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_dark",
                paper_bgcolor="black",
                plot_bgcolor="black",
                font=dict(color="white")
            )
            st.plotly_chart(fig1, use_container_width=True)

            # --- Chart 2: Log Prices ---
            fig2 = go.Figure()
            for ticker in data.columns:
                fig2.add_trace(go.Scatter(
                    x=data.index,
                    y=np.log(data[ticker]),  # <-- use numpy directly
                    mode="lines",
                    name=ticker
                ))
            fig2.update_layout(
                title="Log of Stock Prices (Normalized)",
                xaxis_title="Date",
                yaxis_title="Log(Price)",
                template="plotly_dark",
                paper_bgcolor="black",
                plot_bgcolor="black",
                font=dict(color="white")
            )
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"Error downloading or plotting data: {e}")

# ------------------- RIGHT COLUMN -------------------
with col2:
    st.header("Company Comparison")
    if tickers:
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                st.subheader(f"{ticker} Info", anchor=None)

                # Format large numbers nicely
                def format_number(n):
                    if n is None:
                        return "N/A"
                    elif n >= 1_000_000_000:
                        return f"${n/1_000_000_000:,.2f} B"
                    elif n >= 1_000_000:
                        return f"${n/1_000_000:,.2f} M"
                    else:
                        return f"${n:,.0f}"

                metrics = {
                    "Market Cap": format_number(info.get("marketCap")),
                    "P/E Ratio": info.get("trailingPE"),
                    "Forward P/E": info.get("forwardPE"),
                    "Dividend Yield": info.get("dividendYield"),
                    "Previous Close": info.get("previousClose"),
                    "Open": info.get("open"),
                    "Day High": info.get("dayHigh"),
                    "Day Low": info.get("dayLow")
                }

                for k, v in metrics.items():
                    st.metric(label=k, value=v, delta_color="normal")  # delta_color=normal keeps text bright

            except Exception as e:
                st.warning(f"Could not fetch info for {ticker}: {e}")
