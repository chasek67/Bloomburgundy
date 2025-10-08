# main.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# ---------- Page Setup ----------
st.set_page_config(layout="wide", page_title="Bloomberg Dashboard")

# ---------- Dark theme + readable text ----------
st.markdown("""
<style>
    .stApp {
        background-color: #0e0e0e;
        color: #e6e6e6;
    }
    h1, h2, h3, h4, h5, h6, label, p, span, div, .stMarkdown {
        color: #e6e6e6 !important;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>div>span {
        background-color: #1a1a1a;
        color: #e6e6e6 !important;
    }
    .stMetric {
        color: #e6e6e6 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏠 Bloomberg Dashboard")

# ---------- Layout ----------
col1, col2 = st.columns([2, 1])

# ------------------- LEFT COLUMN -------------------
with col1:
    st.header("Stock Price & Market Overview")

    # Input ticker
    ticker = st.text_input("Enter ticker symbol", "MSFT")

    # Select international market
    market_options = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "FTSE 100": "^FTSE",
        "DAX": "^GDAXI",
        "Nikkei 225": "^N225",
        "Hang Seng": "^HSI"
    }
    market_choice = st.selectbox("Select Market Index", list(market_options.keys()))
    market_ticker = market_options[market_choice]

    # Download stock and market data
    if ticker:
        try:
            # ---- Fix: remove group_by (can break single tickers)
            df_stock = yf.download(ticker, period="1y", auto_adjust=False)
            df_market = yf.download(market_ticker, period="1y", auto_adjust=False)

            # Flatten columns if MultiIndex
            for df in [df_stock, df_market]:
                if df.empty:
                    st.error("Could not fetch data from Yahoo Finance. Try another ticker or check your connection.")
                elif isinstance(df.columns, pd.MultiIndex):
                    df.columns = [' '.join(col).strip() for col in df.columns.values]

            # ---- Robust function to find a valid price column
            def get_price_column(df):
                for col in df.columns:
                    if 'Adj Close' in col:
                        return col
                for col in df.columns:
                    if 'Close' in col:
                        return col
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        return col
                return None

            stock_price_col = get_price_column(df_stock)
            market_price_col = get_price_column(df_market)

            if stock_price_col is None or market_price_col is None:
                st.error("Could not find valid price columns for one of the datasets.")
            else:
                # ---- Plot stock chart
                fig_stock = go.Figure()
                fig_stock.add_trace(go.Scatter(
                    x=df_stock.index,
                    y=df_stock[stock_price_col],
                    mode="lines",
                    name=ticker,
                    line=dict(color="#00FFFF", width=2)
                ))
                fig_stock.update_layout(
                    title=f"{ticker} Stock Price",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    template="plotly_dark",
                    paper_bgcolor="black",
                    plot_bgcolor="black",
                    font=dict(color="white")
                )
                st.plotly_chart(fig_stock, use_container_width=True)

                # ---- Plot market chart
                fig_market = go.Figure()
                fig_market.add_trace(go.Scatter(
                    x=df_market.index,
                    y=df_market[market_price_col],
                    mode="lines",
                    name=market_choice,
                    line=dict(color="#FFAA00", width=2)
                ))
                fig_market.update_layout(
                    title=f"{market_choice} Price Overview",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    template="plotly_dark",
                    paper_bgcolor="black",
                    plot_bgcolor="black",
                    font=dict(color="white")
                )
                st.plotly_chart(fig_market, use_container_width=True)

        except Exception as e:
            st.error(f"Error downloading or plotting data: {e}")

# ------------------- RIGHT COLUMN -------------------
with col2:
    st.header("Stock Info & Indicators")

    if ticker:
        try:
            info = yf.Ticker(ticker).info

            # ---- Display key metrics safely
            st.subheader("Key Metrics")

            def safe_val(val):
                if val is None:
                    return "N/A"
                if isinstance(val, (float, int)):
                    return f"{val:,.2f}"
                return str(val)

            metrics = {
                "Previous Close": info.get("previousClose"),
                "Open": info.get("open"),
                "Day High": info.get("dayHigh"),
                "Day Low": info.get("dayLow"),
                "Market Cap": info.get("marketCap"),
                "PE Ratio": info.get("trailingPE"),
                "Dividend Yield": info.get("dividendYield"),
            }

            for k, v in metrics.items():
                st.metric(label=k, value=safe_val(v))

            # ---- Optional: Volume chart
            if "Volume" in df_stock.columns:
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(
                    x=df_stock.index,
                    y=df_stock["Volume"],
                    name="Volume",
                    marker_color="#00FFFF"
                ))
                fig_vol.update_layout(
                    title="Daily Trading Volume",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    template="plotly_dark",
                    paper_bgcolor="black",
                    plot_bgcolor="black",
                    font=dict(color="white")
                )
                st.plotly_chart(fig_vol, use_container_width=True)

        except Exception as e:
            st.error(f"Error fetching stock info: {e}")
