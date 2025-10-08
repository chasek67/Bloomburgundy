# main.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

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
    .stTextInput>div>div>input {
        background-color: #1a1a1a;
        color: #e6e6e6;
    }
    .stSelectbox>div>div>div>span {
        color: #e6e6e6 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏠 Bloomberg Dashboard")

# Layout: 2 columns
col1, col2 = st.columns([2, 1])

# ------------------- LEFT COLUMN -------------------
with col1:
    st.header("Stock Price & Market Overview")

    ticker = st.text_input("Enter ticker symbol", "MSFT")

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

    if ticker:
        try:
            df_stock = yf.download(ticker, period="1y", group_by='ticker', auto_adjust=False)
            df_market = yf.download(market_ticker, period="1y", group_by='ticker', auto_adjust=False)

            if isinstance(df_stock.columns, pd.MultiIndex):
                df_stock.columns = [' '.join(col).strip() for col in df_stock.columns.values]
            if isinstance(df_market.columns, pd.MultiIndex):
                df_market.columns = [' '.join(col).strip() for col in df_market.columns.values]

            def get_price_column(df):
                for col in df.columns:
                    if 'Adj Close' in col:
                        return col
                for col in df.columns:
                    if 'Close' in col:
                        return col
                return None

            stock_price_col = get_price_column(df_stock)
            market_price_col = get_price_column(df_market)

            fig_stock = go.Figure()
            fig_stock.add_trace(go.Scatter(x=df_stock.index, y=df_stock[stock_price_col],
                                           mode="lines", name=ticker))
            fig_stock.update_layout(title=f"{ticker} Stock Price", xaxis_title="Date",
                                    yaxis_title="Price", template="plotly_dark")
            st.plotly_chart(fig_stock, use_container_width=True)

            fig_market = go.Figure()
            fig_market.add_trace(go.Scatter(x=df_market.index, y=df_market[market_price_col],
                                            mode="lines", name=market_choice))
            fig_market.update_layout(title=f"{market_choice} Market Overview", xaxis_title="Date",
                                     yaxis_title="Price", template="plotly_dark")
            st.plotly_chart(fig_market, use_container_width=True)

        except Exception as e:
            st.error(f"Error downloading data: {e}")

# ------------------- RIGHT COLUMN -------------------
with col2:
    st.header("Stock Info & Indicators")

    if ticker:
        try:
            df_stock_info = yf.Ticker(ticker).info

            st.subheader("Key Metrics")
            metrics = {
                "Previous Close": df_stock_info.get("previousClose"),
                "Open": df_stock_info.get("open"),
                "Day High": df_stock_info.get("dayHigh"),
                "Day Low": df_stock_info.get("dayLow"),
                "Market Cap": df_stock_info.get("marketCap"),
                "PE Ratio": df_stock_info.get("trailingPE"),
                "Dividend Yield": df_stock_info.get("dividendYield"),
            }

            for k, v in metrics.items():
                st.metric(label=k, value=v)

            if "Volume" in df_stock.columns:
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(x=df_stock.index, y=df_stock["Volume"], name="Volume"))
                fig_vol.update_layout(title="Daily Trading Volume", xaxis_title="Date",
                                      yaxis_title="Volume", template="plotly_dark")
                st.plotly_chart(fig_vol, use_container_width=True)

        except Exception as e:
            st.error(f"Error fetching stock info: {e}")

