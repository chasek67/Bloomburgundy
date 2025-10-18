# main.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# ---------- Page Setup ----------
st.set_page_config(layout="wide", page_title="Bloomberg Dashboard")

# ---------- Global dark styling (sidebar + widgets + tables) ----------
st.markdown("""
<style>
/* App background + default text */
.stApp {
  background-color: #0e0e0e !important;
  color: #e6e6e6 !important;
}

/* Sidebar container + text */
[data-testid="stSidebar"] {
  background-color: #111 !important;
  color: #eaeaea !important;
  border-right: 1px solid #222 !important;
}
[data-testid="stSidebar"] * {
  color: #eaeaea !important;
}

/* Sidebar nav links (left page selector) */
[data-testid="stSidebarNav"] a {
  color: #cfe1ff !important;
  border-radius: 8px !important;
  padding: 6px 10px !important;
}
[data-testid="stSidebarNav"] a:hover {
  background: #1d1f23 !important;
}

/* Inputs (fix white text on white bg) */
.stTextInput input,
.stNumberInput input,
.stDateInput input {
  background-color: #1a1a1a !important;
  color: #e6e6e6 !important;
  border: 1px solid #333 !important;
}
div[data-baseweb="select"] > div {  /* selectbox/multiselect wrapper */
  background-color: #1a1a1a !important;
  color: #e6e6e6 !important;
  border-color: #333 !important;
}
div[data-baseweb="tag"] {            /* multiselect pills */
  background-color: #222 !important;
  color: #e6e6e6 !important;
  border-color: #333 !important;
}

/* Slider labels/track */
.stSlider > div > div > div {
  color: #e6e6e6 !important;
}

/* Buttons */
.stButton button {
  background: #242424 !important;
  color: #e6e6e6 !important;
  border: 1px solid #333 !important;
  border-radius: 10px !important;
}
.stButton button:hover {
  background: #2e2e2e !important;
}

/* Metrics + tables contrast */
.stMetric { color: #e6e6e6 !important; }
.stDataFrame, .stTable { color: #e6e6e6 !important; }
.stDataFrame thead tr th, .stDataFrame tbody tr td {
  background-color: #141414 !important;
  color: #e6e6e6 !important;
  border-color: #2a2a2a !important;
}

/* Plotly canvas stay dark */
.js-plotly-plot .plotly, .plotly .main-svg {
  background-color: #000 !important;
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
            # Keep it simple: no group_by for single tickers
            df_stock = yf.download(ticker, period="1y", auto_adjust=False, progress=False)
            df_market = yf.download(market_ticker, period="1y", auto_adjust=False, progress=False)

            # Flatten columns if MultiIndex
            for df in (df_stock, df_market):
                if df.empty:
                    st.error("Could not fetch data from Yahoo Finance. Try another ticker or check your connection.")
                elif isinstance(df.columns, pd.MultiIndex):
                    df.columns = [' '.join(col).strip() for col in df.columns.values]

            # Choose a usable price column
            def get_price_column(df: pd.DataFrame) -> str | None:
                for key in ("Adj Close", "Close"):
                    for col in df.columns:
                        if key in col:
                            return col
                # fallback: first numeric
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        return col
                return None

            stock_price_col = get_price_column(df_stock)
            market_price_col = get_price_column(df_market)

            if stock_price_col is None or market_price_col is None:
                st.error("Could not find valid price columns for one of the datasets.")
            else:
                # ---- Stock chart
                fig_stock = go.Figure()
                fig_stock.add_trace(go.Scatter(
                    x=df_stock.index, y=df_stock[stock_price_col],
                    mode="lines", name=ticker,
                    line=dict(color="#00FFFF", width=2)
                ))
                fig_stock.update_layout(
                    title=f"{ticker} Stock Price",
                    xaxis_title="Date", yaxis_title="Price",
                    template="plotly_dark", paper_bgcolor="#111111", plot_bgcolor="#111111",
                    font=dict(color="white")
                )
                st.plotly_chart(fig_stock, use_container_width=True)

                # ---- Market chart
                fig_market = go.Figure()
                fig_market.add_trace(go.Scatter(
                    x=df_market.index, y=df_market[market_price_col],
                    mode="lines", name=market_choice,
                    line=dict(color="#FFAA00", width=2)
                ))
                fig_market.update_layout(
                    title=f"{market_choice} Price Overview",
                    xaxis_title="Date", yaxis_title="Price",
                    template="plotly_dark", paper_bgcolor="#111111", plot_bgcolor="#111111",
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

            # Volume chart if present
            if "Volume" in df_stock.columns:
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(
                    x=df_stock.index, y=df_stock["Volume"],
                    name="Volume", marker_color="#00FFFF"
                ))
                fig_vol.update_layout(
                    title="Daily Trading Volume",
                    xaxis_title="Date", yaxis_title="Volume",
                    template="plotly_dark", paper_bgcolor="#111111", plot_bgcolor="#111111",
                    font=dict(color="white")
                )
                st.plotly_chart(fig_vol, use_container_width=True)

        except Exception as e:
            st.error(f"Error fetching stock info: {e}")
