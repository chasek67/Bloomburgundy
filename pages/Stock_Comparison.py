import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ---------------- Page Setup & Theme ----------------
st.set_page_config(page_title="Stock Comparison", layout="wide")

st.markdown("""
<style>
  /* Sidebar readable + dark */
  [data-testid="stSidebar"] {
    background-color: #111 !important;
    color: #eee !important;
  }
  [data-testid="stSidebar"] * { color: #eee !important; }
  [data-testid="stSidebarNav"] a { color: #cfe1ff !important; }
  [data-testid="stSidebarNav"] a:hover { background: #1d1f23 !important; }
  /* App background */
  .stApp { background-color: #0e0e0e; color: #e6e6e6; }
  .metric-card { background:#1a1a1a;padding:12px;border-radius:12px;border:1px solid #333; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Stock Comparison")

# ---------------- Controls ----------------
with st.sidebar:
    st.header("Controls")
    tickers_input = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN")
    horizon = st.selectbox("Time Horizon", ["3M", "6M", "1Y", "2Y", "5Y", "YTD", "Max"], index=2)
    sampling = st.selectbox("Sampling Frequency", ["Daily", "Weekly", "Monthly"], index=0)
    rebased = st.checkbox("Normalize to 100 (rebased)", True)
    log_scale = st.checkbox("Log scale", False)
    extras = st.multiselect("Extras", ["Correlation Heatmap", "30D Rolling Volatility", "Return Distribution"],
                            default=["Correlation Heatmap"])

def _period(h: str) -> str:
    return {"3M":"3mo","6M":"6mo","1Y":"1y","2Y":"2y","5Y":"5y","YTD":"ytd","Max":"max"}[h]

def _to_freq(label: str) -> str:
    return {"Daily":"1D","Weekly":"1W","Monthly":"1M"}[label]

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.stop()

# ---------------- Robust Data Loader ----------------
DATA_DIR = Path("Data")

def load_from_combined_csv(tickers, data_dir=DATA_DIR):
    p = data_dir / "prices_combined.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0, header=[0,1], parse_dates=True)
    cols = []
    for t in tickers:
        if ("Adj Close", t) in df.columns:
            cols.append(df[("Adj Close", t)].rename(t))
        elif ("Close", t) in df.columns:
            cols.append(df[("Close", t)].rename(t))
    if not cols:
        return None
    return pd.concat(cols, axis=1).sort_index().dropna(how="all")

def load_from_per_ticker_csvs(tickers, data_dir=DATA_DIR):
    series = []
    for t in tickers:
        p = data_dir / f"{t}.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p, index_col=0, header=[0,1], parse_dates=True)
        if ("Adj Close", t) in df.columns:
            series.append(df[("Adj Close", t)].rename(t))
        elif ("Close", t) in df.columns:
            series.append(df[("Close", t)].rename(t))
        else:
            return None
    return pd.concat(series, axis=1).sort_index().dropna(how="all")

@st.cache_data(show_spinner=False)
def load_prices(tickers, period):
    # 1) Try Yahoo (auto_adjust=True → use "Close")
    try:
        df = yf.download(tickers, period=period, auto_adjust=True, group_by="ticker", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            # Multi-ticker: take "Close" level only
            if "Close" in df.columns.get_level_values(0):
                df = df["Close"]
            elif "Adj Close" in df.columns.get_level_values(0):
                df = df["Adj Close"]
            else:
                raise KeyError("Neither Close nor Adj Close in Yahoo result.")
        else:
            # Single ticker
            if "Close" in df.columns:
                df = df[["Close"]]
            elif "Adj Close" in df.columns:
                df = df[["Adj Close"]]
            df.columns = [tickers[0]]
        df = df.dropna(how="all")
        if df.empty:
            raise ValueError("Empty Yahoo data")
        return df
    except Exception:
        pass

    # 2) Fallback to combined CSV
    local = load_from_combined_csv(tickers)
    if local is not None and not local.empty:
        return local

    # 3) Fallback to per-ticker CSVs
    local2 = load_from_per_ticker_csvs(tickers)
    if local2 is not None and not local2.empty:
        return local2

    raise RuntimeError("No data from Yahoo or local CSVs. Add CSVs under Data/ or check tickers.")

# ---------------- Fetch & Prep ----------------
try:
    prices_raw = load_prices(tickers, _period(horizon))
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

prices = prices_raw.resample(_to_freq(sampling)).last().dropna(how="all")
rets = prices.pct_change().fillna(0.0)
cum = (1 + rets).cumprod()

palette = px.colors.qualitative.Bold + px.colors.qualitative.Plotly + px.colors.qualitative.D3
color_map = {t: palette[i % len(palette)] for i, t in enumerate(prices.columns)}

left, right = st.columns([2,1])

# ---------------- Charts ----------------
with left:
    if rebased:
        base = cum.iloc[0]
        rebased100 = cum.div(base) * 100
        fig1 = go.Figure()
        for t in prices.columns:
            fig1.add_trace(go.Scatter(x=rebased100.index, y=rebased100[t], name=t,
                                      mode="lines", line=dict(width=2.2, color=color_map[t])))
        fig1.update_layout(title="Normalized Price (Rebased to 100)", xaxis_title="Date",
                           yaxis_title="Index (100 = start)", template="plotly_dark",
                           paper_bgcolor="#000", plot_bgcolor="#000",
                           legend=dict(orientation="h", y=-0.2))
    else:
        fig1 = go.Figure()
        for t in prices.columns:
            fig1.add_trace(go.Scatter(x=prices.index, y=prices[t], name=t,
                                      mode="lines", line=dict(width=2.2, color=color_map[t])))
        fig1.update_layout(title="Adjusted Close Price", xaxis_title="Date",
                           yaxis_title="Price", template="plotly_dark",
                           paper_bgcolor="#000", plot_bgcolor="#000",
                           legend=dict(orientation="h", y=-0.2))
    if log_scale:
        fig1.update_yaxes(type="log")
    st.plotly_chart(fig1, use_container_width=True, height=520)

    cumret = cum / cum.iloc[0] - 1
    fig2 = go.Figure()
    for t in prices.columns:
        fig2.add_trace(go.Scatter(x=cumret.index, y=cumret[t]*100, name=t,
                                  mode="lines", line=dict(width=2.2, color=color_map[t])))
    fig2.update_layout(title="Cumulative Return (%)", xaxis_title="Date",
                       yaxis_title="Cumulative Return (%)", template="plotly_dark",
                       paper_bgcolor="#000", plot_bgcolor="#000",
                       legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig2, use_container_width=True, height=520)

with right:
    st.subheader("Snapshot")
    ann_factor = 252 if _to_freq(sampling) == "1D" else (52 if _to_freq(sampling) == "1W" else 12)

    def sharpe_from_prices(price_series, rf=0.0):
        r = price_series.pct_change().dropna()
        return np.nan if r.std()==0 else np.sqrt(ann_factor) * (r.mean() - rf/ann_factor) / r.std()

    rows = []
    for t in prices.columns:
        total = prices[t].iloc[-1]/prices[t].iloc[0]-1
        s = sharpe_from_prices(prices[t])
        rows.append([t, f"{total*100:,.2f}%", f"{s:,.2f}",
                     f"{rets[t].max()*100:,.2f}%", f"{rets[t].min()*100:,.2f}%"])
    st.dataframe(pd.DataFrame(rows, columns=["Ticker","Cumulative Return","Sharpe","Best Day","Worst Day"]),
                 use_container_width=True, height=260)

    if "Correlation Heatmap" in extras:
        corr = rets.corr()
        figc = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis")
        figc.update_layout(title="Return Correlation", template="plotly_dark",
                           paper_bgcolor="#000", plot_bgcolor="#000")
        st.plotly_chart(figc, use_container_width=True)

    if "30D Rolling Volatility" in extras:
        window = 30 if _to_freq(sampling)=="1D" else (26 if _to_freq(sampling)=="1W" else 6)
        roll = rets.rolling(window).std() * np.sqrt(ann_factor)
        figv = go.Figure()
        for t in prices.columns:
            figv.add_trace(go.Scatter(x=roll.index, y=roll[t], name=t,
                                      mode="lines", line=dict(color=color_map[t])))
        figv.update_layout(title=f"Rolling Volatility (annualized, window={window})",
                           template="plotly_dark", paper_bgcolor="#000", plot_bgcolor="#000",
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(figv, use_container_width=True)

    if "Return Distribution" in extras:
        from plotly.subplots import make_subplots
        figh = make_subplots(rows=1, cols=1)
        for t in prices.columns:
            figh.add_histogram(x=(rets[t]*100), name=t, opacity=0.5, marker_color=color_map[t])
        figh.update_layout(barmode="overlay", title="Daily Return Distribution (%)",
                           template="plotly_dark", paper_bgcolor="#000", plot_bgcolor="#000",
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(figh, use_container_width=True)

st.download_button("⬇️ Download returns CSV", data=rets.to_csv().encode(),
                   file_name="returns.csv", mime="text/csv")
