import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Stock Comparison", layout="wide")

st.markdown("""
<style>
  [data-testid="stSidebar"] { background-color: #111 !important; color: #eee !important; }
  [data-testid="stSidebar"] * { color: #eee !important; }
  [data-testid="stSidebarNav"] a { color: #cfe1ff !important; }
  [data-testid="stSidebarNav"] a:hover { background: #1d1f23 !important; }
  .stApp { background-color: #0e0e0e; color: #e6e6e6; }
  .metric-card { background:#1a1a1a;padding:12px;border-radius:12px;border:1px solid #333; }
  .status { font-size:0.9rem; color:#ddd; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Stock Comparison")

with st.sidebar:
    st.header("Controls")
    tickers_input = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN")
    horizon = st.selectbox("Time Horizon", ["3M","6M","1Y","2Y","5Y","YTD","Max"], index=2)
    sampling = st.selectbox("Sampling Frequency", ["Daily","Weekly","Monthly"], index=0)
    rebased = st.checkbox("Normalize to 100 (rebased)", True)
    log_scale = st.checkbox("Log scale", False)
    extras = st.multiselect("Extras", ["Correlation Heatmap","30D Rolling Volatility","Return Distribution"],
                            default=["Correlation Heatmap"])

def _period(h: str) -> str:
    return {"3M":"3mo","6M":"6mo","1Y":"1y","2Y":"2y","5Y":"5y","YTD":"ytd","Max":"max"}[h]

def _to_freq(label: str) -> str:
    return {"Daily":"1D","Weekly":"1W","Monthly":"1M"}[label]

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.stop()

# ---- same robust loader as Portfolio page ----
DATA_DIR = Path("Data")

def _read_csv_robust(path, header_rows):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        for eng in ("python", "c"):
            try:
                return pd.read_csv(path, index_col=0, header=header_rows,
                                   parse_dates=True, encoding=enc, engine=eng,
                                   on_bad_lines="skip")
            except Exception:
                continue
    return None

def _load_from_combined_csv(tickers):
    p = DATA_DIR / "prices_combined.csv"
    if not p.exists(): return None
    df = _read_csv_robust(p, header_rows=[0,1])
    if df is not None and isinstance(df.columns, pd.MultiIndex):
        cols = []
        for t in tickers:
            if ("Adj Close", t) in df.columns:
                cols.append(df[("Adj Close", t)].rename(t))
            elif ("Close", t) in df.columns:
                cols.append(df[("Close", t)].rename(t))
        if cols:
            return pd.concat(cols, axis=1).sort_index().dropna(how="all")
    df2 = _read_csv_robust(p, header_rows=0)
    if df2 is not None:
        df2 = df2.rename(columns={c:c.upper() for c in df2.columns})
        keep = [t for t in [x.upper() for x in tickers] if t in df2.columns]
        if keep:
            return df2[keep].sort_index().dropna(how="all")
    return None

def _load_from_per_ticker_csvs(tickers):
    frames = []
    for t in tickers:
        p = DATA_DIR / f"{t}.csv"
        if not p.exists(): return None
        df = _read_csv_robust(p, header_rows=[0,1])
        if df is not None and isinstance(df.columns, pd.MultiIndex):
            if ("Adj Close", t) in df.columns:
                frames.append(df[("Adj Close", t)].rename(t)); continue
            if ("Close", t) in df.columns:
                frames.append(df[("Close", t)].rename(t)); continue
        df2 = _read_csv_robust(p, header_rows=0)
        if df2 is None: return None
        col = "Adj Close" if "Adj Close" in df2.columns else ("Close" if "Close" in df2.columns else None)
        if col is None: return None
        frames.append(df2[col].rename(t))
    return pd.concat(frames, axis=1).sort_index().dropna(how="all")

@st.cache_data(show_spinner=False)
def load_prices(tickers, period):
    try:
        df = yf.download(tickers, period=period, auto_adjust=True, group_by="ticker", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.get_level_values(0): df = df["Close"]
            elif "Adj Close" in df.columns.get_level_values(0): df = df["Adj Close"]
            else: raise KeyError("No Close/Adj Close")
        else:
            if "Close" in df.columns: df = df[["Close"]]
            elif "Adj Close" in df.columns: df = df[["Adj Close"]]
            df.columns = [tickers[0]]
        df = df.dropna(how="all")
        if df.empty: raise ValueError("Empty Yahoo")
        return df, "Yahoo Finance"
    except Exception:
        pass

    local = _load_from_combined_csv(tickers)
    if local is not None and not local.empty: return local, "Data/prices_combined.csv"

    local2 = _load_from_per_ticker_csvs(tickers)
    if local2 is not None and not local2.empty: return local2, "Data/<ticker>.csv files"

    return None, None

prices_raw, source = load_prices(tickers, _period(horizon))
if prices_raw is None:
    st.error("No data from Yahoo or local CSVs. Verify tickers and that Data/prices_combined.csv exists.")
    st.stop()

with st.expander("Data source & columns", expanded=False):
    st.markdown(f"<div class='status'>Source: <b>{source}</b></div>", unsafe_allow_html=True)
    st.write("Columns loaded:", list(prices_raw.columns))
    st.write("Rows:", len(prices_raw))

# Resample
prices = prices_raw.resample(_to_freq(sampling)).last().dropna(how="all")
rets = prices.pct_change().fillna(0.0)
cum = (1 + rets).cumprod()

palette = px.colors.qualitative.Bold + px.colors.qualitative.Plotly + px.colors.qualitative.D3
color_map = {t: palette[i % len(palette)] for i, t in enumerate(prices.columns)}

left, right = st.columns([2,1])

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
