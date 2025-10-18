import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(page_title="Portfolio Builder", layout="wide")

# --- Local page CSS (kept here too, in case main isn't loaded) ---
st.markdown("""
<style>
  /* Sidebar container */
  [data-testid="stSidebar"] {
    background-color: #111 !important;
    color: #eee !important;
  }
  [data-testid="stSidebar"] * { color: #eee !important; }
  [data-testid="stSidebarNav"] a { color: #cfe1ff !important; }
  [data-testid="stSidebarNav"] a:hover { background: #1d1f23 !important; }
  .stApp { background-color: #0e0e0e; color: #e6e6e6; }

  /* === Make text inputs readable (black text, light background) === */
  .stTextInput input,
  .stNumberInput input,
  .stSelectbox div[data-baseweb="select"],
  .stDateInput input {
      background-color: #f2f2f2 !important;   /* light gray background */
      color: #111 !important;                 /* black text */
      border: 1px solid #444 !important;
      border-radius: 6px !important;
  }
  /* Dropdown text color */
  div[data-baseweb="select"] span {
      color: #111 !important;
  }

  /* Keep other elements dark */
  .stButton button { background:#242424 !important; color:#e6e6e6 !important; }
  .metric-card { background:#1a1a1a; padding:12px; border-radius:12px; border:1px solid #333; }
  .status { font-size:0.9rem; color:#ddd; }
</style>
""", unsafe_allow_html=True)


st.title("💼 Portfolio Builder")

# ---------------- Controls ----------------
with st.sidebar:
    st.header("Portfolio Inputs")
    c1, c2, c3 = st.columns(3)
    with c1: t1 = st.text_input("Asset 1", "AAPL").strip().upper()
    with c2: t2 = st.text_input("Asset 2", "MSFT").strip().upper()
    with c3: t3 = st.text_input("Asset 3", "GOOGL").strip().upper()
    tickers = [t for t in [t1, t2, t3] if t]

    st.markdown("---")
    st.subheader("Weights")
    w1 = st.slider(f"{t1 or 'Asset 1'} weight (%)", 0, 100, 33, 1)
    w2 = st.slider(f"{t2 or 'Asset 2'} weight (%)", 0, 100, 33, 1)
    auto_third = st.checkbox("Auto-balance third weight so total = 100%", True)
    if auto_third:
        w3 = max(0, 100 - w1 - w2)
        st.write(f"{t3 or 'Asset 3'} weight (%): **{w3}**")
    else:
        w3 = st.slider(f"{t3 or 'Asset 3'} weight (%)", 0, 100, 34, 1)

    normalize = st.checkbox("Normalize weights to sum 100%", True)
    rf = st.number_input("Risk-free rate (annual, %)", value=0.0, step=0.25)
    horizon = st.selectbox("Time Horizon", ["6M", "1Y", "3Y", "5Y", "YTD", "Max"], index=1)
    show_table = st.checkbox("Show performance table", True)

def _period(h: str) -> str:
    return {"6M":"6mo","1Y":"1y","3Y":"3y","5Y":"5y","YTD":"ytd","Max":"max"}[h]

if len(tickers) < 3:
    st.info("Please provide three tickers.")
    st.stop()

weights_pct = np.array([w1, w2, w3], dtype=float)
if normalize:
    s = weights_pct.sum()
    if s == 0:
        st.error("Weights sum to zero. Adjust sliders.")
        st.stop()
    weights_pct = 100 * weights_pct / s
w = weights_pct / 100.0

# ---------------- Robust Data Loader with Status ----------------
DATA_DIR = Path("Data")

def _read_csv_robust(path, header_rows):
    # Try multiple engines/encodings, skip bad lines
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
    if not p.exists():
        return None
    # Try yfinance MultiIndex first
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
    # Try simple wide: Date,AAPL,MSFT,...
    df2 = _read_csv_robust(p, header_rows=0)
    if df2 is not None:
        keep = [c for c in df2.columns if c.upper() in [x.upper() for x in tickers]]
        if keep:
            df2 = df2.rename(columns={c:c.upper() for c in df2.columns})
            return df2[[t.upper() for t in tickers if t.upper() in df2.columns]].sort_index().dropna(how="all")
    return None

def _load_from_per_ticker_csvs(tickers):
    frames = []
    for t in tickers:
        p = DATA_DIR / f"{t}.csv"
        if not p.exists():
            return None
        df = _read_csv_robust(p, header_rows=[0,1])
        if df is not None and isinstance(df.columns, pd.MultiIndex):
            if ("Adj Close", t) in df.columns:
                frames.append(df[("Adj Close", t)].rename(t)); continue
            if ("Close", t) in df.columns:
                frames.append(df[("Close", t)].rename(t)); continue
        # Fallback: simple single-level (Date,Adj Close)
        df2 = _read_csv_robust(p, header_rows=0)
        if df2 is None:
            return None
        col = "Adj Close" if "Adj Close" in df2.columns else ("Close" if "Close" in df2.columns else None)
        if col is None:
            return None
        frames.append(df2[col].rename(t))
    return pd.concat(frames, axis=1).sort_index().dropna(how="all")

@st.cache_data(show_spinner=False)
def load_prices(tickers, period):
    # 1) Yahoo first
    try:
        df = yf.download(tickers, period=period, auto_adjust=True, group_by="ticker", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.get_level_values(0):
                df = df["Close"]
            elif "Adj Close" in df.columns.get_level_values(0):
                df = df["Adj Close"]
            else:
                raise KeyError("No Close/Adj Close in Yahoo result.")
        else:
            if "Close" in df.columns:
                df = df[["Close"]]
            elif "Adj Close" in df.columns:
                df = df[["Adj Close"]]
            df.columns = [tickers[0]]
        df = df.dropna(how="all")
        if df.empty:
            raise ValueError("Empty Yahoo data")
        return df, "Yahoo Finance"
    except Exception:
        pass
    # 2) Combined CSV
    local = _load_from_combined_csv(tickers)
    if local is not None and not local.empty:
        return local, "Data/prices_combined.csv"
    # 3) Per-ticker CSVs
    local2 = _load_from_per_ticker_csvs(tickers)
    if local2 is not None and not local2.empty:
        return local2, "Data/<ticker>.csv files"
    # Fail
    return None, None

prices, source = load_prices(tickers, _period(horizon))
if prices is None:
    st.error("No data from Yahoo or local CSVs. Verify tickers and that Data/prices_combined.csv exists.")
    st.stop()

# Status panel
with st.expander("Data source & columns", expanded=False):
    st.markdown(f"<div class='status'>Source: <b>{source}</b></div>", unsafe_allow_html=True)
    st.write("Columns loaded:", list(prices.columns))
    st.write("Rows:", len(prices))

# ---------------- Math & Charts ----------------
rets = prices.pct_change().fillna(0.0)
port_ret = (rets * w).sum(axis=1)
port_cum = (1 + port_ret).cumprod()
indiv_cum = (1 + rets).cumprod()

indiv_rebased = indiv_cum / indiv_cum.iloc[0] * 100
port_rebased = port_cum / port_cum.iloc[0] * 100

fig = go.Figure()
for t in prices.columns:
    fig.add_trace(go.Scatter(x=indiv_rebased.index, y=indiv_rebased[t], name=t, mode="lines", line=dict(width=2)))
fig.add_trace(go.Scatter(x=port_rebased.index, y=port_rebased.values, name="Portfolio",
                         mode="lines", line=dict(color="red", width=5)))
fig.update_layout(title="Normalized Price (Rebased to 100) — Portfolio vs Assets",
                  xaxis_title="Date", yaxis_title="Index (100 = start)",
                  template="plotly_dark", paper_bgcolor="#000", plot_bgcolor="#000",
                  legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, use_container_width=True, height=560)

def sharpe_from_prices(series, rf_annual_pct=0.0, periods_per_year=252):
    r = series.pct_change().dropna()
    if r.std() == 0:
        return np.nan
    rf = rf_annual_pct / 100.0
    return np.sqrt(periods_per_year) * ((r.mean() - rf/periods_per_year) / r.std())

ann = 252
port_sharpe = sharpe_from_prices(port_cum, rf_annual_pct=rf, periods_per_year=ann)
port_cum_return = port_rebased.iloc[-1] / 100 - 1.0

m1, m2 = st.columns(2)
with m1:
    st.markdown(f"<div class='metric-card'><h3>Portfolio Cumulative Return</h3><h2>{port_cum_return*100:,.2f}%</h2></div>", unsafe_allow_html=True)
with m2:
    st.markdown(f"<div class='metric-card'><h3>Portfolio Sharpe Ratio</h3><h2>{port_sharpe:,.2f}</h2></div>", unsafe_allow_html=True)

def cum_return(series):
    return series.iloc[-1] / series.iloc[0] - 1.0

rows = []
for i, t in enumerate(prices.columns):
    rows.append([t, f"{weights_pct[i]:.2f}%", f"{cum_return(indiv_cum[t])*100:,.2f}%",
                 f"{sharpe_from_prices(indiv_cum[t], rf_annual_pct=rf, periods_per_year=ann):.2f}"])
summary = pd.DataFrame(rows, columns=["Ticker","Weight","Cumulative Return","Sharpe Ratio"])
summary.loc[len(summary)] = ["Portfolio","100.00%", f"{port_cum_return*100:,.2f}%", f"{port_sharpe:,.2f}"]

if show_table:
    st.subheader("Performance Summary")
    st.dataframe(summary, use_container_width=True)

st.download_button("⬇️ Download portfolio returns CSV",
                   data=port_ret.to_frame("portfolio_return").to_csv().encode(),
                   file_name="portfolio_returns.csv", mime="text/csv")
