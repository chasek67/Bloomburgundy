import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ---------- Page Setup ----------
st.set_page_config(layout="wide", page_title="Stock Comparison")

# ---------- Theme & UX ----------
st.markdown(
    """
    <style>
        .stApp { background-color: #0e0e0e; color: #e6e6e6; }
        h1,h2,h3,h4,h5,h6,label,p,span,div { color: #e6e6e6 !important; }
        .stTextInput>div>div>input, .stSelectbox>div>div>div>span, .stSlider, .stMultiSelect, .stNumberInput input {
            background-color: #1a1a1a;
            color: #e6e6e6 !important;
        }
        .metric-card { background:#1a1a1a;padding:12px;border-radius:12px;border:1px solid #333; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Stock Comparison")

# ---------- Controls ----------
with st.sidebar:
    st.header("Controls")
    tickers_input = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN")
    horizon = st.selectbox(
        "Time Horizon",
        ["3M", "6M", "1Y", "2Y", "5Y", "YTD", "Max"],
        index=2,
    )
    rebased = st.checkbox("Normalize to 100 (rebased)", True)
    log_scale = st.checkbox("Log scale (prices)", False)
    show_indicators = st.multiselect(
        "Extras",
        ["30D Rolling Volatility", "Return Distribution", "Correlation Heatmap"],
        default=["Correlation Heatmap"],
    )
    sampling = st.selectbox("Sampling Frequency", ["Daily", "Weekly", "Monthly"], index=0)

def _period(h: str) -> str:
    return {
        "3M": "3mo",
        "6M": "6mo",
        "1Y": "1y",
        "2Y": "2y",
        "5Y": "5y",
        "YTD": "ytd",
        "Max": "max",
    }[h]

def _to_freq(freq_label: str) -> str:
    return {"Daily": "1D", "Weekly": "1W", "Monthly": "1M"}[freq_label]

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.stop()

# ---------- Download ----------
@st.cache_data(show_spinner=False)
def load_prices(tickers, period, interval="1d"):
    df = yf.download(tickers, period=period, auto_adjust=True)["Adj Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
    return df

try:
    prices_raw = load_prices(tickers, _period(horizon))
    prices = prices_raw.resample(_to_freq(sampling)).last().dropna(how="all")
    prices = prices.dropna(axis=1, how="all")
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        st.warning(f"Missing/invalid tickers skipped: {', '.join(missing)}")
        tickers = [t for t in tickers if t in prices.columns]
        prices = prices[tickers]
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ---------- Derived Series ----------
rets = prices.pct_change().fillna(0.0)
cum = (1 + rets).cumprod()

palette = px.colors.qualitative.Bold + px.colors.qualitative.Plotly + px.colors.qualitative.D3
color_map = {t: palette[i % len(palette)] for i, t in enumerate(tickers)}

left, right = st.columns([2, 1])

# ---------- Charts ----------
with left:
    if rebased:
        base = cum.iloc[0]
        rebased100 = cum.div(base) * 100
        fig_prices = go.Figure()
        for t in tickers:
            fig_prices.add_trace(
                go.Scatter(x=rebased100.index, y=rebased100[t], name=t,
                           mode="lines", line=dict(width=2.2, color=color_map[t]))
            )
        fig_prices.update_layout(
            title="Normalized Price (Rebased to 100)",
            xaxis_title="Date",
            yaxis_title="Index (100 = start)",
            template="plotly_dark",
            paper_bgcolor="#000",
            plot_bgcolor="#000",
            legend=dict(orientation="h", y=-0.2),
        )
    else:
        fig_prices = go.Figure()
        for t in tickers:
            fig_prices.add_trace(
                go.Scatter(x=prices.index, y=prices[t], name=t,
                           mode="lines", line=dict(width=2.2, color=color_map[t]))
            )
        fig_prices.update_layout(
            title="Adjusted Close Price",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            paper_bgcolor="#000",
            plot_bgcolor="#000",
            legend=dict(orientation="h", y=-0.2),
        )
    if log_scale:
        fig_prices.update_yaxes(type="log")
    st.plotly_chart(fig_prices, use_container_width=True, height=520)

    cumret = cum / cum.iloc[0] - 1
    fig_cum = go.Figure()
    for t in tickers:
        fig_cum.add_trace(
            go.Scatter(x=cumret.index, y=cumret[t] * 100.0, name=t,
                       mode="lines", line=dict(width=2.2, color=color_map[t]))
        )
    fig_cum.update_layout(
        title="Cumulative Return (%)",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        template="plotly_dark",
        paper_bgcolor="#000",
        plot_bgcolor="#000",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_cum, use_container_width=True, height=520)

# ---------- Right column ----------
with right:
    st.subheader("Snapshot")
    ann_factor = 252 if _to_freq(sampling) == "1D" else (52 if _to_freq(sampling) == "1W" else 12)

    def sharpe(series, rf=0.0):
        r = series.pct_change().dropna()
        if r.std() == 0:
            return np.nan
        return np.sqrt(ann_factor) * (r.mean() - rf / ann_factor) / r.std()

    rows = []
    for t in tickers:
        total = (prices[t].iloc[-1] / prices[t].iloc[0]) - 1
        s = sharpe(prices[t])
        best = rets[t].max()
        worst = rets[t].min()
        rows.append([t, f"{total*100:,.2f}%", f"{s:,.2f}", f"{best*100:,.2f}%", f"{worst*100:,.2f}%"])

    st.dataframe(pd.DataFrame(rows, columns=["Ticker", "Cumulative Return", "Sharpe", "Best Day", "Worst Day"]),
                 use_container_width=True, height=260)

    if "Correlation Heatmap" in show_indicators:
        corr = rets[tickers].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Viridis")
        fig_corr.update_layout(template="plotly_dark", title="Return Correlation")
        st.plotly_chart(fig_corr, use_container_width=True)

    if "30D Rolling Volatility" in show_indicators:
        window = 30 if _to_freq(sampling) == "1D" else (26 if _to_freq(sampling) == "1W" else 6)
        roll_vol = rets[tickers].rolling(window).std() * np.sqrt(ann_factor)
        fig_vol = go.Figure()
        for t in tickers:
            fig_vol.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol[t], name=t,
                                         mode="lines", line=dict(color=color_map[t])))
        fig_vol.update_layout(template="plotly_dark",
                              title=f"Rolling Volatility (annualized, window={window})")
        st.plotly_chart(fig_vol, use_container_width=True)

    if "Return Distribution" in show_indicators:
        from plotly.subplots import make_subplots
        fig_hist = make_subplots(rows=1, cols=1)
        for t in tickers:
            fig_hist.add_histogram(x=(rets[t] * 100.0), name=t, opacity=0.5, marker_color=color_map[t])
        fig_hist.update_layout(barmode="overlay", template="plotly_dark",
                               title="Daily Return Distribution (%)")
        st.plotly_chart(fig_hist, use_container_width=True)

st.download_button("⬇️ Download returns CSV", data=rets.to_csv().encode(),
                   file_name="returns.csv", mime="text/csv")


