import streamlit as st
import yfinance as yf
import plotly.graph_objs as go

st.set_page_config(page_title="Market Overview", layout="wide")

st.title("🌍 Market Overview")

# Select tickers for indexes
indexes = {"S&P 500": "^GSPC", "Dow Jones": "^DJI", "NASDAQ": "^IXIC", 
           "FTSE 100": "^FTSE", "DAX": "^GDAXI", "Nikkei 225": "^N225"}

data = {}
for name, ticker in indexes.items():
    df = yf.download(ticker, period="1y")
    if isinstance(df.columns, pd.MultiIndex):
    df = df["Adj Close"]

    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"] if "Adj Close" in df else df, mode="lines", name=name))
    data[name] = df

# Plot comparison
fig = go.Figure()
for name, df in data.items():
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode="lines", name=name))

fig.update_layout(title="Major Indexes", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

# Show live prices
st.subheader("📌 Current Prices")
for name, df in data.items():
    st.metric(name, f"${df['Adj Close'][-1]:.2f}")
