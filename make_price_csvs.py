# make_price_csvs.py  (overwrite your old one)
import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_DIR = Path("Data")
DATA_DIR.mkdir(exist_ok=True)

TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META",
    "NVDA","TSLA","JPM","V","MA",
    "HD","PG","UNH","XOM","JNJ",
    "WMT","KO","PEP","NFLX","DIS"
]

print("Downloading stock data (5 years daily)...")
df_all = yf.download(TICKERS, period="5y", interval="1d",
                     auto_adjust=False, group_by="ticker", progress=False)

# IMPORTANT: write with minimal quoting and UTF-8
df_all.to_csv(DATA_DIR / "prices_combined.csv", index=True)

for t in TICKERS:
    print("Saving:", t)
    df = yf.download(t, period="5y", interval="1d",
                     auto_adjust=False, group_by="ticker", progress=False)
    if not isinstance(df.columns, pd.MultiIndex):
        # Convert single-level columns to MultiIndex (field, ticker)
        df.columns = pd.MultiIndex.from_product([df.columns, [t]])
    df.to_csv(DATA_DIR / f"{t}.csv", index=True)

print("✅ Done. Files saved in Data/")
