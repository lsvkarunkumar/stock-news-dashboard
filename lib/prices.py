import datetime as dt
import pandas as pd
import yfinance as yf

def to_yahoo_ticker(symbol: str, exchange: str) -> str:
    if exchange == "NSE":
        return f"{symbol}.NS"
    if exchange == "BSE":
        return f"{symbol}.BO"
    return symbol

def fetch_prices(symbol: str, exchange: str, period_days: int = 365) -> pd.DataFrame:
    t = to_yahoo_ticker(symbol, exchange)
    df = yf.download(t, period=f"{period_days}d", interval="1d", progress=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["Date"]).dt.date.astype(str)
    df = df.rename(columns={"Close": "close", "Volume": "volume"})
    return df[["date", "close", "volume"]]
