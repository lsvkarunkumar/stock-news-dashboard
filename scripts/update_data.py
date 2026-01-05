import os
from datetime import date

import pandas as pd
import numpy as np
import yfinance as yf

from lib.db import init_db, db
from lib.prices import to_yahoo_ticker
from lib.scoring import indicator_features, indicator_score, combined_score

WATCHLIST_CSV = "data/watchlist.csv"


def read_watchlist() -> pd.DataFrame:
    if not os.path.exists(WATCHLIST_CSV):
        return pd.DataFrame(columns=["symbol", "exchange"])
    df = pd.read_csv(WATCHLIST_CSV)
    if df.empty:
        return pd.DataFrame(columns=["symbol", "exchange"])
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["exchange"] = df["exchange"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=["symbol", "exchange"]).reset_index(drop=True)
    return df


def flatten_yf(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return raw
    if isinstance(raw.columns, pd.MultiIndex):
        # pick first ticker level if multi
        try:
            tickers = list(dict.fromkeys([c[1] for c in raw.columns if len(c) > 1]))
            if tickers:
                raw = raw.xs(tickers[0], axis=1, level=1, drop_level=True)
        except Exception:
            pass
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] if isinstance(c, tuple) else str(c) for c in raw.columns]
    raw.columns = [str(c) for c in raw.columns]
    return raw


def fetch_prices_yahoo(symbol: str, exchange: str) -> pd.DataFrame:
    t = to_yahoo_ticker(symbol, exchange)
    if not t:
        return pd.DataFrame(columns=["date", "close", "volume"])

    raw = yf.download(
        t,
        period="370d",
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=True,
    )

    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])

    df = flatten_yf(raw.reset_index())

    # date column could be Date or Datetime depending on yfinance/pandas
    dcol = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else df.columns[0])

    df["date"] = pd.to_datetime(df[dcol], errors="coerce").dt.date.astype(str)

    close = df["Close"] if "Close" in df.columns else None
    vol = df["Volume"] if "Volume" in df.columns else None

    if close is None:
        return pd.DataFrame(columns=["date", "close", "volume"])

    # sometimes these are DataFrames if multi
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(vol, pd.DataFrame):
        vol = vol.iloc[:, 0] if vol is not None else None

    out = pd.DataFrame(
        {
            "date": df["date"],
            "close": pd.to_numeric(close, errors="coerce"),
            "volume": pd.to_numeric(vol, errors="coerce") if vol is not None else np.nan,
        }
    ).dropna(subset=["date", "close"])

    return out


def upsert_prices(symbol: str, exchange: str, px: pd.DataFrame) -> int:
    if px is None or px.empty:
        return 0
    rows = []
    for r in px.itertuples(index=False):
        rows.append((symbol, exchange, str(r.date), float(r.close), float(r.volume) if pd.notna(r.volume) else None))

    with db() as con:
        con.executemany(
            "INSERT OR REPLACE INTO prices(symbol, exchange, date, close, volume) VALUES (?,?,?,?,?)",
            rows,
        )
        con.commit()
    return len(rows)


def upsert_signal(symbol: str, exchange: str, ind_score: int, news_score: int, comb_score: int, price_rows: int):
    asof = date.today().isoformat()
    reasons = {
        "source": "GitHub Actions update",
        "scores": {
            "indicator_score": int(ind_score),
            "news_score": int(news_score),
            "combined_score": int(comb_score),
            "w_ind": 0.8,
            "w_news": 0.2,
        },
        "price_rows_db": int(price_rows),
        "yahoo_ticker": to_yahoo_ticker(symbol, exchange),
    }
    import json as _json

    with db() as con:
        con.execute(
            "INSERT OR REPLACE INTO signals(symbol, exchange, asof, score, reasons) VALUES (?,?,?,?,?)",
            (symbol, exchange, asof, int(comb_score), _json.dumps(reasons)),
        )
        con.commit()


def main():
    init_db()

    wl = read_watchlist()
    if wl.empty:
        print("Watchlist is empty. Nothing to update.")
        return

    print(f"Watchlist rows: {len(wl)}")

    for r in wl.itertuples(index=False):
        symbol = r.symbol
        exchange = r.exchange
        print(f"Fetching prices: {symbol} ({exchange}) => {to_yahoo_ticker(symbol, exchange)}")

        px = fetch_prices_yahoo(symbol, exchange)
        inserted = upsert_prices(symbol, exchange, px)
        print(f"Inserted/updated price rows: {inserted}")

        # Compute IndicatorScore from DB prices (freshly inserted)
        with db() as con:
            dbpx = pd.read_sql_query(
                "SELECT date, close, volume FROM prices WHERE symbol=? AND exchange=? ORDER BY date",
                con,
                params=(symbol, exchange),
            )

        feats = indicator_features(dbpx)
        ind_score, _ = indicator_score(feats)

        # NewsScore: keep 0 in Actions for now (you already compute in app via RSS)
        news_score = 0
        comb = combined_score(ind_score, news_score, 0.8, 0.2)

        upsert_signal(symbol, exchange, ind_score, news_score, comb, price_rows=len(dbpx))

    print("Done.")


if __name__ == "__main__":
    main()
