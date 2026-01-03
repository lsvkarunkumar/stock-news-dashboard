import datetime as dt
import pandas as pd

from lib.db import init_db, db
from lib.universe import fetch_universe, upsert_universe, get_universe
from lib.news import build_company_query, gdelt_mentions, gdelt_latest_headline
from lib.prices import fetch_prices
from lib.scoring import compute_score

ASOF = dt.date.today()

def upsert_news_row(symbol: str, exchange: str, date: str, mentions: int, headline, url):
    with db() as con:
        con.execute("""
            INSERT INTO news_mentions(symbol, exchange, date, mentions, sample_headline, sample_url)
            VALUES (?,?,?,?,?,?)
        """, (symbol, exchange, date, mentions, headline, url))
        con.commit()

def upsert_prices(symbol: str, exchange: str, df: pd.DataFrame):
    if df.empty:
        return
    rows = [(symbol, exchange, r.date, float(r.close) if pd.notna(r.close) else None, float(r.volume) if pd.notna(r.volume) else None)
            for r in df.itertuples(index=False)]
    with db() as con:
        con.executemany("""
            INSERT OR REPLACE INTO prices(symbol, exchange, date, close, volume)
            VALUES (?,?,?,?,?)
        """, rows)
        con.commit()

def upsert_signal(symbol: str, exchange: str, asof: str, score: int, reasons: str):
    with db() as con:
        con.execute("""
            INSERT OR REPLACE INTO signals(symbol, exchange, asof, score, reasons)
            VALUES (?,?,?,?,?)
        """, (symbol, exchange, asof, score, reasons))
        con.commit()

def main():
    init_db()

    # 1) Universe refresh
    uni = fetch_universe()
    upsert_universe(uni)

    uni = get_universe()

    # 2) Only compute signals for your watchlist (fast)
    with db() as con:
        wl = pd.read_sql_query("SELECT symbol, exchange FROM watchlist", con)

    if wl.empty:
        print("Watchlist empty. Universe updated. Add stocks in the app first.")
        return

    for r in wl.itertuples(index=False):
        row = uni[(uni.symbol == r.symbol) & (uni.exchange == r.exchange)]
        if row.empty:
            continue
        name = row.iloc[0]["name"]

        # News windows
        d5 = ASOF - dt.timedelta(days=5)
        d30 = ASOF - dt.timedelta(days=30)
        d60 = ASOF - dt.timedelta(days=60)

        q = build_company_query(r.symbol, name)
        m5 = gdelt_mentions(q, d5, ASOF)
        m30 = gdelt_mentions(q, d30, ASOF)
        m60 = gdelt_mentions(q, d60, ASOF)
        headline, url = gdelt_latest_headline(q, d60, ASOF)

        upsert_news_row(r.symbol, r.exchange, ASOF.isoformat(), m60, headline, url)

        # Prices (1y)
        px = fetch_prices(r.symbol, r.exchange, period_days=365)
        upsert_prices(r.symbol, r.exchange, px)

        # Score
        score, reasons = compute_score(m5, m30, m60, px)
        upsert_signal(r.symbol, r.exchange, ASOF.isoformat(), score, reasons)

        print(r.symbol, r.exchange, score)

if __name__ == "__main__":
    main()
