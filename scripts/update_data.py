import json
from datetime import date

import pandas as pd

from lib.db import init_db, db
from lib.news_rss import fetch_google_news_rss
from lib.prices import to_yahoo_ticker
from lib.scoring import indicator_features, indicator_score, combined_score

import yfinance as yf


def yf_daily(symbol: str, exchange: str) -> pd.DataFrame:
    t = to_yahoo_ticker(symbol, exchange)
    raw = yf.download(t, period="370d", interval="1d", progress=False, threads=True, auto_adjust=False)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])
    df = raw.reset_index()
    dcol = "Date" if "Date" in df.columns else df.columns[0]
    df["date"] = pd.to_datetime(df[dcol], errors="coerce").dt.date.astype(str)

    close_col = "Close" if "Close" in df.columns else ("close" if "close" in df.columns else None)
    vol_col = "Volume" if "Volume" in df.columns else ("volume" if "volume" in df.columns else None)
    if close_col is None:
        return pd.DataFrame(columns=["date", "close", "volume"])

    df["close"] = pd.to_numeric(df[close_col], errors="coerce")
    df["volume"] = pd.to_numeric(df[vol_col], errors="coerce") if vol_col else None
    return df[["date", "close", "volume"]].dropna(subset=["date", "close"])


def upsert_prices(symbol, exchange, df):
    if df is None or df.empty:
        return
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r["date"]) or pd.isna(r["close"]):
            continue
        rows.append((symbol, exchange, str(r["date"]), float(r["close"]), float(r["volume"]) if pd.notna(r["volume"]) else None))
    if not rows:
        return
    with db() as con:
        con.executemany(
            "INSERT OR REPLACE INTO prices(symbol, exchange, date, close, volume) VALUES (?,?,?,?,?)",
            rows,
        )
        con.commit()


def upsert_news(symbol, exchange, name, days=60):
    items = fetch_google_news_rss(name, symbol, days=days) or []
    if not items:
        return 0
    with db() as con:
        con.executemany(
            """
            INSERT OR IGNORE INTO news_items(symbol, exchange, published, title, url, source)
            VALUES (?,?,?,?,?,?)
            """,
            [(symbol, exchange, it["published"], it["title"], it["url"], it["source"]) for it in items],
        )
        con.commit()
    return len(items)


def rss_counts(symbol, exchange):
    from datetime import timedelta
    import datetime as dt

    today = dt.date.today()
    with db() as con:
        df = pd.read_sql_query(
            "SELECT published FROM news_items WHERE symbol=? AND exchange=?",
            con,
            params=(symbol, exchange),
        )
    if df.empty:
        return 0, 0, 999
    df["published"] = pd.to_datetime(df["published"], errors="coerce").dt.date
    df = df.dropna(subset=["published"])
    if df.empty:
        return 0, 0, 999
    m5 = int((today - df["published"] <= timedelta(days=5)).sum())
    m60 = int((today - df["published"] <= timedelta(days=60)).sum())
    last = df["published"].max()
    recency_days = (today - last).days if last else 999
    return m5, m60, recency_days


def news_score(m5, m60, recency_days):
    baseline = max(1.0, m60 / 12.0)
    intensity_ratio = m5 / baseline

    intensity_pts = (intensity_ratio - 0.5) * 30.0
    intensity_pts = float(max(0.0, min(60.0, intensity_pts)))

    if recency_days <= 2:
        recency_pts = 40
    elif recency_days <= 5:
        recency_pts = 30
    elif recency_days <= 10:
        recency_pts = 20
    elif recency_days <= 20:
        recency_pts = 10
    else:
        recency_pts = 0

    score = int(round(min(100.0, intensity_pts + recency_pts)))
    return max(0, min(100, score))


def upsert_signal(symbol, exchange, asof, score_val, reasons):
    with db() as con:
        con.execute(
            "INSERT OR REPLACE INTO signals(symbol, exchange, asof, score, reasons) VALUES (?,?,?,?,?)",
            (symbol, exchange, asof, int(score_val), json.dumps(reasons)),
        )
        con.commit()


def main():
    init_db()
    asof = date.today().isoformat()

    with db() as con:
        wl = pd.read_sql_query(
            """
            SELECT w.symbol, w.exchange, COALESCE(u.name,w.symbol) AS name
            FROM watchlist w
            LEFT JOIN universe u ON u.symbol=w.symbol AND u.exchange=w.exchange
            """,
            con,
        )

    if wl.empty:
        print("Watchlist empty. Nothing to update.")
        return

    for r in wl.itertuples(index=False):
        symbol = str(r.symbol).strip().upper()
        exchange = str(r.exchange).strip().upper()
        name = str(r.name).strip()

        # Prices
        px = yf_daily(symbol, exchange)
        upsert_prices(symbol, exchange, px)

        # RSS
        fetched = upsert_news(symbol, exchange, name, days=60)
        m5, m60, recency_days = rss_counts(symbol, exchange)
        nscore = news_score(m5, m60, recency_days)

        # Indicator scoring (from DB prices to be consistent)
        with db() as con:
            dbpx = pd.read_sql_query(
                "SELECT date, close, volume FROM prices WHERE symbol=? AND exchange=? ORDER BY date",
                con,
                params=(symbol, exchange),
            )

        feats = indicator_features(dbpx)
        iscore, ibreak = indicator_score(feats)
        cscore = combined_score(iscore, nscore, 0.8, 0.2)

        reasons = {
            "source": "Daily update (Actions)",
            "name": name,
            "rss": {"fetched": fetched, "m5": m5, "m60": m60, "recency_days": recency_days, "news_score": nscore},
            "indicator": ibreak,
            "combined": {"indicator_score": iscore, "news_score": nscore, "combined_score": cscore, "w_ind": 0.8, "w_news": 0.2},
        }

        # Store combined score in score field (so tables can sort by it)
        upsert_signal(symbol, exchange, asof, cscore, reasons)

        print(symbol, exchange, "news", nscore, "ind", iscore, "combined", cscore)

    print("Done.")


if __name__ == "__main__":
    main()
