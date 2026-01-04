import json
from datetime import date

import pandas as pd
import yfinance as yf

from lib.db import init_db, db, DB_PATH
from lib.prices import to_yahoo_ticker
from lib.news_rss import fetch_google_news_rss
from lib.scoring import indicator_features, indicator_score, combined_score


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        try:
            tickers = list(dict.fromkeys([c[1] for c in df.columns if len(c) > 1]))
            if tickers:
                df = df.xs(tickers[0], axis=1, level=1, drop_level=True)
        except Exception:
            pass
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) and len(c) > 0 else str(c) for c in df.columns]
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    return df


def yf_daily(symbol: str, exchange: str):
    t = to_yahoo_ticker(symbol, exchange)
    raw = yf.download(t, period="370d", interval="1d", progress=False, threads=True, auto_adjust=False)

    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "close", "volume"]), {"ticker": t, "rows": 0}

    df = _flatten_yf_columns(raw.reset_index())
    dcol = "Date" if "Date" in df.columns else df.columns[0]
    df["date"] = pd.to_datetime(df[dcol], errors="coerce").dt.date.astype(str)

    close_s = df["Close"] if "Close" in df.columns else None
    vol_s = df["Volume"] if "Volume" in df.columns else None
    if close_s is None:
        return pd.DataFrame(columns=["date", "close", "volume"]), {"ticker": t, "rows": 0}

    if isinstance(close_s, pd.DataFrame):
        close_s = close_s.iloc[:, 0]
    if isinstance(vol_s, pd.DataFrame):
        vol_s = vol_s.iloc[:, 0]

    df["close"] = pd.to_numeric(close_s, errors="coerce")
    df["volume"] = pd.to_numeric(vol_s, errors="coerce") if vol_s is not None else None

    out = df[["date", "close", "volume"]].dropna(subset=["date", "close"])
    return out, {"ticker": t, "rows": int(len(out))}


def upsert_prices(symbol, exchange, df):
    if df is None or df.empty:
        return 0
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r["date"]) or pd.isna(r["close"]):
            continue
        rows.append(
            (symbol, exchange, str(r["date"]), float(r["close"]), float(r["volume"]) if pd.notna(r["volume"]) else None)
        )
    if not rows:
        return 0
    with db() as con:
        con.executemany(
            "INSERT OR REPLACE INTO prices(symbol, exchange, date, close, volume) VALUES (?,?,?,?,?)",
            rows,
        )
        con.commit()
    return len(rows)


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


def upsert_signal(symbol, exchange, asof, combined_val, reasons):
    with db() as con:
        con.execute(
            "INSERT OR REPLACE INTO signals(symbol, exchange, asof, score, reasons) VALUES (?,?,?,?,?)",
            (symbol, exchange, asof, int(combined_val), json.dumps(reasons)),
        )
        con.commit()


def main():
    init_db()
    asof = date.today().isoformat()
    print("DB_PATH =", DB_PATH)

    with db() as con:
        wl = pd.read_sql_query(
            """
            SELECT w.symbol, w.exchange, COALESCE(u.name,w.symbol) AS name
            FROM watchlist w
            LEFT JOIN universe u ON u.symbol=w.symbol AND u.exchange=w.exchange
            """,
            con,
        )
    print("watchlist rows =", len(wl))

    if wl.empty:
        print("Watchlist empty. Nothing to update.")
        return

    for r in wl.itertuples(index=False):
        symbol = str(r.symbol).strip().upper()
        exchange = str(r.exchange).strip().upper()
        name = str(r.name).strip()

        px, meta = yf_daily(symbol, exchange)
        ins_px = upsert_prices(symbol, exchange, px)

        ins_news = upsert_news(symbol, exchange, name, days=60)

        # Pull DB prices for scoring
        with db() as con:
            dbpx = pd.read_sql_query(
                "SELECT date, close, volume FROM prices WHERE symbol=? AND exchange=? ORDER BY date",
                con,
                params=(symbol, exchange),
            )

        feats = indicator_features(dbpx)
        iscore, ibreak = indicator_score(feats)

        # (NewsScore can remain 0 for now; we’ll wire sentiment next after prices work)
        nscore = 0
        cscore = combined_score(iscore, nscore, 0.8, 0.2)

        reasons = {
            "source": "Daily update (Actions)",
            "scores": {"indicator_score": iscore, "news_score": nscore, "combined_score": cscore, "w_ind": 0.8, "w_news": 0.2},
            "prices": {"yahoo_ticker": meta["ticker"], "rows_fetched": meta["rows"], "rows_inserted": ins_px, "rows_db": int(len(dbpx))},
            "rss": {"rows_inserted": ins_news},
            "indicator": ibreak,
        }
        upsert_signal(symbol, exchange, asof, cscore, reasons)

        print(symbol, exchange, "yf_rows", meta["rows"], "ins_px", ins_px, "db_px", len(dbpx), "news_ins", ins_news)

    with db() as con:
        total_px = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        total_news = con.execute("SELECT COUNT(*) FROM news_items").fetchone()[0]
        total_sig = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    print("TOTAL prices =", total_px, "news_items =", total_news, "signals =", total_sig)
    print("Done.")


if __name__ == "__main__":
    main()
