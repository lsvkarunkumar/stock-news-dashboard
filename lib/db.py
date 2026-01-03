import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.path.join("data", "app.db")

def ensure_dirs():
    os.makedirs("data", exist_ok=True)

def init_db():
    ensure_dirs()
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS universe (
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            name TEXT,
            isin TEXT,
            sector TEXT,
            PRIMARY KEY(symbol, exchange)
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS news_mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            date TEXT NOT NULL,       -- YYYY-MM-DD
            mentions INTEGER NOT NULL,
            sample_headline TEXT,
            sample_url TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            date TEXT NOT NULL,       -- YYYY-MM-DD
            close REAL,
            volume REAL,
            PRIMARY KEY(symbol, exchange, date)
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            asof TEXT NOT NULL,       -- YYYY-MM-DD
            score INTEGER NOT NULL,
            reasons TEXT NOT NULL,    -- JSON-ish string
            PRIMARY KEY(symbol, exchange, asof)
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            added_at TEXT NOT NULL,
            PRIMARY KEY(symbol, exchange)
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            side TEXT NOT NULL,         -- BUY/SELL
            qty REAL NOT NULL,
            price REAL NOT NULL,
            notes TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS paper_wallet (
            id INTEGER PRIMARY KEY CHECK (id=1),
            cash REAL NOT NULL
        )
        """)
        cur.execute("INSERT OR IGNORE INTO paper_wallet (id, cash) VALUES (1, 1000000.0)")  # demo coins

        con.commit()

@contextmanager
def db():
    ensure_dirs()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()
