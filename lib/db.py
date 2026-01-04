import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

# Always store DB inside repo so GitHub Actions + Streamlit read the SAME file
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = str(DATA_DIR / "stock.db")

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")

        con.execute("""
        CREATE TABLE IF NOT EXISTS universe(
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            name TEXT,
            sector TEXT,
            isin TEXT,
            PRIMARY KEY(symbol, exchange)
        )
        """)

        con.execute("""
        CREATE TABLE IF NOT EXISTS watchlist(
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            added_at TEXT,
            PRIMARY KEY(symbol, exchange)
        )
        """)

        con.execute("""
        CREATE TABLE IF NOT EXISTS prices(
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            date TEXT NOT NULL,
            close REAL,
            volume REAL,
            PRIMARY KEY(symbol, exchange, date)
        )
        """)

        con.execute("""
        CREATE TABLE IF NOT EXISTS news_items(
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            published TEXT,
            title TEXT,
            url TEXT,
            source TEXT,
            UNIQUE(symbol, exchange, published, title)
        )
        """)

        con.execute("""
        CREATE TABLE IF NOT EXISTS signals(
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            asof TEXT NOT NULL,
            score INTEGER,
            reasons TEXT,
            PRIMARY KEY(symbol, exchange, asof)
        )
        """)

        con.execute("""
        CREATE TABLE IF NOT EXISTS paper_wallet(
            id INTEGER PRIMARY KEY CHECK (id=1),
            cash REAL
        )
        """)

        con.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            symbol TEXT,
            exchange TEXT,
            side TEXT,
            qty REAL,
            price REAL,
            note TEXT
        )
        """)

        # ensure wallet row exists
        cur = con.execute("SELECT COUNT(*) FROM paper_wallet WHERE id=1")
        if cur.fetchone()[0] == 0:
            con.execute("INSERT INTO paper_wallet(id, cash) VALUES(1, ?)", (100000.0,))
        con.commit()

@contextmanager
def db():
    con = sqlite3.connect(DB_PATH)
    try:
        yield con
    finally:
        con.close()
