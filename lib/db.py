import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path("data") / "app.db"


def _ensure_paths():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def db():
    _ensure_paths()
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    try:
        yield con
    finally:
        con.close()


def init_db():
    _ensure_paths()
    with db() as con:
        cur = con.cursor()

        # Universe
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

        # Watchlist
        cur.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            added_at TEXT,
            PRIMARY KEY(symbol, exchange)
        )
        """)

        # Prices
        cur.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            date TEXT NOT NULL,
            close REAL,
            volume REAL,
            PRIMARY KEY(symbol, exchange, date)
        )
        """)

        # RSS News items (raw headlines)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS news_items (
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            published TEXT,
            title TEXT,
            url TEXT,
            source TEXT,
            PRIMARY KEY(symbol, exchange, published, title)
        )
        """)

        # Signals (Score snapshots)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            asof TEXT NOT NULL,
            score INTEGER,
            reasons TEXT,
            PRIMARY KEY(symbol, exchange, asof)
        )
        """)

        # Paper trading (single wallet for now; we’ll split Auto/Manual next step)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS paper_wallet (
            id INTEGER PRIMARY KEY,
            cash REAL
        )
        """)
        cur.execute("INSERT OR IGNORE INTO paper_wallet(id, cash) VALUES (1, 1000000)")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            ts TEXT,
            symbol TEXT,
            exchange TEXT,
            side TEXT,
            qty REAL,
            price REAL,
            notes TEXT
        )
        """)

        # Indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_prices_symex ON prices(symbol, exchange)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_news_items_symex ON news_items(symbol, exchange)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_symex ON signals(symbol, exchange)")

        con.commit()
