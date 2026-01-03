import pandas as pd
import requests
from io import StringIO
from .db import db

NSE_EQUITY_L = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
BSE_LIST = "https://www.bseindia.com/downloads1/List_of_companies.csv"

def _get(url: str) -> str:
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text

def fetch_universe() -> pd.DataFrame:
    # NSE
    nse = pd.read_csv(StringIO(_get(NSE_EQUITY_L)))
    nse = nse.rename(columns={
        "SYMBOL": "symbol",
        "NAME OF COMPANY": "name",
        "ISIN NUMBER": "isin"
    })
    nse["exchange"] = "NSE"
    nse = nse[["symbol", "exchange", "name", "isin"]]

    # BSE
    bse = pd.read_csv(StringIO(_get(BSE_LIST)))
    # Common columns: "Scrip Code","Security Name","ISIN No"
    bse = bse.rename(columns={
        "Scrip Code": "symbol",
        "Security Name": "name",
        "ISIN No": "isin"
    })
    bse["symbol"] = bse["symbol"].astype(str)
    bse["exchange"] = "BSE"
    bse = bse[["symbol", "exchange", "name", "isin"]]

    uni = pd.concat([nse, bse], ignore_index=True).dropna(subset=["symbol", "exchange"])
    uni["symbol"] = uni["symbol"].astype(str).str.strip()
    uni["name"] = uni["name"].astype(str).str.strip()
    uni["isin"] = uni["isin"].astype(str).str.strip()
    return uni

def upsert_universe(df: pd.DataFrame) -> None:
    with db() as con:
        cur = con.cursor()
        cur.execute("DELETE FROM universe")
        cur.executemany(
            "INSERT OR REPLACE INTO universe(symbol, exchange, name, isin, sector) VALUES (?,?,?,?,?)",
            [(r.symbol, r.exchange, r.name, r.isin, None) for r in df.itertuples(index=False)]
        )
        con.commit()

def get_universe() -> pd.DataFrame:
    with db() as con:
        return pd.read_sql_query("SELECT * FROM universe", con)
