import pandas as pd
import requests
from io import StringIO
from .db import db

# Official sources (can change columns over time, so we handle flexibly)
NSE_EQUITY_L = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
BSE_LIST = "https://www.bseindia.com/downloads1/List_of_companies.csv"

def _get(url: str) -> str:
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace("\ufeff", "") for c in df.columns]
    return df

def _pick(df: pd.DataFrame, candidates: list[str], default=None):
    """Pick the first matching column from candidates (case-normalized)."""
    for c in candidates:
        c2 = c.strip().lower()
        if c2 in df.columns:
            return df[c2]
    return default

def fetch_universe() -> pd.DataFrame:
    # -------- NSE --------
    nse_raw = pd.read_csv(StringIO(_get(NSE_EQUITY_L)))
    nse = _norm_cols(nse_raw)

    nse_symbol = _pick(nse, ["symbol", "sym", "security symbol"])
    nse_name   = _pick(nse, ["name of company", "company name", "security name", "name"])
    nse_isin   = _pick(nse, ["isin number", "isin", "isin no", "isin code"], default="")

    nse_df = pd.DataFrame({
        "symbol": nse_symbol.astype(str).str.strip() if nse_symbol is not None else pd.Series(dtype=str),
        "exchange": "NSE",
        "name": nse_name.astype(str).str.strip() if nse_name is not None else "",
        "isin": nse_isin.astype(str).str.strip() if isinstance(nse_isin, pd.Series) else ""
    })
    nse_df = nse_df.dropna(subset=["symbol"])
    nse_df = nse_df[nse_df["symbol"].astype(str).str.len() > 0]

    # -------- BSE --------
    bse_raw = pd.read_csv(StringIO(_get(BSE_LIST)))
    bse = _norm_cols(bse_raw)

    bse_symbol = _pick(bse, ["scrip code", "scripcode", "code", "security code"])
    bse_name   = _pick(bse, ["security name", "name", "company name"])
    bse_isin   = _pick(bse, ["isin no", "isin", "isin number", "isin code"], default="")

    bse_df = pd.DataFrame({
        "symbol": bse_symbol.astype(str).str.strip() if bse_symbol is not None else pd.Series(dtype=str),
        "exchange": "BSE",
        "name": bse_name.astype(str).str.strip() if bse_name is not None else "",
        "isin": bse_isin.astype(str).str.strip() if isinstance(bse_isin, pd.Series) else ""
    })
    bse_df = bse_df.dropna(subset=["symbol"])
    bse_df = bse_df[bse_df["symbol"].astype(str).str.len() > 0]

    # Combine
    uni = pd.concat([nse_df, bse_df], ignore_index=True)
    uni["symbol"] = uni["symbol"].astype(str).str.strip()
    uni["name"] = uni["name"].astype(str).str.strip()
    uni["isin"] = uni["isin"].astype(str).str.strip()

    return uni[["symbol", "exchange", "name", "isin"]]

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
