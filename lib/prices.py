def to_yahoo_ticker(symbol: str, exchange: str) -> str:
    """
    Convert NSE/BSE symbol to Yahoo Finance ticker.
    NSE -> .NS
    BSE -> .BO

    Handles common exchange labels from different universe sources.
    """
    sym = (symbol or "").strip().upper()
    ex = (exchange or "").strip().upper()

    # If already a Yahoo-style ticker, return as-is
    if sym.endswith(".NS") or sym.endswith(".BO"):
        return sym

    # Normalize exchange
    # Many sources use NSE/BSE, sometimes NSEEQ, BSEEQ, "NSE" with extra text.
    if "NSE" in ex or ex in {"NS"}:
        return f"{sym}.NS"
    if "BSE" in ex or ex in {"BO"}:
        return f"{sym}.BO"

    # Fallback: assume NSE if unknown (best default for India watchlists)
    return f"{sym}.NS"
