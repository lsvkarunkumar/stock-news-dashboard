def to_yahoo_ticker(symbol: str, exchange: str) -> str:
    """
    NSE -> .NS, BSE -> .BO for Yahoo Finance.
    """
    s = (symbol or "").strip().upper()
    ex = (exchange or "").strip().upper()

    if not s:
        return ""

    # already has suffix
    if s.endswith(".NS") or s.endswith(".BO"):
        return s

    if ex == "NSE":
        return f"{s}.NS"
    if ex == "BSE":
        return f"{s}.BO"

    # default: return symbol as-is
    return s
