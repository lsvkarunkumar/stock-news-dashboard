import numpy as np
import pandas as pd

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    ema_up = up.ewm(alpha=1/period, adjust=False).mean()
    ema_down = down.ewm(alpha=1/period, adjust=False).mean()

    rs = ema_up / ema_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _safe_last(s: pd.Series):
    if s is None or len(s) == 0:
        return np.nan
    s = s.dropna()
    if s.empty:
        return np.nan
    return float(s.iloc[-1])

def _pct(a, b):
    if b is None or pd.isna(b) or b == 0 or a is None or pd.isna(a):
        return np.nan
    return (a / b - 1.0) * 100.0

def indicator_features(prices: pd.DataFrame) -> dict:
    """
    prices: DataFrame with columns ['date','close','volume']
    Returns indicator features dict. Needs >= 60 rows for stability.
    """
    out = {}
    if prices is None or not isinstance(prices, pd.DataFrame) or prices.empty:
        return out
    if "close" not in prices.columns:
        return out

    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    else:
        df["volume"] = np.nan

    df = df.dropna(subset=["date", "close"]).sort_values("date")
    if len(df) < 60:
        return out

    close = df["close"]
    vol = df["volume"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    macd_signal = _ema(macd, 9)
    macd_hist = macd - macd_signal

    rsi14 = _rsi(close, 14)

    c_last = _safe_last(close)
    c_20 = _safe_last(close.shift(20))
    c_60 = _safe_last(close.shift(60))
    c_120 = _safe_last(close.shift(120))

    ret_1m = _pct(c_last, c_20)
    ret_3m = _pct(c_last, c_60)
    ret_6m = _pct(c_last, c_120)

    ret_daily = close.pct_change()
    vol20 = _safe_last(ret_daily.rolling(20).std())  # fraction

    roll_high_252 = close.rolling(252).max()
    dd_from_high = (close / roll_high_252 - 1.0) * 100.0
    dd_last = _safe_last(dd_from_high)

    vol20_avg = vol.rolling(20).mean()
    vol_ratio = _safe_last(vol / vol20_avg)

    out.update({
        "close_last": c_last,
        "sma20_last": _safe_last(sma20),
        "sma50_last": _safe_last(sma50),
        "sma200_last": _safe_last(sma200),
        "rsi14_last": _safe_last(rsi14),
        "macd_hist_last": _safe_last(macd_hist),
        "ret_1m": ret_1m,
        "ret_3m": ret_3m,
        "ret_6m": ret_6m,
        "vol20": vol20,
        "dd_from_52w_high": dd_last,
        "vol_ratio": vol_ratio,
    })
    return out

def indicator_score(features: dict) -> tuple[int, dict]:
    """
    Returns (IndicatorScore 0..100, breakdown dict)
    Weights:
      Trend 30, Momentum 25, RSI 15, Volume 10, Risk 20
    """
    if not features:
        return 0, {"reason": "Insufficient price history"}

    score = 0
    breakdown = {}

    close = features.get("close_last")
    sma20 = features.get("sma20_last")
    sma50 = features.get("sma50_last")
    sma200 = features.get("sma200_last")
    rsi = features.get("rsi14_last")
    macd_hist = features.get("macd_hist_last")
    ret_1m = features.get("ret_1m")
    ret_3m = features.get("ret_3m")
    ret_6m = features.get("ret_6m")
    vol_ratio = features.get("vol_ratio")
    vol20 = features.get("vol20")
    dd = features.get("dd_from_52w_high")

    # Trend (30)
    trend_pts = 0
    if pd.notna(close) and pd.notna(sma50) and close > sma50:
        trend_pts += 12
    if pd.notna(sma20) and pd.notna(sma50) and sma20 > sma50:
        trend_pts += 8
    if pd.notna(sma50) and pd.notna(sma200) and sma50 > sma200:
        trend_pts += 10
    score += trend_pts
    breakdown["trend_pts"] = trend_pts

    # Momentum (25)
    mom_pts = 0
    for r, pts in [(ret_1m, 7), (ret_3m, 9), (ret_6m, 9)]:
        if pd.notna(r):
            if r > 10:
                mom_pts += pts
            elif r > 0:
                mom_pts += int(pts * 0.6)
            elif r > -5:
                mom_pts += int(pts * 0.25)
    if pd.notna(macd_hist) and macd_hist > 0:
        mom_pts += 2
    mom_pts = min(25, mom_pts)
    score += mom_pts
    breakdown["momentum_pts"] = mom_pts

    # RSI health (15)
    rsi_pts = 0
    if pd.notna(rsi):
        if 45 <= rsi <= 65:
            rsi_pts = 15
        elif 35 <= rsi < 45 or 65 < rsi <= 75:
            rsi_pts = 10
        elif 25 <= rsi < 35 or 75 < rsi <= 85:
            rsi_pts = 5
    score += rsi_pts
    breakdown["rsi_pts"] = rsi_pts

    # Volume confirmation (10)
    vol_pts = 0
    if pd.notna(vol_ratio):
        if vol_ratio >= 1.5:
            vol_pts = 10
        elif vol_ratio >= 1.1:
            vol_pts = 7
        elif vol_ratio >= 0.9:
            vol_pts = 4
    score += vol_pts
    breakdown["volume_pts"] = vol_pts

    # Risk control (20)
    risk_pts = 0
    if pd.notna(vol20):
        if vol20 <= 0.015:
            risk_pts += 10
        elif vol20 <= 0.025:
            risk_pts += 6
        elif vol20 <= 0.04:
            risk_pts += 3
    if pd.notna(dd):
        if dd >= -10:
            risk_pts += 10
        elif dd >= -25:
            risk_pts += 7
        elif dd >= -45:
            risk_pts += 4
    risk_pts = min(20, risk_pts)
    score += risk_pts
    breakdown["risk_pts"] = risk_pts

    score = int(max(0, min(100, round(score))))
    breakdown["indicator_score"] = score
    return score, breakdown

def combined_score(indicator: int, news: int, w_ind: float = 0.8, w_news: float = 0.2) -> int:
    indicator = int(max(0, min(100, indicator)))
    news = int(max(0, min(100, news)))
    cs = w_ind * indicator + w_news * news
    return int(max(0, min(100, round(cs))))
