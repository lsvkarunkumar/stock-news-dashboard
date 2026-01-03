import json
import numpy as np
import pandas as pd

def _atr_like(close: pd.Series) -> float:
    # crude volatility proxy for V1
    r = close.pct_change().dropna()
    return float(r.tail(20).std() * np.sqrt(252)) if len(r) > 20 else float(r.std() * np.sqrt(252)) if len(r) else 0.0

def compute_score(mentions_5: int, mentions_30: int, mentions_60: int, prices: pd.DataFrame) -> tuple[int, str]:
    reasons = {}

    # --- Strategy 1: News intensity + confirmation ---
    m5 = mentions_5 / 5.0
    m30 = mentions_30 / 30.0
    m60 = mentions_60 / 60.0
    intensity = 0.60*m5 + 0.30*m30 + 0.10*m60
    intensity_score = int(min(40, round(intensity * 40)))  # up to 40 points
    reasons["news_intensity"] = {"m5": mentions_5, "m30": mentions_30, "m60": mentions_60, "score": intensity_score}

    # --- Strategy 2: Trend filter ---
    trend_score = 0
    if not prices.empty and prices["close"].notna().any():
        c = prices["close"].astype(float)
        ma20 = c.rolling(20).mean().iloc[-1]
        ma50 = c.rolling(50).mean().iloc[-1]
        last = c.iloc[-1]
        if np.isfinite(ma20) and np.isfinite(ma50):
            if last > ma20 > ma50:
                trend_score = 30
            elif last > ma20:
                trend_score = 18
            elif last > ma50:
                trend_score = 10
    reasons["trend"] = {"score": trend_score}

    # --- Strategy 3: Risk/volatility guardrail ---
    risk_score = 0
    if not prices.empty and prices["close"].notna().any():
        c = prices["close"].astype(float)
        vol = _atr_like(c)
        # lower vol = better (for “safer” entries)
        if vol < 0.25:
            risk_score = 30
        elif vol < 0.40:
            risk_score = 18
        else:
            risk_score = 8
        reasons["risk"] = {"ann_vol_proxy": round(vol, 3), "score": risk_score}
    else:
        reasons["risk"] = {"ann_vol_proxy": None, "score": 0}

    total = int(max(0, min(100, intensity_score + trend_score + risk_score)))
    return total, json.dumps(reasons, ensure_ascii=False)
