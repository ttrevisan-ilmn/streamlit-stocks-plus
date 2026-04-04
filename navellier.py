import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from services.logger import setup_logger

logger = setup_logger(__name__)

def normalize(value, min_val, max_val, invert=False):
    """Normalize a value to 0-100 scale based on range."""
    if value is None or np.isnan(value):
        return 50 # Neutral if missing
    
    score = (value - min_val) / (max_val - min_val) * 100
    score = max(0, min(100, score)) # Clamp
    
    if invert:
        return 100 - score
    return score

def calculate_fundamental_grade(info):
    """
    Calculate Fundamental Grade based on 8 metrics.
    """
    metrics = {}
    
    # 1. Sales Growth (Higher is better)
    metrics['Sales Growth'] = {
        'value': info.get('revenueGrowth', None),
        'score': normalize(info.get('revenueGrowth', None), -0.1, 0.4)
    }
    
    # 2. Operating Margin (Higher is better)
    metrics['Operating Margin'] = {
        'value': info.get('operatingMargins', None),
        'score': normalize(info.get('operatingMargins', None), 0, 0.3)
    }
    
    # 3. Earnings Growth
    metrics['Earnings Growth'] = {
        'value': info.get('earningsQuarterlyGrowth', None),
        'score': normalize(info.get('earningsQuarterlyGrowth', None), -0.2, 0.5)
    }
    
    # 4. Earnings Momentum (Using trailing vs forward P/E proxy)
    fpe = info.get('forwardPE', None)
    tpe = info.get('trailingPE', None)
    ratio = None
    if fpe and tpe and fpe > 0:
        ratio = tpe / fpe # > 1 implies forward earnings are expected to be higher
    metrics['Earnings Momentum'] = {
        'value': ratio,
        'score': normalize(ratio, 0.8, 1.5) if ratio is not None else normalize(info.get('pegRatio'), 0.5, 2, invert=True)
    }
    
    # 5. Earnings Surprises (Proxy: EPS growth trend / general EPS momentum)
    eg = info.get('earningsGrowth', None)
    metrics['Earnings Surprises'] = {
        'value': eg,
        'score': normalize(eg, -0.1, 0.4)
    }
    
    # 6. Analyst Earnings Revisions (Analyst Recommendation Mean: 1 is Strong Buy)
    rec = info.get('recommendationMean', None)
    metrics['Analyst Revisions'] = {
        'value': rec,
        'score': normalize(rec, 1.5, 3.5, invert=True)
    }
    
    # 7. Cash Flow (FCF Yield Proxy)
    fcf = info.get('freeCashflow', None)
    mcap = info.get('marketCap', 1)
    fcf_yield = (fcf / mcap) if fcf and mcap and mcap > 0 else None
    metrics['Cash Flow'] = {
        'value': fcf_yield,
        'score': normalize(fcf_yield, 0, 0.05)
    }
    
    # 8. Return on Equity
    roe = info.get('returnOnEquity', None)
    metrics['Return on Equity'] = {
        'value': roe,
        'score': normalize(roe, 0.05, 0.25)
    }
    
    fundamental_score = np.mean([data['score'] for data in metrics.values()])
    
    return {
        'score': fundamental_score,
        'metrics': metrics
    }

def calculate_quantitative_grade(history):
    """
    Calculate Quantitative Grade evaluating Buying Pressure (Momentum).
    """
    if history is None or history.empty or len(history) < 21:
        return {'score': 50, 'metrics': {'Buying Pressure': {'value': None, 'score': 50}}}
    
    close = history['Close']
    volume = history['Volume']
    high = history['High']
    low = history['Low']
    
    # Chaikin Money Flow (CMF) - 21 Day
    mfv = ((close - low) - (high - close)) / (high - low) * volume
    mfv = mfv.fillna(0)
    cmf = mfv.rolling(21).sum() / volume.rolling(21).sum()
    cmf_val = cmf.iloc[-1]
    
    # Relative Volume (5d vs 20d)
    vol_5 = volume.rolling(5).mean().iloc[-1]
    vol_20 = volume.rolling(20).mean().iloc[-1]
    rvol = vol_5 / vol_20 if vol_20 > 0 else 1
    
    # Price Momentum (20 day max vs current)
    # Are we near the highs?
    recent_high = high.rolling(20).max().iloc[-1]
    dist_from_high = (close.iloc[-1] / recent_high) if recent_high > 0 else 1
    
    cmf_score = normalize(cmf_val, -0.2, 0.2)
    rvol_score = normalize(rvol, 0.8, 1.5)
    dist_score = normalize(dist_from_high, 0.85, 1.0)
    
    # Combine scores for Buying Pressure
    quant_score = (cmf_score * 0.5) + (rvol_score * 0.2) + (dist_score * 0.3)
    
    return {
        'score': quant_score,
        'metrics': {
            'Buying Pressure (CMF)': {'value': cmf_val, 'score': cmf_score},
            'Rel Vol (Strength)': {'value': rvol, 'score': rvol_score},
            'Price vs 20d High': {'value': dist_from_high, 'score': dist_score}
        }
    }

def get_letter_grade(score):
    if score >= 80: return "A (Strong Buy)"
    if score >= 60: return "B (Buy)"
    if score >= 40: return "C (Hold)"
    if score >= 20: return "D (Sell)"
    return "F (Strong Sell)"

def get_color_for_grade(grade):
    if grade.startswith("A"): return "#4ade80" # Green
    if grade.startswith("B"): return "#86efac" # Light Green
    if grade.startswith("C"): return "#fbbf24" # Yellow
    if grade.startswith("D"): return "#f87171" # Red
    if grade.startswith("F"): return "#ef4444" # Strong Red
    return "gray"

@st.cache_data(ttl=3600*12)
def calculate_navellier_grader(ticker):
    """
    Calculates the Louis Navellier Portfolio Grader scores.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        history = t.history(period="6mo")
        
        fund_grade = calculate_fundamental_grade(info)
        quant_grade = calculate_quantitative_grade(history)
        
        # Total Grade (Equal weighted for now)
        total_score = (fund_grade['score'] + quant_grade['score']) / 2
        
        return {
            "total_score": total_score,
            "total_grade": get_letter_grade(total_score),
            "fundamental_grade": {
                "score": fund_grade['score'],
                "grade": get_letter_grade(fund_grade['score']),
                "metrics": fund_grade['metrics']
            },
            "quantitative_grade": {
                "score": quant_grade['score'],
                "grade": get_letter_grade(quant_grade['score']),
                "metrics": quant_grade['metrics']
            }
        }
        
    except Exception as e:
        logger.error(f"Navellier Grader Error: {e}")
        return None
