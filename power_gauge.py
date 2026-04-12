
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from services.logger import setup_logger
logger = setup_logger(__name__)

# --- SCORE NORMALIZATION HELPERS ---
def normalize(value, min_val, max_val, invert=False):
    """Normalize a value to 0-100 scale based on range."""
    if value is None or np.isnan(value):
        return 50 # Neutral if missing
    
    score = (value - min_val) / (max_val - min_val) * 100
    score = max(0, min(100, score)) # Clamp
    
    if invert:
        return 100 - score
    return score

def get_financial_score(info):
    """
    Financial Factors (Balance Sheet Strength) - 5 Factors
    """
    scores = {}
    
    # 1. Debt-to-Equity (Lower is better)
    de = info.get('debtToEquity', None)
    # Range: 0 (Great) to 200 (High Debt)
    scores['Debt/Equity'] = normalize(de, 0, 200, invert=True) 

    # 2. Price-to-Book (Lower is better, value)
    pb = info.get('priceToBook', None)
    # Range: 1 (Value) to 10 (Growth/Expensive)
    scores['Price/Book'] = normalize(pb, 1, 10, invert=True)

    # 3. Return on Equity (Higher is better)
    roe = info.get('returnOnEquity', None)
    # Range: 0.05 (Poor) to 0.25 (Great) - Data is often 0.15 for 15%
    scores['ROE'] = normalize(roe, 0.05, 0.25)

    # 4. Price-to-Sales (Lower is better)
    ps = info.get('priceToSalesTrailing12Months', None)
    # Range: 1 (Cheap) to 10 (Expensive)
    scores['Price/Sales'] = normalize(ps, 1, 10, invert=True)

    # 5. Free Cash Flow Yield (Higher is better)
    # Approximation: Free Cashflow / Market Cap
    fcf = info.get('freeCashflow', None)
    mcap = info.get('marketCap', 1)
    fcf_yield = (fcf / mcap) if fcf is not None and mcap is not None and mcap != 0 else None
    # Range: 0 (None) to 0.05 (5% Yield)
    scores['FCF Yield'] = normalize(fcf_yield, 0, 0.05)
    
    return scores

def get_earnings_score(info):
    """
    Earnings Factors (Performance) - 5 Factors
    """
    scores = {}
    
    # 1. Earnings Growth (Higher better)
    eg = info.get('earningsGrowth', None)
    # Range: -0.1 to 0.5 (50%)
    scores['Growth Rate'] = normalize(eg, -0.1, 0.5)
    
    # 2. Earnings Surprise (Compare actual vs estimate from info if available, else proxy)
    # Proxy: simply check if trailing EPS > forward EPS? No.
    # We'll use 'earningsQuarterlyGrowth' as a proxy for "Surprise/Momentum"
    eqg = info.get('earningsQuarterlyGrowth', None)
    scores['Earnings Surprise'] = normalize(eqg, -0.2, 0.5)

    # 3. Earnings Trend (Revenue Growth as proxy)
    rg = info.get('revenueGrowth', None)
    scores['Earnings Trend'] = normalize(rg, -0.1, 0.4)

    # 4. Projected P/E (Lower is better)
    fpe = info.get('forwardPE', None)
    tpe = info.get('trailingPE', None)
    # If Forward < Trailing, expectations are improving (Bullish)
    if fpe and tpe:
        ratio = tpe / fpe 
        # > 1 means Forward is lower (Cheaper) -> Good
        scores['Projected P/E'] = normalize(ratio, 0.8, 1.5)
    else:
        scores['Projected P/E'] = normalize(fpe, 10, 50, invert=True) # Fallback

    # 5. Earnings Consistency (Margins as proxy for quality)
    pm = info.get('profitMargins', None)
    scores['Consistency'] = normalize(pm, 0.05, 0.25)
    
    return scores

def get_expert_score(info, ticker_obj):
    """
    Expert & Industry Factors (Sentiment) - 5 Factors
    """
    scores = {}
    
    # 1. Estimate Trend (Target Price vs Current)
    current = info.get('currentPrice', 1)
    target = info.get('targetMeanPrice', current)
    upside = (target - current) / current
    # Range: -0.1 to 0.3 (30% upside)
    scores['Analyst Target'] = normalize(upside, -0.1, 0.3)
    
    # 2. Short Interest (Lower is better usually, unless squeeze)
    si = info.get('shortPercentOfFloat', 0)
    # Range: 0 to 0.2 (20% short interest is high)
    scores['Short Interest'] = normalize(si, 0, 0.2, invert=True)
    
    # 3. Insider Activity
    try:
        insider = ticker_obj.insider_transactions
        if not insider.empty:
            # Net shares bought/sold in last 6 months
            recent = insider.sort_values('Start Date', ascending=False).head(10)
            net_shares = recent['Shares'].sum() # Assuming positive is buy? 
            # Actually yfinance insider data structure varies.
            # Simplified: Random walk proxy or neutral if complex
            scores['Insider Activity'] = 50 
        else:
            scores['Insider Activity'] = 50
    except Exception as e:
        logger.info(f"Error fetching insider transactions: {e}")
        scores['Insider Activity'] = 50

    # 4. Analyst Rating
    rec = info.get('recommendationMean', 3) # 1 is Strong Buy, 5 is Sell
    # Range: 1 (Best) to 5 (Worst)
    scores['Analyst Rating'] = normalize(rec, 1.5, 3.5, invert=True)

    # 5. Industry Rel Strength
    # Compare Ticker Beta to 1? (Higher Beta = likely outperforming in bull market)
    beta = info.get('beta', 1)
    scores['Industry Relative'] = normalize(beta, 0.5, 1.5)
    
    return scores

def get_technical_score(ticker, history):
    """
    Technical Factors (Price/Volume) - 5 Factors
    """
    scores = {}
    
    if history.empty or len(history) < 50:
        return {k: 50 for k in ['Rel Strength', 'Chaikin Money Flow', 'Chaikin Trend', 'Price Trend ROC', 'Volume Trend']}

    close = history['Close']
    volume = history['Volume']
    high = history['High']
    low = history['Low']
    
    # 1. Relative Strength vs SPY (Approximate using beta adjustment or just absolute momentum)
    # We will fetch SPY history briefly? No, expensive.
    # Use simple ROC (Rate of Change) 6-month as proxy for "Strength"
    roc_126 = (close.iloc[-1] / close.iloc[-126]) - 1 if len(close) > 126 else 0
    scores['Rel Strength'] = normalize(roc_126, -0.1, 0.3)

    # 2. Chaikin Money Flow (CMF) - 21 Day
    # MFV = ((C - L) - (H - C)) / (H - L) * V
    mfv = ((close - low) - (high - close)) / (high - low) * volume
    mfv = mfv.fillna(0)
    cmf = mfv.rolling(21).sum() / volume.rolling(21).sum()
    cmf_val = cmf.iloc[-1]
    # Range: -0.2 to 0.2
    scores['Chaikin Money Flow'] = normalize(cmf_val, -0.2, 0.2)

    # 3. Chaikin Trend (EMA(3) calc'd on ADL - EMA(10) calc'd on ADL)
    adl = mfv.cumsum()
    chaikin_osc = adl.ewm(span=3, adjust=False).mean() - adl.ewm(span=10, adjust=False).mean()
    # Normalize simply based on positive/negative
    # We need a reference scale for Oscillator. It's volume based, so huge numbers.
    # Compare to moving average of itself?
    # Simplified: Is it rising?
    osc_change = chaikin_osc.diff(5).iloc[-1]
    scores['Chaikin Trend'] = 60 if osc_change > 0 else 40

    # 4. Price Trend ROC (42 day)
    roc_42 = (close.iloc[-1] / close.iloc[-42]) - 1 if len(close) > 42 else 0
    scores['Price Trend ROC'] = normalize(roc_42, -0.1, 0.2)
    
    # 5. Volume Trend (Vol 20 vs Vol 90)
    vol_20 = volume.rolling(20).mean().iloc[-1]
    vol_90 = volume.rolling(90).mean().iloc[-1]
    vol_ratio = vol_20 / vol_90 if vol_90 > 0 else 1
    scores['Volume Trend'] = normalize(vol_ratio, 0.8, 1.5)
    
    return scores

@st.cache_data(ttl=3600*12) # Cache for 12 hours
def calculate_power_gauge(ticker):
    """
    Master function to compute the 20-factor Power Gauge.
    Returns nested dict of scores and final rating.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        history = t.history(period="1y")
        
        financials = get_financial_score(info)
        earnings = get_earnings_score(info)
        experts = get_expert_score(info, t)
        technicals = get_technical_score(ticker, history)
        
        # Aggregate
        categories = {
            "Financials": financials,
            "Earnings": earnings,
            "Experts": experts,
            "Technicals": technicals
        }
        
        # Calculate Category Scores
        cat_scores = {}
        total_score = 0
        
        for cat, metrics in categories.items():
            avg = np.mean(list(metrics.values()))
            cat_scores[cat] = avg
            total_score += avg
            
        final_score = total_score / 4 # Equal weight for now
        
        # Determine Rating
        if final_score >= 65: rating = "BULLISH"
        elif final_score <= 35: rating = "BEARISH"
        else: rating = "NEUTRAL"
        
        # Extract metadata for the detailed UI panels
        metadata = {
            "currentRatio": info.get("currentRatio"),
            "debtToEquity": info.get("debtToEquity", 0) / 100 if info.get("debtToEquity") else None, # Assuming it's given as 29 for 0.29
            "marketCap": info.get("marketCap"),
            "totalRevenue": info.get("totalRevenue"),
            "trailingPE": info.get("trailingPE"),
            "pegRatio": info.get("pegRatio"),
            "priceToBook": info.get("priceToBook"),
            "priceToSales": info.get("priceToSalesTrailing12Months"),
            "dividendYield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
            "annualEps": info.get("trailingEps"),
            "beta": info.get("beta"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "avgVol20Day": history['Volume'].rolling(20).mean().iloc[-1] if not history.empty and len(history)>=20 else None,
            "avgVol90Day": history['Volume'].rolling(90).mean().iloc[-1] if not history.empty and len(history)>=90 else None,
            "chg4wk": ((history['Close'].iloc[-1] / history['Close'].iloc[-20]) - 1) * 100 if not history.empty and len(history)>=20 else None,
            "chg24wk": ((history['Close'].iloc[-1] / history['Close'].iloc[-120]) - 1) * 100 if not history.empty and len(history)>=120 else None
        }
        
        return {
            "rating": rating,
            "score": final_score,
            "categories": cat_scores,
            "details": categories,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.info(f"Power Gauge Error: {e}")
        return None
