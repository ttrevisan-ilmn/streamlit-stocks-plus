
import pandas as pd
import numpy as np
import yfinance as yf
from yahooquery import Ticker
import streamlit as st
from datetime import datetime
from services.logger import setup_logger
logger = setup_logger(__name__)

# Cached function to get universe
@st.cache_data(ttl=3600*24) # Cache for 24 hours
def get_screener_universe():
    try:
        df = pd.read_csv('tickers.csv')
        # Ensure Symbol column exists
        if 'Symbol' in df.columns:
            return df['Symbol'].tolist()
        elif 'Ticker' in df.columns:
            return df['Ticker'].tolist()
        return []
    except Exception:
        # Fallback to a smaller list if file missing
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "UNH", "JNJ"]

@st.cache_data(ttl=3600*6) # Cache for 6 hours
def fetch_screener_data(tickers, limit=None):
    """
    Batch fetch fundamental + price + technical data for screener.
    Optimized: 2 yahooquery calls per chunk, 3mo history, threading enabled.
    """
    if not tickers:
        return pd.DataFrame()

    if limit:
        tickers = tickers[:limit]

    CHUNK_SIZE = 50  # Larger chunks = fewer round trips
    all_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_chunks = (len(tickers) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i + CHUNK_SIZE]
        current_chunk_idx = i // CHUNK_SIZE + 1
        status_text.text(f"Scanning batch {current_chunk_idx}/{total_chunks} ({len(chunk)} tickers)...")
        progress_bar.progress(current_chunk_idx / total_chunks)
        
        try:
            # --- STEP 1: Single yahooquery call with 2 modules (not 4) ---
            # financial_data covers: price, ROE, margins, growth, recommendations
            # key_stats covers: beta, float, PE, marketcap
            yq = Ticker(chunk, asynchronous=False)
            financials = yq.financial_data   # ROE, margins, growth, price, recs
            stats = yq.key_stats             # beta, float, PE, earningsQuarterlyGrowth
            
            df_fin = pd.DataFrame(financials).T
            df_stats = pd.DataFrame(stats).T

            # Helper to safely extract numeric from df
            def safe_num(df, col):
                if col in df.columns:
                    return pd.to_numeric(df[col], errors='coerce')
                return pd.Series(dtype=float, index=df.index)

            chunk_data = pd.DataFrame(index=chunk)
            chunk_data['Price']     = safe_num(df_fin, 'currentPrice')
            chunk_data['MarketCap'] = safe_num(df_stats, 'enterpriseValue')
            chunk_data['Beta']      = safe_num(df_stats, 'beta')
            chunk_data['PE']        = safe_num(df_stats, 'forwardPE')
            chunk_data['Float']     = safe_num(df_stats, 'floatShares')
            chunk_data['DivYield']  = safe_num(df_stats, 'lastDividendValue')
            chunk_data['AvgVol']    = safe_num(df_stats, 'sharesOutstanding')  # Approximation if vol missing
            # Navellier Fundamentals
            chunk_data['ROE']       = safe_num(df_fin, 'returnOnEquity')
            chunk_data['OpMargin']  = safe_num(df_fin, 'operatingMargins')
            chunk_data['RevGrowth'] = safe_num(df_fin, 'revenueGrowth')
            chunk_data['EarnGrowth']= safe_num(df_fin, 'earningsGrowth')
            # Analyst
            chunk_data['AnalystRec']= safe_num(df_fin, 'recommendationMean')
            # Earnings growth from key_stats
            chunk_data['EQGrowth']  = safe_num(df_stats, 'earningsQuarterlyGrowth')
            
            # --- STEP 2: Historical Data for Technicals (3 months, threaded) ---
            hist_data = yf.download(
                chunk, period="3mo", interval="1d",
                group_by='ticker', progress=False,
                threads=True,    # Use threading for parallel downloads
                auto_adjust=True
            )
            
            rsi_d, sma50_d, sma200_d, hv_d = {}, {}, {}, {}
            ema8_d, ema21_d, ema34_d, ema55_d, ema89_d = {}, {}, {}, {}, {}
            adx_d, stochk_d = {}, {}
            vol_d, change_d = {}, {}

            for ticker in chunk:
                try:
                    if isinstance(hist_data.columns, pd.MultiIndex):
                        if ticker not in hist_data.columns.get_level_values(0):
                            continue
                        df_t = hist_data.xs(ticker, axis=1, level=0).copy()
                    else:
                        df_t = hist_data.copy()
                    
                    df_t = df_t.dropna(subset=['Close'])
                    if len(df_t) < 20:
                        continue
                    
                    close = df_t['Close'].squeeze()
                    high  = df_t['High'].squeeze()
                    low   = df_t['Low'].squeeze()
                    volume= df_t['Volume'].squeeze()

                    # Volume
                    vol_d[ticker] = volume.iloc[-1]
                    # Change %
                    if len(close) >= 2:
                        change_d[ticker] = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100

                    # RSI (14)
                    delta = close.diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rsi_d[ticker] = (100 - 100 / (1 + gain / loss)).iloc[-1]

                    # SMAs
                    sma50_d[ticker]  = close.rolling(50).mean().iloc[-1]
                    sma200_d[ticker] = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma50_d[ticker]
                    hv = np.log(close / close.shift(1)).rolling(20).std().iloc[-1] * np.sqrt(252) * 100
                    hv_d[ticker] = hv

                    # EMAs
                    ema8_d[ticker]  = close.ewm(span=8,  adjust=False).mean().iloc[-1]
                    ema21_d[ticker] = close.ewm(span=21, adjust=False).mean().iloc[-1]
                    ema34_d[ticker] = close.ewm(span=34, adjust=False).mean().iloc[-1]
                    ema55_d[ticker] = close.ewm(span=55, adjust=False).mean().iloc[-1]
                    ema89_d[ticker] = close.ewm(span=89, adjust=False).mean().iloc[-1]

                    # ADX
                    pdm = high.diff().clip(lower=0)
                    ndm = (-low.diff()).clip(lower=0)
                    tr  = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
                    atr = tr.rolling(14).mean()
                    pdi = 100 * pdm.ewm(alpha=1/14).mean() / atr
                    ndi = 100 * ndm.ewm(alpha=1/14).mean() / atr
                    dx  = (pdi - ndi).abs() / (pdi + ndi).abs() * 100
                    adx_d[ticker] = dx.rolling(14).mean().iloc[-1]

                    # Stoch %K
                    k = 100 * (close - low.rolling(8).min()) / (high.rolling(8).max() - low.rolling(8).min())
                    stochk_d[ticker] = k.rolling(3).mean().iloc[-1]

                except Exception:
                    continue

            chunk_data['Volume']  = pd.Series(vol_d)
            chunk_data['Change%'] = pd.Series(change_d)
            chunk_data['PreMkt%'] = 0.0  # Not available without live feed
            chunk_data['RSI']     = pd.Series(rsi_d)
            chunk_data['SMA50']   = pd.Series(sma50_d)
            chunk_data['SMA200']  = pd.Series(sma200_d)
            chunk_data['HV_20']   = pd.Series(hv_d)
            chunk_data['EMA8']    = pd.Series(ema8_d)
            chunk_data['EMA21']   = pd.Series(ema21_d)
            chunk_data['EMA34']   = pd.Series(ema34_d)
            chunk_data['EMA55']   = pd.Series(ema55_d)
            chunk_data['EMA89']   = pd.Series(ema89_d)
            chunk_data['ADX']     = pd.Series(adx_d)
            chunk_data['StochK']  = pd.Series(stochk_d)

            all_data.append(chunk_data)

        except Exception as e:
            logger.info(f"Error processing chunk {current_chunk_idx}: {e}")
            continue

    # Cleanup UI
    progress_bar.empty()
    status_text.empty()
    
    if not all_data:
        return pd.DataFrame()
        
    # Combine all chunks
    full_df = pd.concat(all_data)
    return full_df.dropna(subset=['Price'])

def apply_strategy(df, strategy):
    """
    Filter DataFrame based on strategy presets.
    """
    if df.empty: return df
    
    filtered = df.copy()
    
    if strategy == "Cash Secured Puts (CSP)":
        # Strategy: High Volatility (HV > 40), Uptrend (Price > SMA50), Pullback (RSI < 50)
        mask = (
            (filtered['HV_20'] > 30) & 
            (filtered['Price'] > filtered['SMA50']) & 
            (filtered['RSI'] < 55) & 
            (filtered['RSI'] > 30) # Don't catch falling knives perfectly
        )
        filtered = filtered[mask].sort_values('HV_20', ascending=False)
        
    elif strategy == "Covered Calls (CC)":
        # Strategy: Moderate Volatility, Strong Trend, RSI Neutral/High
        mask = (
            (filtered['HV_20'] > 20) & 
            (filtered['HV_20'] < 60) &
            (filtered['Price'] > filtered['SMA50']) &
            (filtered['RSI'] > 50)
        )
        filtered = filtered[mask].sort_values('DivYield', ascending=False)
        
    elif strategy == "Short Momentum":
        # Strategy: Downtrend, RSI breaking down
        mask = (
            (filtered['Price'] < filtered['SMA50']) &
            (filtered['RSI'] < 40)
        )
        filtered = filtered[mask].sort_values('RSI', ascending=True)
        
    elif strategy == "Mid Momentum":
        # Strategy: Strong Uptrend (SMA20? approx by Price/SMA50 gap), RSI Bullish
        mask = (
            (filtered['Price'] > filtered['SMA50']) &
            (filtered['RSI'] > 55) & (filtered['RSI'] < 75)
        )
        filtered = filtered[mask].sort_values('RSI', ascending=False)
        
    elif strategy == "Safe Long":
        # Strategy: Low Beta, Dividend, Long Term Uptrend
        mask = (
            (filtered['Beta'] < 1.0) &
            (filtered['DivYield'] > 0.02) &
            (filtered['Price'] > filtered['SMA200'])
        )
        filtered = filtered[mask].sort_values('DivYield', ascending=False)
        
    elif strategy == "Ultimate Stacked Bulls": 
        # "Tao Bull" Swing Strategy
        # 1. EMA 8 > 21 > 34 > 55 > 89
        # 2. ADX > 20
        # 3. AvgVol > 1M
        # 4. Sort by Stochastic %K Ascending
        
        # Check alignment
        aligned = (
            (filtered['EMA8'] > filtered['EMA21']) &
            (filtered['EMA21'] > filtered['EMA34']) &
            (filtered['EMA34'] > filtered['EMA55']) &
            (filtered['EMA55'] > filtered['EMA89'])
        )
        
        mask = (
            aligned &
            (filtered['ADX'] > 20) &
            (filtered['AvgVol'] > 1_000_000)
        )
        filtered = filtered[mask].sort_values('StochK', ascending=True) # Find the ones pulling back
        
    elif strategy == "Day Trade Runners":
        # "Premarket Screener" / Intraday Runner
        # 1. Float < 50M
        # 2. Price < $20
        # 3. Change% > 20% OR PreMkt% > 20%
        # 4. RVOL > 4.0
        
        # Calculate RVOL if not present? We fetched Volume and AvgVol
        # RVOL = Volume / AvgVol
        filtered['RVOL'] = filtered['Volume'] / filtered['AvgVol']
        
        mask = (
            (filtered['Float'] < 50_000_000) &
            (filtered['Price'] < 20.0) &
            (filtered['RVOL'] > 4.0) &
            ((filtered['Change%'] > 20.0) | (filtered['PreMkt%'] > 20.0))
        )
        filtered = filtered[mask].sort_values('Change%', ascending=False)
        
    elif strategy == "Navellier A-Rated Growth":
        # Screen for stocks exhibiting strong fundamentals and momentum
        # Criteria match the Navellier 'A' methodology
        # Technical confirmation: Price > SMA50 (Uptrend) and RSI > 50 (Momentum)
        
        if 'RVOL' not in filtered:
            filtered['RVOL'] = filtered['Volume'] / filtered['AvgVol']
            
        mask = (
            (filtered['ROE'].fillna(0) > 0.05) &              # Return on equity > 5%
            (filtered['OpMargin'].fillna(0) > 0.05) &         # Operating Margins > 5%
            (filtered['RevGrowth'].fillna(0) > 0.0) &         # Positive Sales growth
            (filtered['Price'] > filtered['SMA50'])           # Positive Trend
        )
        filtered = filtered[mask].sort_values('RevGrowth', ascending=False)

    return filtered.head(40) # Return top 40 results
