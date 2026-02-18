
import pandas as pd
import numpy as np
import yfinance as yf
from yahooquery import Ticker
import streamlit as st
from datetime import datetime

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

@st.cache_data(ttl=3600*4) # Cache for 4 hours
def fetch_screener_data(tickers):
    """
    Batch fetch fundamental and price data for screener.
    Refactored to use chunking and synchronous calls to prevent "Too many open files" error.
    """
    if not tickers:
        return pd.DataFrame()

    CHUNK_SIZE = 50
    all_data = []
    
    # Progress bar for long operations
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_chunks = (len(tickers) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i + CHUNK_SIZE]
        current_chunk_idx = i // CHUNK_SIZE + 1
        status_text.text(f"Scanning batch {current_chunk_idx}/{total_chunks}...")
        progress_bar.progress(current_chunk_idx / total_chunks)
        
        try:
            # 1. Fetch Fundamentals (Synchronous to save file descriptors)
            # Remove asynchronous=True to avoid OSError: Too many open files
            yq = Ticker(chunk, asynchronous=False) 
            
            summary = yq.summary_detail
            stats = yq.key_stats
            price = yq.price
            
            # Convert to DataFrames
            df_summary = pd.DataFrame(summary).T
            df_stats = pd.DataFrame(stats).T
            df_price = pd.DataFrame(price).T
            
            # Merge key fields
            chunk_data = pd.DataFrame(index=chunk)
            
            # Helper to safely extract column
            def safe_extract(df, col):
                if col in df.columns:
                    return df[col]
                return pd.Series(index=df.index)

            # Helper to safely extract numeric
            def safe_extract_numeric(df, col):
                series = safe_extract(df, col)
                return pd.to_numeric(series, errors='coerce')

            chunk_data['Price'] = safe_extract_numeric(df_price, 'regularMarketPrice')
            chunk_data['MarketCap'] = safe_extract_numeric(df_summary, 'marketCap')
            chunk_data['Beta'] = safe_extract_numeric(df_stats, 'beta')
            chunk_data['PE'] = safe_extract_numeric(df_summary, 'trailingPE')
            chunk_data['DivYield'] = safe_extract_numeric(df_summary, 'dividendYield')
            chunk_data['Volume'] = safe_extract_numeric(df_summary, 'volume')
            chunk_data['AvgVol'] = safe_extract_numeric(df_summary, 'averageVolume')
            chunk_data['Sector'] = safe_extract(df_summary, 'section')
            
            # 2. Fetch History for Technicals (Trend, RSI, Volatility)
            # Using yfinance for history as it handles multi-ticker adjusted close well
            # Download last 6 months to safe bandwidth
            hist_data = yf.download(chunk, period="6mo", interval="1d", group_by='ticker', progress=False, threads=False) # Disable threads for safety
            
            # Calculate Technicals per ticker
            tech_data = {
                'RSI': {},
                'SMA20': {},
                'SMA50': {},
                'SMA200': {},
                'HV_20': {}, # Historical Volatility
                'Trend_50': {}, # Price vs SMA50
                'Trend_200': {} # Price vs SMA200
            }
            
            for ticker in chunk:
                try:
                    # Handle MultiIndex
                    if isinstance(hist_data.columns, pd.MultiIndex):
                        if ticker in hist_data.columns.get_level_values(0):
                             df_t = hist_data[ticker].copy()
                        else:
                            continue
                    else:
                        # Single ticker case (if chunk has only 1 item)
                        df_t = hist_data.copy()
                    
                    if df_t.empty: continue
                    
                    # Cleanup
                    df_t = df_t.dropna(subset=['Close'])
                    if len(df_t) < 50: continue
                    
                    close = df_t['Close']
                    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0] # Handle rare duplicate col case

                    # RSI
                    delta = close.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    tech_data['RSI'][ticker] = rsi.iloc[-1]
                    
                    # SMAs
                    tech_data['SMA20'][ticker] = close.rolling(window=20).mean().iloc[-1]
                    tech_data['SMA50'][ticker] = close.rolling(window=50).mean().iloc[-1]
                    tech_data['SMA200'][ticker] = close.rolling(window=200).mean().iloc[-1]
                    
                    # Historical Volatility (20D)
                    log_ret = np.log(close / close.shift(1))
                    hv = log_ret.rolling(window=20).std().iloc[-1] * np.sqrt(252) * 100
                    tech_data['HV_20'][ticker] = hv
                    
                    curr_price = close.iloc[-1]
                    tech_data['Trend_50'][ticker] = "Up" if curr_price > tech_data['SMA50'][ticker] else "Down"
                    # Safe check for SMA200 existence
                    sma200_val = tech_data['SMA200'].get(ticker)
                    if sma200_val and not pd.isna(sma200_val):
                         tech_data['Trend_200'][ticker] = "Up" if curr_price > sma200_val else "Down"
                    else:
                         tech_data['Trend_200'][ticker] = "Down"
                    
                except Exception:
                    continue
                    
            # Merge Technicals into chunk_data
            chunk_data['RSI'] = pd.Series(tech_data['RSI'])
            chunk_data['SMA50'] = pd.Series(tech_data['SMA50'])
            chunk_data['SMA200'] = pd.Series(tech_data['SMA200'])
            chunk_data['HV_20'] = pd.Series(tech_data['HV_20'])
            
            all_data.append(chunk_data)
            
        except Exception as e:
            print(f"Error processing chunk {current_chunk_idx}: {e}")
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
        
    return filtered.head(20) # Return top 20
