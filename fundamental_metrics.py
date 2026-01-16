"""
Fundamental Analysis Module (Inspired by ROIC.ai)
Fetches key fundamental quality and valuation metrics.
"""

import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_fundamental_data(ticker):
    """Fetch key fundamental metrics for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract key metrics with safe defaults
        metrics = {
            'quote_type': info.get('quoteType'),
            'long_name': info.get('longName'),
            'market_cap': info.get('marketCap', 0),
            'enterprise_value': info.get('enterpriseValue', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
            'price_to_book': info.get('priceToBook', 0),
            
            # Efficiency & Returns
            'roic': info.get('returnOnAssets', 0) * 1.5, # Rough proxy if ROIC missing
            'roe': info.get('returnOnEquity', 0),
            'roa': info.get('returnOnAssets', 0),
            'operating_margin': info.get('operatingMargins', 0),
            'profit_margin': info.get('profitMargins', 0),
            
            # Cash Flow & Balance Sheet
            'fcf': info.get('freeCashflow', 0),
            'operating_cash_flow': info.get('operatingCashflow', 0),
            'total_cash': info.get('totalCash', 0),
            'total_debt': info.get('totalDebt', 0),
            'quick_ratio': info.get('quickRatio', 0),
            'current_ratio': info.get('currentRatio', 0),
            
            # Growth
            'revenue_growth': info.get('revenueGrowth', 0),
            'earnings_growth': info.get('earningsGrowth', 0),
        }
        
        # Refine ROIC if possible (NOPAT / Invested Capital)
        # Often not directly available, so we stick to proxies or check for it
        if 'returnOnInvestedCapital' in info: # Occasionally present
            metrics['roic'] = info['returnOnInvestedCapital']
            
        return metrics
        
    except Exception as e:
        print(f"Error fetching fundamental data: {e}")
        return None


def format_large_number(num):
    """Format large numbers (Billions, Trillions)."""
    if num >= 1e12:
        return f"{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    else:
        return f"{num:,.0f}"

if __name__ == '__main__':
    # Test
    print("Fetching TSM data...")
    data = fetch_fundamental_data("TSM")
    if data:
        for k, v in data.items():
            print(f"{k}: {v}")
