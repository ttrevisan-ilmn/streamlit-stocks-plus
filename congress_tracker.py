"""
Congressional Trading Tracker
Fetches stock trading disclosures from Congress members using congress.gov API.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

# Congress.gov API key
BASE_URL = "https://api.congress.gov/v3"


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_congress_members(api_key=None):
    """Fetch current Congress members."""
    # Prioritize passed key, then secrets
    if not api_key:
        api_key = st.secrets.get("congress_api_key")
        
    if not api_key:
        # Graceful fallback or warning if key is missing
        return pd.DataFrame()

    try:
        url = f"{BASE_URL}/member"
        params = {
            "api_key": api_key,
            "limit": 250,
            "currentMember": "true",
            "format": "json"
        }
        # Add basic UA to avoid weak bot blocks
        headers = {'User-Agent': 'Streamlit-Stock-Analysis-App/1.0'}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            members = data.get('members', [])
            
            # Parse into DataFrame
            records = []
            for member in members:
                # Parse terms safely
                terms = member.get('terms')
                chamber = None
                if isinstance(terms, list) and terms:
                    chamber = terms[-1].get('chamber')
                elif isinstance(terms, dict):
                    chamber = terms.get('chamber')

                records.append({
                    'bioguideId': member.get('bioguideId'),
                    'name': member.get('name'),
                    'party': member.get('partyName'),
                    'state': member.get('state'),
                    'district': member.get('district'),
                    'chamber': chamber
                })
            
            return pd.DataFrame(records)
        else:
            st.error(f"Congress.gov API Error {response.status_code}: {response.reason}")
            # print(f"API Response: {response.text}") # Debugging
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Congress API Connection Failed")
        st.exception(e)
        return pd.DataFrame()


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_stock_disclosures():
    """
    Fetch recent stock trading disclosures.
    Note: Congress.gov API provides legislative data. For actual stock trades,
    we'd need to web scrape or use a dedicated service. This is a mock implementation
    that provides realistic example data structure.
    """
    # Since congress.gov API doesn't directly provide STOCK Act disclosures,
    # we'll create a mock dataset that represents what the data would look like.
    # In production, this would connect to House/Senate disclosure databases.
    
    # Example stock trades (realistic structure)
    mock_trades = [
        {"member": "Nancy Pelosi", "party": "D", "state": "CA", "ticker": "NVDA", 
         "transaction": "Purchase", "amount": "$1,000,001 - $5,000,000", 
         "date": "2026-01-03", "disclosure_date": "2026-01-06"},
        {"member": "Nancy Pelosi", "party": "D", "state": "CA", "ticker": "AAPL", 
         "transaction": "Purchase", "amount": "$500,001 - $1,000,000", 
         "date": "2026-01-02", "disclosure_date": "2026-01-05"},
        {"member": "Dan Crenshaw", "party": "R", "state": "TX", "ticker": "TSLA", 
         "transaction": "Sale", "amount": "$100,001 - $250,000", 
         "date": "2025-12-28", "disclosure_date": "2026-01-04"},
        {"member": "Josh Gottheimer", "party": "D", "state": "NJ", "ticker": "MSFT", 
         "transaction": "Purchase", "amount": "$250,001 - $500,000", 
         "date": "2025-12-27", "disclosure_date": "2026-01-03"},
        {"member": "Tommy Tuberville", "party": "R", "state": "AL", "ticker": "META", 
         "transaction": "Purchase", "amount": "$50,001 - $100,000", 
         "date": "2025-12-26", "disclosure_date": "2026-01-02"},
        {"member": "Mark Green", "party": "R", "state": "TN", "ticker": "XOM", 
         "transaction": "Sale", "amount": "$15,001 - $50,000", 
         "date": "2025-12-23", "disclosure_date": "2025-12-30"},
        {"member": "Michael McCaul", "party": "R", "state": "TX", "ticker": "BA", 
         "transaction": "Purchase", "amount": "$100,001 - $250,000", 
         "date": "2025-12-20", "disclosure_date": "2025-12-27"},
        {"member": "Alexandria Ocasio-Cortez", "party": "D", "state": "NY", "ticker": "AMZN", 
         "transaction": "Purchase", "amount": "$1,001 - $15,000", 
         "date": "2025-12-18", "disclosure_date": "2025-12-24"},
        {"member": "Kevin Hern", "party": "R", "state": "OK", "ticker": "CVX", 
         "transaction": "Purchase", "amount": "$50,001 - $100,000", 
         "date": "2025-12-15", "disclosure_date": "2025-12-22"},
        {"member": "Ro Khanna", "party": "D", "state": "CA", "ticker": "GOOGL", 
         "transaction": "Purchase", "amount": "$15,001 - $50,000", 
         "date": "2025-12-12", "disclosure_date": "2025-12-19"},
    ]
    
    return pd.DataFrame(mock_trades)


def get_top_traded_tickers(trades_df, n=10):
    """Get the most frequently traded tickers by Congress."""
    if trades_df.empty:
        return pd.DataFrame()
    
    ticker_counts = trades_df.groupby('ticker').agg({
        'transaction': 'count',
        'member': lambda x: list(x.unique())
    }).reset_index()
    
    ticker_counts.columns = ['ticker', 'trade_count', 'members']
    ticker_counts = ticker_counts.sort_values('trade_count', ascending=False).head(n)
    
    return ticker_counts


def get_active_traders(trades_df, n=10):
    """Get the most active Congress traders."""
    if trades_df.empty:
        return pd.DataFrame()
    
    trader_counts = trades_df.groupby(['member', 'party', 'state']).agg({
        'ticker': 'count'
    }).reset_index()
    
    trader_counts.columns = ['member', 'party', 'state', 'trade_count']
    trader_counts = trader_counts.sort_values('trade_count', ascending=False).head(n)
    
    return trader_counts


def check_watchlist_overlap(trades_df, watchlist):
    """Check if any Congressional trades overlap with user's watchlist."""
    if trades_df.empty or not watchlist:
        return pd.DataFrame()
    
    watchlist_upper = [t.upper() for t in watchlist]
    overlap = trades_df[trades_df['ticker'].str.upper().isin(watchlist_upper)]
    
    return overlap


if __name__ == '__main__':
    print("\nCongressional Trading Tracker Test")
    print("=" * 60)
    
    # Test fetching trades
    trades = fetch_stock_disclosures()
    print(f"\nRecent Trades: {len(trades)} found")
    print(trades.head())
    
    # Test top tickers
    top_tickers = get_top_traded_tickers(trades)
    print("\nTop Traded Tickers:")
    print(top_tickers)
    
    # Test active traders
    active = get_active_traders(trades)
    print("\nMost Active Traders:")
    print(active)
    
    # Test watchlist
    overlap = check_watchlist_overlap(trades, ['NVDA', 'AAPL', 'TSLA'])
    print("\nWatchlist Overlap (NVDA, AAPL, TSLA):")
    print(overlap)
