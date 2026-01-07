import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Add parent dir to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gamma_profile import get_gamma_profile
from congress_tracker import fetch_congress_members
from seaf_model import get_seaf_model
from options_flow import get_daily_flow_snapshot

class TestCoreModules(unittest.TestCase):
    
    @patch('gamma_profile.get_cached_options_chain')
    def test_gamma_profile_structure(self, mock_get_chain):
        """Verify get_gamma_profile returns the exact keys expected by the UI."""
        # Mock options data
        mock_df = pd.DataFrame({
            'strike': [100, 105, 110],
            'option_type': ['call', 'put', 'call'],
            'open_interest': [1000, 500, 1000],
            'volume': [50, 50, 50],
            'underlying_price': [105, 105, 105],
            'bid': [1.0, 1.0, 1.0],
            'ask': [1.1, 1.1, 1.1]
        })
        mock_get_chain.return_value = mock_df
        
        result = get_gamma_profile("TEST")
        
        # UI expects specific keys. If these are missing, UI crashes.
        self.assertIn('gex', result)
        self.assertIn('volume', result)
        self.assertIn('spot', result)
        self.assertIn('stats', result)
        self.assertIn('timestamp', result)
        
        # Verify 'df' key is NOT expected (since we removed it from UI logic)
        # But if it WAS expected, this test would fail if it's missing.
        # We confirm it is NOT there to be sure our mental model matches code.
        self.assertNotIn('df', result)
        
        # Verify types
        self.assertIsInstance(result['gex'], pd.Series)
        self.assertIsInstance(result['volume'], pd.DataFrame)
        
    @patch('congress_tracker.requests.get')
    def test_congress_api_error_handling(self, mock_get):
        """Verify fetch_congress_members handles API errors properly."""
        # Simulate 403 Forbidden
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.reason = "Forbidden"
        mock_get.return_value = mock_response
        
        # Should return empty DataFrame (and log error to streamlit, which we ignore here)
        result = fetch_congress_members(api_key="BAD_KEY")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    @patch('congress_tracker.requests.get')
    def test_congress_api_success(self, mock_get):
        """Verify fetch_congress_members parses valid response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "members": [
                {"name": "Test Rep", "partyName": "D", "state": "CA", "bioguideId": "T001"}
            ]
        }
        mock_get.return_value = mock_response
        
        result = fetch_congress_members(api_key="GOOD_KEY")
        self.assertFalse(result.empty)
        self.assertIn('name', result.columns)
        self.assertEqual(result.iloc[0]['name'], "Test Rep")

    @patch('seaf_model.fetch_sector_data')
    def test_seaf_model_structure(self, mock_fetch):
        """Verify SEAF model returns proper rankings structure."""
        # Mock sector data returns
        dates = pd.date_range(end=pd.Timestamp.now(), periods=300)
        mock_data = pd.DataFrame({
            'Close': [100 + i for i in range(300)],
            'Volume': [1000 for _ in range(300)]
        }, index=dates)
        mock_fetch.return_value = mock_data
        
        result = get_seaf_model()
        
        # UI expects: 'Rank', 'Ticker', 'Sector', 'Total_Score', 'Category'
        self.assertFalse(result.empty)
        expected_cols = ['Rank', 'Ticker', 'Sector', 'Total_Score', 'Category']
        for col in expected_cols:
            self.assertIn(col, result.columns)

if __name__ == '__main__':
    unittest.main()
