"""
Unit tests for NBA API Client

Tests the core functionality of the NBAApiClient class including:
- Timeout handling
- Error handling
- API call safety mechanisms
- Data processing
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime

from src.data_collection.nba_api_client import NBAApiClient


class TestNBAApiClient:
    """Test suite for NBAApiClient"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.client = NBAApiClient(delay_seconds=0.1, timeout_seconds=2.0)

    def test_init_default_values(self):
        """Test client initialization with default values"""
        client = NBAApiClient()
        assert client.delay_seconds == 1.0
        assert client.timeout_seconds == 30.0

    def test_init_custom_values(self):
        """Test client initialization with custom values"""
        client = NBAApiClient(delay_seconds=0.5, timeout_seconds=10.0)
        assert client.delay_seconds == 0.5
        assert client.timeout_seconds == 10.0

    def test_timeout_handling_slow_function(self):
        """Test that timeout handling works for slow operations"""
        def slow_function():
            print(f"Slow function starting, will sleep 5 seconds...")
            time.sleep(5)  # Much longer than our 2-second timeout
            print(f"Slow function completed (should not see this)")
            return "Should not reach this"

        print(f"Client timeout setting: {self.client.timeout_seconds}s")
        start_time = time.time()
        result = self.client._safe_api_call(slow_function)
        end_time = time.time()
        
        # Verify it actually timed out close to our timeout setting (2 seconds)
        duration = end_time - start_time
        print(f"Actual duration: {duration:.2f}s")
        print(f"Result: {result}")
        
        assert result is None, f"Expected None but got {result}"
        assert 1.8 <= duration <= 2.5, f"Expected ~2s timeout, but took {duration:.2f}s"

    def test_timeout_handling_fast_function(self):
        """Test that fast operations complete successfully"""
        def fast_function():
            return "Success"

        result = self.client._safe_api_call(fast_function)
        assert result == "Success"

    def test_safe_api_call_with_exception(self):
        """Test handling of exceptions in API calls"""
        def failing_function():
            raise ValueError("Test error")

        result = self.client._safe_api_call(failing_function)
        assert result is None

    @patch('src.data_collection.nba_api_client.requests.get')
    def test_requests_timeout_patching(self, mock_get):
        """Test that requests.get gets timeout parameter"""
        def mock_api_call():
            import requests
            requests.get("http://example.com")
            return "success"

        # Mock requests.get to capture the timeout parameter
        mock_get.return_value = Mock()
        
        result = self.client._safe_api_call(mock_api_call)
        
        # Verify timeout was added to the request
        assert result == "success"
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs.get('timeout') == self.client.timeout_seconds

    def test_wait_between_calls(self):
        """Test rate limiting delay"""
        start_time = time.time()
        self.client._wait_between_calls()
        end_time = time.time()
        
        # Should have waited at least delay_seconds (0.1s in our test)
        assert end_time - start_time >= self.client.delay_seconds

    @patch('src.data_collection.nba_api_client.players.get_players')
    def test_get_all_players_success(self, mock_get_players):
        """Test successful retrieval of all players"""
        # Mock NBA API response
        mock_players = [
            {'id': 1, 'full_name': 'LeBron James'},
            {'id': 2, 'full_name': 'Stephen Curry'}
        ]
        mock_get_players.return_value = mock_players.copy()

        result = self.client.get_all_players()

        assert len(result) == 2
        assert result[0]['full_name'] == 'LeBron James'
        assert 'retrieved_at' in result[0]  # Should add timestamp
        mock_get_players.assert_called_once()

    @patch('src.data_collection.nba_api_client.teams.get_teams')
    def test_get_all_teams_success(self, mock_get_teams):
        """Test successful retrieval of all teams"""
        mock_teams = [
            {'id': 1, 'full_name': 'Los Angeles Lakers'},
            {'id': 2, 'full_name': 'Golden State Warriors'}
        ]
        mock_get_teams.return_value = mock_teams.copy()

        result = self.client.get_all_teams()

        assert len(result) == 2
        assert result[0]['full_name'] == 'Los Angeles Lakers'
        assert 'retrieved_at' in result[0]
        mock_get_teams.assert_called_once()

    @patch('src.data_collection.nba_api_client.leagueleaders.LeagueLeaders')
    def test_get_player_season_stats_success(self, mock_league_leaders):
        """Test successful retrieval of player season stats"""
        # Mock DataFrame response
        mock_df = pd.DataFrame({
            'PLAYER': ['LeBron James', 'Stephen Curry'],
            'PTS': [25.3, 29.5],
            'TEAM': ['LAL', 'GSW']
        })
        
        mock_stats = Mock()
        mock_stats.get_data_frames.return_value = [mock_df]
        mock_league_leaders.return_value = mock_stats

        result = self.client.get_player_season_stats("2023-24")

        assert len(result) == 2
        assert result[0]['PLAYER'] == 'LeBron James'
        assert result[0]['season'] == '2023-24'
        assert 'retrieved_at' in result[0]
        mock_league_leaders.assert_called_once_with(
            season="2023-24",
            season_type_all_star='Regular Season'
        )

    @patch('src.data_collection.nba_api_client.leagueleaders.LeagueLeaders')
    def test_get_player_season_stats_api_failure(self, mock_league_leaders):
        """Test handling of API failure in player stats"""
        # Mock API call that times out
        def slow_api_call(*args, **kwargs):
            time.sleep(3)  # Longer than timeout
            
        mock_league_leaders.side_effect = slow_api_call

        result = self.client.get_player_season_stats("2023-24")

        assert result == []  # Should return empty list on failure

    def test_integration_basic_client_operations(self):
        """Integration test for basic client operations without network calls"""
        # Test that client can be created and basic methods exist
        client = NBAApiClient()
        
        assert hasattr(client, 'get_all_players')
        assert hasattr(client, 'get_all_teams')
        assert hasattr(client, 'get_player_season_stats')
        assert hasattr(client, '_safe_api_call')
        assert hasattr(client, '_wait_between_calls')


class TestNBAApiClientEdgeCases:
    """Test edge cases and error conditions"""

    def test_very_short_timeout(self):
        """Test behavior with very short timeout"""
        client = NBAApiClient(timeout_seconds=0.001)  # 1 millisecond
        
        def normal_function():
            return "success"
        
        # Even fast operations might timeout with 1ms
        result = client._safe_api_call(normal_function)
        # Could be None (timeout) or "success" (completed in time)
        assert result is None or result == "success"

    def test_zero_delay(self):
        """Test behavior with zero delay"""
        client = NBAApiClient(delay_seconds=0.0)
        
        start_time = time.time()
        client._wait_between_calls()
        end_time = time.time()
        
        # Should complete almost instantly
        assert end_time - start_time < 0.1


if __name__ == "__main__":
    pytest.main([__file__])
