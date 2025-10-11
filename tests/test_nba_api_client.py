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
import requests

from src.data_collection.nba_api_client import NBAApiClient
from src.data_collection.response_validator import ValidationResult, ValidationIssue, ValidationSeverity


class TestNBAApiClient:
    """Test suite for NBAApiClient"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.client = NBAApiClient(delay_seconds=0.1, timeout_seconds=2.0, max_retries=2, 
                                  enable_validation=True, strict_validation=False)
        self.client_no_validation = NBAApiClient(delay_seconds=0.1, timeout_seconds=2.0, max_retries=2,
                                                enable_validation=False)

    def test_init_default_values(self):
        """Test client initialization with default values"""
        client = NBAApiClient()
        assert client.delay_seconds == 1.0
        assert client.timeout_seconds == 30.0
        assert client.max_retries == 3
        assert client.enable_validation is True
        assert client.strict_validation is False

    def test_init_custom_values(self):
        """Test client initialization with custom values"""
        client = NBAApiClient(delay_seconds=0.5, timeout_seconds=10.0, max_retries=1, 
                             enable_validation=False, strict_validation=True)
        assert client.delay_seconds == 0.5
        assert client.timeout_seconds == 10.0
        assert client.max_retries == 1
        assert client.enable_validation is False
        assert client.strict_validation is True

    def test_timeout_handling_slow_function(self):
        """Test that timeout handling works for slow operations"""
        def slow_function():
            print(f"Slow function starting, will sleep 5 seconds...")
            time.sleep(5)  # Much longer than our 2-second timeout
            print(f"Slow function completed (should not see this)")
            return "Should not reach this"

        print(f"Client timeout setting: {self.client.timeout_seconds}s")
        print(f"Client max retries: {self.client.max_retries}")
        start_time = time.time()
        result = self.client._safe_api_call(slow_function)
        end_time = time.time()
        
        # Verify it actually timed out and returned None
        duration = end_time - start_time
        print(f"Actual duration: {duration:.2f}s")
        print(f"Result: {result}")
        
        assert result is None, f"Expected None but got {result}"
        
        # With retry logic: 3 attempts * 2s timeout + 1s + 2s backoff = ~9s total
        # Allow some variance for system timing
        expected_min = (self.client.max_retries + 1) * self.client.timeout_seconds + (1 + 2) - 1  # ~8s
        expected_max = (self.client.max_retries + 1) * self.client.timeout_seconds + (1 + 2) + 2  # ~11s
        
        assert expected_min <= duration <= expected_max, (
            f"Expected ~{expected_min}-{expected_max}s with retries "
            f"({self.client.max_retries + 1} attempts * {self.client.timeout_seconds}s + backoff), "
            f"but took {duration:.2f}s"
        )

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

    def test_retry_logic_eventual_success(self):
        """Test that retry logic eventually succeeds after failures"""
        attempt_count = [0]  # Mutable counter
        
        def flaky_function():
            attempt_count[0] += 1
            if attempt_count[0] < 3:  # Fail first 2 attempts
                raise requests.exceptions.ConnectionError("Network error")
            return "Success on attempt 3"
        
        start_time = time.time()
        result = self.client._safe_api_call(flaky_function)
        end_time = time.time()
        
        assert result == "Success on attempt 3"
        assert attempt_count[0] == 3  # Should have tried 3 times
        # Should take roughly: 0 + 1 + 2 = 3 seconds for backoff delays
        assert 2.5 <= (end_time - start_time) <= 4.0

    def test_retry_logic_all_attempts_fail(self):
        """Test that retry logic gives up after max attempts"""
        attempt_count = [0]
        
        def always_failing_function():
            attempt_count[0] += 1
            raise requests.exceptions.ConnectionError("Always fails")
        
        result = self.client._safe_api_call(always_failing_function)
        
        assert result is None
        assert attempt_count[0] == self.client.max_retries + 1  # 3 total attempts (0, 1, 2)

    def test_retry_logic_no_retry_on_timeout(self):
        """Test that timeouts are retried like other failures"""
        def timeout_function():
            time.sleep(3)  # Longer than timeout
            return "Should not reach"
        
        start_time = time.time()
        result = self.client._safe_api_call(timeout_function)
        end_time = time.time()
        
        assert result is None
        # Should retry: first timeout (~2s) + 1s delay + second timeout (~2s) + 2s delay + third timeout (~2s)
        # Total: roughly 2 + 1 + 2 + 2 + 2 = 9 seconds
        assert 8.0 <= (end_time - start_time) <= 12.0


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


class TestNBAApiClientValidation:
    """Test response validation integration with NBA API client"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.client = NBAApiClient(delay_seconds=0.1, timeout_seconds=2.0, max_retries=1,
                                  enable_validation=True, strict_validation=False)
        self.strict_client = NBAApiClient(delay_seconds=0.1, timeout_seconds=2.0, max_retries=1,
                                         enable_validation=True, strict_validation=True)
        self.no_validation_client = NBAApiClient(delay_seconds=0.1, timeout_seconds=2.0, max_retries=1,
                                                enable_validation=False)
    
    def test_validate_response_data_disabled(self):
        """Test validation when disabled"""
        data = [{'invalid': 'data'}]
        result = self.no_validation_client._validate_response_data(data, 'player_stats')
        assert result is True  # Should always return True when disabled
    
    def test_validate_response_data_empty(self):
        """Test validation with empty data"""
        result = self.client._validate_response_data([], 'player_stats')
        assert result is True  # Empty data is valid
    
    @patch('src.data_collection.nba_api_client.validate_nba_response')
    def test_validate_response_data_success(self, mock_validate):
        """Test successful validation"""
        # Mock successful validation
        mock_result = ValidationResult(
            is_valid=True,
            total_records=2,
            issues=[],
            data_quality_score=95.0
        )
        mock_validate.return_value = mock_result
        
        data = [
            {'PLAYER_ID': 1, 'PLAYER': 'Test Player', 'TEAM': 'TEST', 'GP': 50, 'MIN': 30, 'PTS': 20}
        ]
        
        result = self.client._validate_response_data(data, 'player_stats')
        
        assert result is True
        mock_validate.assert_called_once_with(data, 'player_stats', False)
    
    @patch('src.data_collection.nba_api_client.validate_nba_response')
    def test_validate_response_data_failure(self, mock_validate):
        """Test validation failure"""
        # Mock failed validation
        mock_result = ValidationResult(
            is_valid=False,
            total_records=1,
            issues=[ValidationIssue(ValidationSeverity.ERROR, 'test_field', 'Test error')],
            data_quality_score=60.0
        )
        mock_validate.return_value = mock_result
        
        data = [{'invalid': 'data'}]
        
        result = self.client._validate_response_data(data, 'player_stats')
        
        assert result is False
        mock_validate.assert_called_once_with(data, 'player_stats', False)
    
    @patch('src.data_collection.nba_api_client.validate_nba_response')
    def test_validate_response_data_validation_exception(self, mock_validate):
        """Test handling of validation exceptions"""
        # Mock validation raising an exception
        mock_validate.side_effect = Exception("Validation error")
        
        data = [{'test': 'data'}]
        
        result = self.client._validate_response_data(data, 'player_stats')
        
        # Should return True (fail-open) when validation itself fails
        assert result is True
    
    @patch('src.data_collection.nba_api_client.players.get_players')
    @patch('src.data_collection.nba_api_client.validate_nba_response')
    def test_get_all_players_with_validation(self, mock_validate, mock_get_players):
        """Test get_all_players with validation enabled"""
        # Mock NBA API response
        mock_players = [
            {'id': 1, 'full_name': 'LeBron James'},
            {'id': 2, 'full_name': 'Stephen Curry'}
        ]
        mock_get_players.return_value = mock_players.copy()
        
        # Mock successful validation
        mock_result = ValidationResult(
            is_valid=True,
            total_records=2,
            issues=[],
            data_quality_score=100.0
        )
        mock_validate.return_value = mock_result
        
        result = self.client.get_all_players()
        
        assert len(result) == 2
        assert 'retrieved_at' in result[0]
        mock_validate.assert_called_once()
        # Check that validation was called with the right endpoint type
        args, kwargs = mock_validate.call_args
        assert args[1] == 'players'  # endpoint_type
    
    @patch('src.data_collection.nba_api_client.players.get_players')
    @patch('src.data_collection.nba_api_client.validate_nba_response')
    def test_get_all_players_validation_failure_strict(self, mock_validate, mock_get_players):
        """Test get_all_players with validation failure in strict mode"""
        # Mock NBA API response
        mock_players = [{'id': 1, 'full_name': 'Test Player'}]
        mock_get_players.return_value = mock_players.copy()
        
        # Mock failed validation
        mock_result = ValidationResult(
            is_valid=False,
            total_records=1,
            issues=[ValidationIssue(ValidationSeverity.ERROR, 'test_field', 'Test error')],
            data_quality_score=40.0
        )
        mock_validate.return_value = mock_result
        
        result = self.strict_client.get_all_players()
        
        # In strict mode with validation failure, should return empty list
        assert result == []
    
    @patch('src.data_collection.nba_api_client.leagueleaders.LeagueLeaders')
    @patch('src.data_collection.nba_api_client.validate_nba_response')
    def test_get_player_season_stats_with_validation(self, mock_validate, mock_league_leaders):
        """Test get_player_season_stats with validation"""
        # Mock DataFrame response
        mock_df = pd.DataFrame({
            'PLAYER_ID': [2544, 201939],
            'PLAYER': ['LeBron James', 'Stephen Curry'],
            'TEAM_ID': [1610612747, 1610612744],
            'TEAM': ['LAL', 'GSW'],
            'GP': [71, 74],
            'MIN': [35.3, 32.7],
            'PTS': [25.7, 29.5]
        })
        
        mock_stats = Mock()
        mock_stats.get_data_frames.return_value = [mock_df]
        mock_league_leaders.return_value = mock_stats
        
        # Mock successful validation
        mock_result = ValidationResult(
            is_valid=True,
            total_records=2,
            issues=[],
            data_quality_score=95.0
        )
        mock_validate.return_value = mock_result
        
        result = self.client.get_player_season_stats("2023-24")
        
        assert len(result) == 2
        assert result[0]['season'] == '2023-24'
        assert 'retrieved_at' in result[0]
        mock_validate.assert_called_once()
        # Check that validation was called with the right endpoint type
        args, kwargs = mock_validate.call_args
        assert args[1] == 'player_stats'
    
    @patch('src.data_collection.nba_api_client.teams.get_teams')
    def test_get_all_teams_no_validation(self, mock_get_teams):
        """Test get_all_teams with validation disabled"""
        mock_teams = [{'id': 1, 'full_name': 'Test Team'}]
        mock_get_teams.return_value = mock_teams.copy()
        
        result = self.no_validation_client.get_all_teams()
        
        assert len(result) == 1
        assert 'retrieved_at' in result[0]
        # No validation should have been called
    
    def test_integration_validation_enabled_by_default(self):
        """Test that validation is enabled by default"""
        client = NBAApiClient()
        assert client.enable_validation is True
        assert client.strict_validation is False
    
    @patch('src.data_collection.nba_api_client.validate_nba_response')
    def test_validation_with_quality_score_warning(self, mock_validate):
        """Test validation with low quality score"""
        # Mock validation with low quality score
        mock_result = ValidationResult(
            is_valid=True,
            total_records=10,
            issues=[ValidationIssue(ValidationSeverity.WARNING, 'test_field', 'Quality issue')],
            data_quality_score=85.0  # Below 90%
        )
        mock_validate.return_value = mock_result
        
        data = [{'test': 'data'}]
        
        # Should still return True but log warning about quality score
        result = self.client._validate_response_data(data, 'player_stats')
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__])
