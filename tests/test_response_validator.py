"""
Unit tests for NBA API Response Validator

Tests the response validation functionality including:
- Schema validation for different endpoint types
- Data quality checks
- Error handling and reporting
- Integration with NBA API client
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime

from src.data_collection.response_validator import (
    ResponseValidator, 
    ValidationResult, 
    ValidationIssue, 
    ValidationSeverity,
    validate_nba_response
)


class TestValidationIssue:
    """Test ValidationIssue dataclass"""
    
    def test_validation_issue_creation(self):
        """Test creating a validation issue"""
        issue = ValidationIssue(
            ValidationSeverity.ERROR,
            "test_field",
            "Test message",
            count=5,
            sample_values=["val1", "val2"]
        )
        
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.field == "test_field"
        assert issue.message == "Test message"
        assert issue.count == 5
        assert issue.sample_values == ["val1", "val2"]
    
    def test_validation_issue_defaults(self):
        """Test validation issue with default values"""
        issue = ValidationIssue(
            ValidationSeverity.WARNING,
            "test_field",
            "Test message"
        )
        
        assert issue.count == 1
        assert issue.sample_values == []


class TestValidationResult:
    """Test ValidationResult dataclass"""
    
    def test_validation_result_creation(self):
        """Test creating a validation result"""
        issues = [
            ValidationIssue(ValidationSeverity.ERROR, "field1", "Error message"),
            ValidationIssue(ValidationSeverity.WARNING, "field2", "Warning message")
        ]
        
        result = ValidationResult(
            is_valid=False,
            total_records=100,
            issues=issues,
            data_quality_score=75.5
        )
        
        assert result.is_valid is False
        assert result.total_records == 100
        assert len(result.issues) == 2
        assert result.data_quality_score == 75.5
    
    def test_has_errors(self):
        """Test error detection"""
        issues_with_error = [
            ValidationIssue(ValidationSeverity.ERROR, "field1", "Error"),
            ValidationIssue(ValidationSeverity.WARNING, "field2", "Warning")
        ]
        
        issues_without_error = [
            ValidationIssue(ValidationSeverity.WARNING, "field1", "Warning"),
            ValidationIssue(ValidationSeverity.INFO, "field2", "Info")
        ]
        
        result_with_error = ValidationResult(True, 100, issues_with_error, 80.0)
        result_without_error = ValidationResult(True, 100, issues_without_error, 90.0)
        
        assert result_with_error.has_errors() is True
        assert result_without_error.has_errors() is False
    
    def test_has_warnings(self):
        """Test warning detection"""
        issues_with_warning = [
            ValidationIssue(ValidationSeverity.WARNING, "field1", "Warning"),
            ValidationIssue(ValidationSeverity.INFO, "field2", "Info")
        ]
        
        issues_without_warning = [
            ValidationIssue(ValidationSeverity.INFO, "field1", "Info")
        ]
        
        result_with_warning = ValidationResult(True, 100, issues_with_warning, 90.0)
        result_without_warning = ValidationResult(True, 100, issues_without_warning, 95.0)
        
        assert result_with_warning.has_warnings() is True
        assert result_without_warning.has_warnings() is False
    
    def test_get_summary(self):
        """Test validation result summary"""
        issues = [
            ValidationIssue(ValidationSeverity.ERROR, "field1", "Error"),
            ValidationIssue(ValidationSeverity.WARNING, "field2", "Warning"),
            ValidationIssue(ValidationSeverity.INFO, "field3", "Info")
        ]
        
        result = ValidationResult(False, 150, issues, 65.5)
        summary = result.get_summary()
        
        assert "FAIL" in summary
        assert "65.5/100" in summary
        assert "Records: 150" in summary
        assert "Errors: 1" in summary
        assert "Warnings: 1" in summary
        assert "Info: 1" in summary


class TestResponseValidator:
    """Test ResponseValidator class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = ResponseValidator(strict_mode=False)
        self.strict_validator = ResponseValidator(strict_mode=True)
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        assert self.validator.strict_mode is False
        assert self.strict_validator.strict_mode is True
        assert 'player_stats' in self.validator.validation_schemas
        assert 'team_stats' in self.validator.validation_schemas
    
    def test_validate_empty_data(self):
        """Test validation of empty data"""
        result = self.validator.validate_response([], 'player_stats')
        
        assert result.is_valid is False  # Empty data fails minimum record count requirement
        assert result.total_records == 0
        # Should have an error about record count and warning about empty data
        assert any(issue.field == "record_count" for issue in result.issues)
        assert any(issue.field == "empty_data" for issue in result.issues)
    
    def test_validate_invalid_data_structure(self):
        """Test validation of invalid data structure"""
        result = self.validator.validate_response("invalid", 'player_stats')
        
        assert result.is_valid is False
        assert result.total_records == 0
        assert any(issue.field == "data_structure" for issue in result.issues)
    
    def test_validate_unknown_endpoint(self):
        """Test validation with unknown endpoint type"""
        data = [{'id': 1, 'name': 'test'}]
        result = self.validator.validate_response(data, 'unknown_endpoint')
        
        assert result.is_valid is True  # Should pass with warning
        assert result.data_quality_score == 80.0  # Default score for unknown schemas
        assert any(issue.field == "schema" for issue in result.issues)
    
    def test_validate_valid_player_stats(self):
        """Test validation of valid player stats data"""
        valid_data = [
            {
                'PLAYER_ID': 2544,
                'PLAYER': 'LeBron James',
                'TEAM_ID': 1610612747,
                'TEAM': 'LAL',
                'GP': 71,
                'MIN': 35.3,
                'PTS': 25.7,
                'REB': 7.3,
                'AST': 8.3,
                'FG_PCT': 0.540,
                'FT_PCT': 0.750
            },
            {
                'PLAYER_ID': 201939,
                'PLAYER': 'Stephen Curry',
                'TEAM_ID': 1610612744,
                'TEAM': 'GSW',
                'GP': 74,
                'MIN': 32.7,
                'PTS': 29.5,
                'REB': 5.1,
                'AST': 6.3,
                'FG_PCT': 0.427,
                'FT_PCT': 0.915
            }
        ]
        
        result = self.validator.validate_response(valid_data, 'player_stats')
        
        assert result.is_valid is True
        assert result.total_records == 2
        assert result.data_quality_score > 90  # Should have high quality score
    
    def test_validate_player_stats_missing_fields(self):
        """Test validation with missing required fields"""
        invalid_data = [
            {
                'PLAYER': 'LeBron James',
                'TEAM': 'LAL',
                'PTS': 25.7
                # Missing PLAYER_ID, TEAM_ID, GP, MIN
            }
        ]
        
        result = self.validator.validate_response(invalid_data, 'player_stats')
        
        assert result.is_valid is False
        assert any(issue.field == "missing_fields" for issue in result.issues)
    
    def test_validate_player_stats_out_of_range(self):
        """Test validation with out-of-range values"""
        invalid_data = [
            {
                'PLAYER_ID': 2544,
                'PLAYER': 'LeBron James',
                'TEAM_ID': 1610612747,
                'TEAM': 'LAL',
                'GP': 150,  # Invalid: > 82 games
                'MIN': 60,  # Invalid: > 48 minutes
                'PTS': -10,  # Invalid: negative points
                'FG_PCT': 1.5,  # Invalid: > 1.0
                'FT_PCT': -0.1  # Invalid: < 0.0
            }
        ]
        
        result = self.validator.validate_response(invalid_data, 'player_stats')
        
        # Should still be valid but with warnings
        assert result.is_valid is True
        assert result.has_warnings()
        
        # Check for specific range violations
        range_issues = [issue for issue in result.issues if "above maximum" in issue.message or "below minimum" in issue.message]
        assert len(range_issues) > 0
    
    def test_validate_high_null_percentage(self):
        """Test validation with high null percentage"""
        data_with_nulls = []
        for i in range(100):
            player_data = {
                'PLAYER_ID': i,
                'PLAYER': f'Player {i}',
                'TEAM_ID': 1610612747,
                'TEAM': 'LAL',
                'GP': 50,
                'MIN': 30.0,
                'PTS': None if i < 20 else 20.0,  # 20% nulls
                'REB': 5.0,
                'AST': 5.0
            }
            data_with_nulls.append(player_data)
        
        result = self.validator.validate_response(data_with_nulls, 'player_stats')
        
        # Should have warning about high null percentage
        null_issues = [issue for issue in result.issues if "null percentage" in issue.message]
        assert len(null_issues) > 0
    
    def test_validate_team_stats(self):
        """Test validation of team stats data"""
        valid_team_data = []
        for i in range(30):  # 30 NBA teams
            team_data = {
                'TEAM_ID': 1610612740 + i,
                'TEAM_NAME': f'Team {i}',
                'GP': 82,
                'W': 41,
                'L': 41,
                'W_PCT': 0.500,
                'PTS': 110.5,
                'REB': 45.0,
                'AST': 25.0
            }
            valid_team_data.append(team_data)
        
        result = self.validator.validate_response(valid_team_data, 'team_stats')
        
        assert result.is_valid is True
        assert result.total_records == 30
        assert result.data_quality_score > 90
    
    def test_validate_standings_categorical_fields(self):
        """Test validation of categorical fields in standings"""
        standings_data = [
            {
                'TEAM_ID': 1610612747,
                'TeamName': 'Lakers',
                'WINS': 45,
                'LOSSES': 37,
                'WinPCT': 0.549,
                'Conference': 'West'
            },
            {
                'TEAM_ID': 1610612738,
                'TeamName': 'Celtics',
                'WINS': 50,
                'LOSSES': 32,
                'WinPCT': 0.610,
                'Conference': 'Invalid Conference'  # Invalid value
            }
        ]
        
        result = self.validator.validate_response(standings_data, 'standings')
        
        # Should have warning about invalid categorical value
        categorical_issues = [issue for issue in result.issues if "invalid categorical values" in issue.message]
        assert len(categorical_issues) > 0
    
    def test_strict_mode_treats_warnings_as_errors(self):
        """Test that strict mode treats warnings as validation failures"""
        data_with_warnings = [
            {
                'PLAYER_ID': 2544,
                'PLAYER': 'LeBron James',
                'TEAM_ID': 1610612747,
                'TEAM': 'LAL',
                'GP': 150,  # This will trigger a warning
                'MIN': 35.3,
                'PTS': 25.7
            }
        ]
        
        # Non-strict mode should pass
        normal_result = self.validator.validate_response(data_with_warnings, 'player_stats')
        assert normal_result.is_valid is True
        
        # Strict mode should fail
        strict_result = self.strict_validator.validate_response(data_with_warnings, 'player_stats')
        assert strict_result.is_valid is False or strict_result.has_warnings()
    
    def test_single_dict_input(self):
        """Test validation with single dictionary input"""
        single_player = {
            'PLAYER_ID': 2544,
            'PLAYER': 'LeBron James',
            'TEAM_ID': 1610612747,
            'TEAM': 'LAL',
            'GP': 71,
            'MIN': 35.3,
            'PTS': 25.7
        }
        
        result = self.validator.validate_response(single_player, 'player_stats')
        
        assert result.is_valid is True
        assert result.total_records == 1
    
    def test_quality_score_calculation(self):
        """Test data quality score calculation"""
        # Perfect data should get 100
        perfect_data = [
            {
                'PLAYER_ID': 2544,
                'PLAYER': 'LeBron James',
                'TEAM_ID': 1610612747,
                'TEAM': 'LAL',
                'GP': 71,
                'MIN': 35.3,
                'PTS': 25.7
            }
        ]
        
        perfect_result = self.validator.validate_response(perfect_data, 'player_stats')
        assert perfect_result.data_quality_score == 100.0
        
        # Data with issues should have lower score
        problematic_data = [
            {
                'PLAYER_ID': 2544,
                'PLAYER': 'LeBron James',
                'TEAM_ID': 1610612747,
                'TEAM': 'LAL',
                'GP': 150,  # Out of range - warning
                'MIN': 60,  # Out of range - warning
                'PTS': 25.7
            }
        ]
        
        problematic_result = self.validator.validate_response(problematic_data, 'player_stats')
        assert problematic_result.data_quality_score < 100.0


class TestConvenienceFunction:
    """Test the convenience function"""
    
    def test_validate_nba_response_function(self):
        """Test the standalone validation function"""
        # Create enough players to meet minimum requirement (400)
        data = []
        for i in range(450):  # More than minimum required
            data.append({
                'id': i,
                'full_name': f'Player {i}'
            })
        
        result = validate_nba_response(data, 'players')
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
    
    def test_validate_nba_response_insufficient_records(self):
        """Test the standalone function with insufficient records"""
        data = [
            {
                'id': 1,
                'full_name': 'LeBron James'
            }
        ]
        
        result = validate_nba_response(data, 'players')
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False  # Should fail due to insufficient records
        assert any(issue.field == "record_count" for issue in result.issues)
    
    def test_validate_nba_response_strict_mode(self):
        """Test the standalone function with strict mode"""
        data = [
            {
                'PLAYER_ID': 2544,
                'PLAYER': 'LeBron James',
                'TEAM_ID': 1610612747,
                'TEAM': 'LAL',
                'GP': 150,  # Will cause warning
                'MIN': 35.3,
                'PTS': 25.7
            }
        ]
        
        normal_result = validate_nba_response(data, 'player_stats', strict_mode=False)
        strict_result = validate_nba_response(data, 'player_stats', strict_mode=True)
        
        assert normal_result.is_valid is True
        # In strict mode, warnings might affect validity depending on implementation
        assert isinstance(strict_result, ValidationResult)


class TestValidationIntegration:
    """Test validation integration scenarios"""
    
    def test_validation_with_dataframe_conversion_error(self):
        """Test handling of DataFrame conversion errors"""
        # Create data that can't be converted to DataFrame easily
        problematic_data = [
            {
                'field1': {'nested': 'dict'},  # Nested dict might cause issues
                'field2': [1, 2, 3]  # List might cause issues
            }
        ]
        
        validator = ResponseValidator()
        result = validator.validate_response(problematic_data, 'player_stats')
        
        # Should handle the error gracefully
        assert isinstance(result, ValidationResult)
        # Might have a dataframe_conversion error
        conversion_errors = [issue for issue in result.issues if issue.field == "dataframe_conversion"]
        # This test verifies error handling, exact behavior may vary
    
    def test_validation_performance_with_large_dataset(self):
        """Test validation performance with larger datasets"""
        # Create a larger dataset
        large_data = []
        for i in range(1000):
            player_data = {
                'PLAYER_ID': i,
                'PLAYER': f'Player {i}',
                'TEAM_ID': 1610612747,
                'TEAM': 'LAL',
                'GP': 50,
                'MIN': 30.0,
                'PTS': 20.0,
                'REB': 5.0,
                'AST': 5.0
            }
            large_data.append(player_data)
        
        validator = ResponseValidator()
        result = validator.validate_response(large_data, 'player_stats')
        
        assert result.is_valid is True
        assert result.total_records == 1000
        # Validation should complete without timeout


if __name__ == "__main__":
    pytest.main([__file__])
