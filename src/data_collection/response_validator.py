"""
NBA API Response Validator

Validates NBA API responses to ensure data quality and completeness before processing.
Provides structured validation with detailed error reporting and data quality metrics.

Key features:
- Schema-based validation for different endpoint types
- Data quality checks (nulls, ranges, formats)
- Flexible validation rules with severity levels
- Detailed validation reports for debugging
- Production-ready error handling
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import pandas as pd
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    ERROR = "error"      # Critical issues that prevent data usage
    WARNING = "warning"  # Issues that may affect data quality
    INFO = "info"       # Informational notices about data


@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    severity: ValidationSeverity
    field: str
    message: str
    count: int = 1
    sample_values: List[Any] = None
    
    def __post_init__(self):
        if self.sample_values is None:
            self.sample_values = []


@dataclass
class ValidationResult:
    """Results of response validation"""
    is_valid: bool
    total_records: int
    issues: List[ValidationIssue]
    data_quality_score: float  # 0-100 score based on issues
    
    def has_errors(self) -> bool:
        """Check if there are any error-level issues"""
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)
    
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues"""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues) 
    
    def get_summary(self) -> str:
        """Get a human-readable summary of validation results"""
        error_count = sum(1 for issue in self.issues if issue.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for issue in self.issues if issue.severity == ValidationSeverity.WARNING)
        info_count = sum(1 for issue in self.issues if issue.severity == ValidationSeverity.INFO)
        
        return (f"Validation Result: {'PASS' if self.is_valid else 'FAIL'} "
                f"(Score: {self.data_quality_score:.1f}/100, "
                f"Records: {self.total_records}, "
                f"Errors: {error_count}, Warnings: {warning_count}, Info: {info_count})")


class ResponseValidator:
    """Validates NBA API responses for data quality and completeness"""
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the response validator
        
        Args:
            strict_mode: If True, warnings are treated as errors
        """
        self.strict_mode = strict_mode
        self.validation_schemas = self._load_validation_schemas()
    
    def _load_validation_schemas(self) -> Dict[str, Dict]:
        """Load validation schemas for different NBA API endpoints"""
        return {
            'player_stats': {
                'required_fields': ['PLAYER_ID', 'PLAYER', 'TEAM_ID', 'TEAM', 'GP', 'MIN', 'PTS'],
                'numeric_fields': ['PLAYER_ID', 'TEAM_ID', 'GP', 'MIN', 'PTS', 'REB', 'AST', 'FG_PCT', 'FT_PCT'],
                'string_fields': ['PLAYER', 'TEAM'],
                'field_ranges': {
                    'GP': (0, 82),      # Games played: 0-82 for regular season
                    'MIN': (0, 48),     # Minutes: 0-48 per game
                    'PTS': (0, 100),    # Points: reasonable upper bound
                    'FG_PCT': (0, 1),   # Field goal percentage: 0-1
                    'FT_PCT': (0, 1),   # Free throw percentage: 0-1
                },
                'min_records': 1,
                'max_null_percentage': 0.05,  # General threshold for most fields
                'critical_fields': ['PLAYER_ID', 'PLAYER', 'TEAM_ID', 'TEAM', 'GP'],  # Must never be null
                'nullable_fields': [
                    'FG_PCT', 'FT_PCT', 'FG3_PCT',  # Percentages when 0 attempts
                    'REB', 'AST', 'STL', 'BLK', 'TOV',  # Can be 0/null for limited minutes
                    'PLUS_MINUS', 'OREB', 'DREB'  # Advanced stats that might be null
                ]
            },
            'team_stats': {
                'required_fields': ['TEAM_ID', 'TEAM_NAME', 'GP', 'W', 'L', 'W_PCT', 'PTS'],
                'numeric_fields': ['TEAM_ID', 'GP', 'W', 'L', 'W_PCT', 'PTS', 'REB', 'AST'],
                'string_fields': ['TEAM_NAME'],
                'field_ranges': {
                    'GP': (0, 82),
                    'W': (0, 82),
                    'L': (0, 82),
                    'W_PCT': (0, 1),
                    'PTS': (50, 300),   # Team points per game: reasonable range
                },
                'min_records': 30,  # Should have all 30 NBA teams
                'max_null_percentage': 0.02,
                'critical_fields': ['TEAM_ID', 'TEAM_NAME', 'GP', 'W', 'L'],  # Core team data
                'nullable_fields': ['PLUS_MINUS', 'NET_RATING']  # Advanced team metrics
            },
            'standings': {
                'required_fields': ['TEAM_ID', 'TeamName', 'WINS', 'LOSSES', 'WinPCT', 'Conference'],
                'numeric_fields': ['TEAM_ID', 'WINS', 'LOSSES', 'WinPCT'],
                'string_fields': ['TeamName', 'Conference'],
                'field_ranges': {
                    'WINS': (0, 82),
                    'LOSSES': (0, 82),
                    'WinPCT': (0, 1),
                },
                'categorical_fields': {
                    'Conference': ['East', 'West']
                },
                'min_records': 30,
                'max_null_percentage': 0.01,
                'critical_fields': ['TEAM_ID', 'TeamName', 'WINS', 'LOSSES', 'Conference'],
                'nullable_fields': []  # Standings data should be complete
            },
            'games': {
                'required_fields': ['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'GAME_STATUS_TEXT'],
                'numeric_fields': ['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID'],
                'string_fields': ['GAME_STATUS_TEXT'],
                'min_records': 0,  # Could be 0 games on some dates (off-season, breaks)
                'max_null_percentage': 0.0  # Game identifiers should never be null in existing records
            },
            'players': {
                'required_fields': ['id', 'full_name'],
                'numeric_fields': ['id'],
                'string_fields': ['full_name'],
                'min_records': 400,  # Should have hundreds of players
                'max_null_percentage': 0.01
            },
            'teams': {
                'required_fields': ['id', 'full_name', 'abbreviation'],
                'numeric_fields': ['id'],
                'string_fields': ['full_name', 'abbreviation'],
                'min_records': 30,  # Exactly 30 NBA teams
                'max_null_percentage': 0.0  # No nulls allowed for basic team data
            }
        }
    
    def validate_response(self, data: Union[List[Dict], Dict], endpoint_type: str) -> ValidationResult:
        """
        Validate NBA API response data
        
        Args:
            data: Response data (list of dicts or single dict)
            endpoint_type: Type of endpoint (player_stats, team_stats, etc.)
            
        Returns:
            ValidationResult with detailed validation information
        """
        issues = []
        
        # Convert single dict to list for uniform processing
        if isinstance(data, dict):
            data = [data]
        
        if not isinstance(data, list):
            issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                "data_structure",
                f"Expected list or dict, got {type(data).__name__}"
            ))
            return ValidationResult(False, 0, issues, 0.0)
        
        total_records = len(data)
        
        # Get validation schema
        schema = self.validation_schemas.get(endpoint_type)
        if not schema:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                "schema",
                f"No validation schema found for endpoint type: {endpoint_type}"
            ))
            return ValidationResult(True, total_records, issues, 80.0)
        
        # Basic structure validation
        issues.extend(self._validate_basic_structure(data, schema))
        
        if data:  # Only validate content if we have data
            # Convert to DataFrame for easier analysis
            try:
                df = pd.DataFrame(data)
                issues.extend(self._validate_data_quality(df, schema))
                issues.extend(self._validate_field_types(df, schema))
                issues.extend(self._validate_field_ranges(df, schema))
                issues.extend(self._validate_categorical_fields(df, schema))
            except Exception as e:
                issues.append(ValidationIssue(
                    ValidationSeverity.ERROR,
                    "dataframe_conversion",
                    f"Failed to convert data to DataFrame: {str(e)}"
                ))
        
        # Calculate data quality score
        quality_score = self._calculate_quality_score(issues, total_records)
        
        # Determine if validation passes
        is_valid = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        if self.strict_mode:
            is_valid = is_valid and not any(issue.severity == ValidationSeverity.WARNING for issue in issues)
        
        return ValidationResult(is_valid, total_records, issues, quality_score)
    
    def _validate_basic_structure(self, data: List[Dict], schema: Dict) -> List[ValidationIssue]:
        """Validate basic structure requirements"""
        issues = []
        
        # Check minimum record count
        min_records = schema.get('min_records', 0)
        if len(data) < min_records:
            issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                "record_count",
                f"Expected at least {min_records} records, got {len(data)}"
            ))
        
        # Check for empty data
        if not data:
            issues.append(ValidationIssue(
                ValidationSeverity.WARNING,
                "empty_data",
                "No data records found"
            ))
            return issues
        
        # Check required fields
        required_fields = schema.get('required_fields', [])
        first_record = data[0]
        missing_fields = [field for field in required_fields if field not in first_record]
        
        if missing_fields:
            issues.append(ValidationIssue(
                ValidationSeverity.ERROR,
                "missing_fields",
                f"Missing required fields: {missing_fields}",
                sample_values=missing_fields
            ))
        
        return issues
    
    def _validate_data_quality(self, df: pd.DataFrame, schema: Dict) -> List[ValidationIssue]:
        """Validate data quality metrics"""
        issues = []
        max_null_pct = schema.get('max_null_percentage', 0.1)
        critical_fields = schema.get('critical_fields', [])
        nullable_fields = schema.get('nullable_fields', [])
        
        # Check null percentages for each column with field-specific rules
        for column in df.columns:
            null_count = df[column].isnull().sum()
            null_percentage = null_count / len(df)
            
            # Determine appropriate threshold based on field type
            if column in critical_fields:
                # Critical fields (IDs, names) should have very low null rates
                field_threshold = 0.01  # 1% max for critical fields
                severity = ValidationSeverity.ERROR if null_percentage > field_threshold else ValidationSeverity.WARNING
            elif column in nullable_fields:
                # Nullable fields (percentages, advanced stats) can have higher null rates
                field_threshold = 0.5  # 50% max for nullable fields (e.g., FG_PCT when no attempts)
                severity = ValidationSeverity.WARNING if null_percentage > field_threshold else ValidationSeverity.INFO
            else:
                # Default fields use the schema's general threshold
                field_threshold = max_null_pct
                severity = ValidationSeverity.ERROR if null_percentage > 0.2 else ValidationSeverity.WARNING
            
            if null_percentage > field_threshold:
                # Provide context-aware messaging
                if column in nullable_fields and null_percentage <= 0.5:
                    message = f"Moderate null percentage in nullable field: {null_percentage:.1%} (common for {column})"
                    severity = ValidationSeverity.INFO  # Downgrade to info for expected nulls
                else:
                    message = f"High null percentage: {null_percentage:.1%} (limit: {field_threshold:.1%})"
                
                issues.append(ValidationIssue(
                    severity,
                    column,
                    message,
                    count=null_count
                ))
        
        # Check for duplicate records (if applicable)
        if 'PLAYER_ID' in df.columns or 'TEAM_ID' in df.columns:
            id_column = 'PLAYER_ID' if 'PLAYER_ID' in df.columns else 'TEAM_ID'
            duplicate_count = df[id_column].duplicated().sum()
            
            if duplicate_count > 0:
                issues.append(ValidationIssue(
                    ValidationSeverity.WARNING,
                    id_column,
                    f"Found {duplicate_count} duplicate records",
                    count=duplicate_count
                ))
        
        return issues
    
    def _validate_field_types(self, df: pd.DataFrame, schema: Dict) -> List[ValidationIssue]:
        """Validate field data types"""
        issues = []
        
        # Check numeric fields
        numeric_fields = schema.get('numeric_fields', [])
        for field in numeric_fields:
            if field in df.columns:
                non_numeric = pd.to_numeric(df[field], errors='coerce').isnull().sum()
                if non_numeric > 0:
                    issues.append(ValidationIssue(
                        ValidationSeverity.WARNING,
                        field,
                        f"Found {non_numeric} non-numeric values in numeric field",
                        count=non_numeric,
                        sample_values=df[field][pd.to_numeric(df[field], errors='coerce').isnull()].head(3).tolist()
                    ))
        
        # Check string fields
        string_fields = schema.get('string_fields', [])
        for field in string_fields:
            if field in df.columns:
                non_string_count = df[field].apply(lambda x: not isinstance(x, str) and pd.notna(x)).sum()
                if non_string_count > 0:
                    issues.append(ValidationIssue(
                        ValidationSeverity.INFO,
                        field,
                        f"Found {non_string_count} non-string values in string field",
                        count=non_string_count
                    ))
        
        return issues
    
    def _validate_field_ranges(self, df: pd.DataFrame, schema: Dict) -> List[ValidationIssue]:
        """Validate numeric field ranges"""
        issues = []
        field_ranges = schema.get('field_ranges', {})
        
        for field, (min_val, max_val) in field_ranges.items():
            if field in df.columns:
                # Convert to numeric for range checking
                numeric_series = pd.to_numeric(df[field], errors='coerce')
                
                # Check minimum values
                below_min = (numeric_series < min_val).sum()
                if below_min > 0:
                    issues.append(ValidationIssue(
                        ValidationSeverity.WARNING,
                        field,
                        f"Found {below_min} values below minimum ({min_val})",
                        count=below_min,
                        sample_values=df[field][numeric_series < min_val].head(3).tolist()
                    ))
                
                # Check maximum values
                above_max = (numeric_series > max_val).sum()
                if above_max > 0:
                    issues.append(ValidationIssue(
                        ValidationSeverity.WARNING,
                        field,
                        f"Found {above_max} values above maximum ({max_val})",
                        count=above_max,
                        sample_values=df[field][numeric_series > max_val].head(3).tolist()
                    ))
        
        return issues
    
    def _validate_categorical_fields(self, df: pd.DataFrame, schema: Dict) -> List[ValidationIssue]:
        """Validate categorical field values"""
        issues = []
        categorical_fields = schema.get('categorical_fields', {})
        
        for field, allowed_values in categorical_fields.items():
            if field in df.columns:
                invalid_values = df[~df[field].isin(allowed_values)][field].unique()
                if len(invalid_values) > 0:
                    issues.append(ValidationIssue(
                        ValidationSeverity.WARNING,
                        field,
                        f"Found invalid categorical values: {list(invalid_values)}",
                        count=len(invalid_values),
                        sample_values=list(invalid_values)
                    ))
        
        return issues
    
    def _calculate_quality_score(self, issues: List[ValidationIssue], total_records: int) -> float:
        """Calculate data quality score (0-100)"""
        if not issues:
            return 100.0
        
        # Weight different severity levels
        error_weight = 20
        warning_weight = 5
        info_weight = 1
        
        total_penalty = 0
        for issue in issues:
            if issue.severity == ValidationSeverity.ERROR:
                total_penalty += error_weight
            elif issue.severity == ValidationSeverity.WARNING:
                total_penalty += warning_weight
            else:
                total_penalty += info_weight
        
        # Calculate score (minimum 0)
        base_score = 100
        score = max(0, base_score - total_penalty)
        
        return score


def validate_nba_response(data: Union[List[Dict], Dict], endpoint_type: str, 
                         strict_mode: bool = False) -> ValidationResult:
    """
    Convenience function to validate NBA API response
    
    Args:
        data: Response data
        endpoint_type: Type of NBA API endpoint
        strict_mode: Whether to treat warnings as errors
        
    Returns:
        ValidationResult
    """
    validator = ResponseValidator(strict_mode=strict_mode)
    return validator.validate_response(data, endpoint_type)


# Example usage and testing
if __name__ == "__main__":
    # Example validation
    sample_player_data = [
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
        }
    ]
    
    result = validate_nba_response(sample_player_data, 'player_stats')
    print(result.get_summary())
    
    for issue in result.issues:
        print(f"  {issue.severity.value.upper()}: {issue.field} - {issue.message}")
