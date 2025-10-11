"""
NBA API Client

A client for accessing NBA.com's official APIs through the nba_api package.
Collects player stats, team data, and game results for analysis.

Key features:
- Official NBA.com API access
- Robust error handling
- Clean data extraction
- Structured output for database storage
"""

from nba_api.stats.endpoints import (
    leagueleaders, 
    leaguedashteamstats, 
    playercareerstats,
    playerindex,
    leaguestandings,
    scoreboardv2,
    playergamelog
)
from nba_api.stats.static import players, teams
import pandas as pd
import logging
from datetime import datetime, date
from typing import Dict, List, Optional
import time
import requests
from functools import wraps
import threading
import signal
from .response_validator import validate_nba_response, ValidationSeverity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBAApiClient:
    """Client for NBA.com APIs"""
    
    def __init__(self, delay_seconds: float = 1.0, timeout_seconds: float = 30.0, max_retries: int = 3, 
                 enable_validation: bool = True, strict_validation: bool = False):
        """
        Initialize the NBA API client
        
        Args:
            delay_seconds: Delay between API calls to be respectful
            timeout_seconds: Timeout for API requests to prevent hanging
            max_retries: Maximum number of retry attempts for failed requests
            enable_validation: Whether to validate API responses
            strict_validation: Whether to treat validation warnings as errors
        """
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.enable_validation = enable_validation
        self.strict_validation = strict_validation
        logger.info(f"NBA API Client initialized (delay: {delay_seconds}s, timeout: {timeout_seconds}s, "
                   f"retries: {max_retries}, validation: {enable_validation}, strict: {strict_validation})")
    
    def _wait_between_calls(self):
        """Add delay between API calls"""
        time.sleep(self.delay_seconds)
    
    def _safe_api_call_single_attempt(self, api_call_func, *args, **kwargs):
        """
        Execute NBA API calls with timeout handling using threading
        
        Args:
            api_call_func: The NBA API function to call
            *args, **kwargs: Arguments to pass to the API function
            
        Returns:
            API response object or None if failed
        """
        func_name = getattr(api_call_func, '__name__', str(api_call_func))
        logger.info(f"Making API call: {func_name}")
        self._wait_between_calls()
        
        # Use threading for robust timeout
        result = [None]  # Mutable container for thread result
        exception = [None]  # Container for any exception
        
        def target():
            try:
                # Monkey patch requests to add timeout
                original_get = requests.get
                original_post = requests.post
                
                def get_with_timeout(*args, **kwargs):
                    kwargs.setdefault('timeout', self.timeout_seconds)
                    return original_get(*args, **kwargs)
                
                def post_with_timeout(*args, **kwargs):
                    kwargs.setdefault('timeout', self.timeout_seconds)
                    return original_post(*args, **kwargs)
                
                # Apply the patch
                requests.get = get_with_timeout
                requests.post = post_with_timeout
                
                try:
                    # Make the API call
                    result[0] = api_call_func(*args, **kwargs)
                finally:
                    # Restore original functions
                    requests.get = original_get
                    requests.post = original_post
                    
            except Exception as e:
                exception[0] = e
        
        # Run the API call in a separate thread
        thread = threading.Thread(target=target)
        thread.daemon = True  # Dies with main thread
        thread.start()
        thread.join(timeout=self.timeout_seconds)
        
        if thread.is_alive():
            # Thread is still running - timeout occurred
            logger.error(f"API call timed out after {self.timeout_seconds}s: {func_name}")
            return None
        
        if exception[0]:
            if isinstance(exception[0], requests.exceptions.Timeout):
                logger.error(f"Network timeout in API call: {func_name}")
            elif isinstance(exception[0], requests.exceptions.ConnectionError):
                logger.error(f"Connection error in API call: {func_name}")
            else:
                logger.error(f"Error in API call {func_name}: {exception[0]}")
            return None
        
        logger.info(f"API call successful: {func_name}")
        return result[0]
    
    def _safe_api_call(self, api_call_func, *args, **kwargs):
        """
        Execute NBA API calls with timeout handling and retry logic
        
        Args:
            api_call_func: The NBA API function to call
            *args, **kwargs: Arguments to pass to the API function
            
        Returns:
            API response object or None if all attempts failed
        """
        func_name = getattr(api_call_func, '__name__', str(api_call_func))
        
        for attempt in range(self.max_retries + 1):  # +1 because we count from 0
            if attempt > 0:
                # Calculate exponential backoff delay: 2^(attempt-1) seconds
                backoff_delay = 2 ** (attempt - 1)
                logger.info(f"Retrying {func_name} in {backoff_delay}s (attempt {attempt + 1}/{self.max_retries + 1})")
                time.sleep(backoff_delay)
            
            try:
                result = self._safe_api_call_single_attempt(api_call_func, *args, **kwargs)
                
                if result is not None:
                    if attempt > 0:
                        logger.info(f"API call succeeded on attempt {attempt + 1}: {func_name}")
                    return result
                else:
                    # Timeout or error occurred
                    if attempt < self.max_retries:
                        logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries + 1}): {func_name}")
                    else:
                        logger.error(f"API call failed after {self.max_retries + 1} attempts: {func_name}")
                        
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"Unexpected error on attempt {attempt + 1}: {e}")
                else:
                    logger.error(f"API call failed after {self.max_retries + 1} attempts with error: {e}")
        
        return None
    
    def _validate_response_data(self, data: List[Dict], endpoint_type: str) -> bool:
        """
        Validate response data quality and structure
        
        Args:
            data: Response data to validate
            endpoint_type: Type of endpoint for validation schema selection
            
        Returns:
            True if data passes validation, False otherwise
        """
        if not self.enable_validation:
            return True
        
        if not data:
            logger.warning(f"Empty response data for {endpoint_type}")
            return True  # Empty data is valid, just a warning
        
        try:
            validation_result = validate_nba_response(data, endpoint_type, self.strict_validation)
            
            # Log validation summary
            logger.info(f"Validation result for {endpoint_type}: {validation_result.get_summary()}")
            
            # Log detailed issues if any
            if validation_result.issues:
                for issue in validation_result.issues:
                    log_level = logging.ERROR if issue.severity == ValidationSeverity.ERROR else logging.WARNING
                    logger.log(log_level, f"Validation {issue.severity.value}: {issue.field} - {issue.message}")
                    
                    # Log sample values for debugging
                    if issue.sample_values:
                        logger.debug(f"Sample problematic values for {issue.field}: {issue.sample_values}")
            
            # Return validation result
            if not validation_result.is_valid:
                logger.error(f"Response validation failed for {endpoint_type}")
                return False
            
            # Log data quality score
            if validation_result.data_quality_score < 90:
                logger.warning(f"Data quality score below 90%: {validation_result.data_quality_score:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during response validation for {endpoint_type}: {e}")
            # In case of validation error, return True to not block the data (fail-open approach)
            return True
    
    def get_all_players(self) -> List[Dict]:
        """
        Get all NBA players from the static data
        
        Returns:
            List of player dictionaries with basic info
        """
        try:
            logger.info("Fetching all NBA players...")
            all_players = players.get_players()
            
            # Add timestamp
            for player in all_players:
                player['retrieved_at'] = datetime.now()
            
            # Validate response data
            if not self._validate_response_data(all_players, 'players'):
                logger.error("Player data failed validation")
                if self.strict_validation:
                    return []
            
            logger.info(f"Retrieved {len(all_players)} players")
            return all_players
            
        except Exception as e:
            logger.error(f"Error fetching players: {e}")
            return []
    
    def get_all_teams(self) -> List[Dict]:
        """
        Get all NBA teams from the static data
        
        Returns:
            List of team dictionaries with basic info
        """
        try:
            logger.info("Fetching all NBA teams...")
            all_teams = teams.get_teams()
            
            # Add timestamp
            for team in all_teams:
                team['retrieved_at'] = datetime.now()
            
            # Validate response data
            if not self._validate_response_data(all_teams, 'teams'):
                logger.error("Team data failed validation")
                if self.strict_validation:
                    return []
            
            logger.info(f"Retrieved {len(all_teams)} teams")
            return all_teams
            
        except Exception as e:
            logger.error(f"Error fetching teams: {e}")
            return []
    
    def get_player_season_stats(self, season: str = "2023-24") -> List[Dict]:
        """
        Get player stats for a given season
        
        Args:
            season: Season in format "2023-24"
            
        Returns:
            List of player stat dictionaries
        """
        logger.info(f"Fetching player stats for {season} season...")
        
        # Use safe API call with timeout handling
        stats = self._safe_api_call(
            leagueleaders.LeagueLeaders,
            season=season,
            season_type_all_star='Regular Season'
        )
        
        if stats is None:
            logger.error(f"Failed to fetch player stats for {season}")
            return []
        
        try:
            df = stats.get_data_frames()[0]
            
            # Convert to list of dictionaries and add metadata
            players_stats = df.to_dict('records')
            
            for player_stat in players_stats:
                player_stat['season'] = season
                player_stat['retrieved_at'] = datetime.now()
            
            # Validate response data
            if not self._validate_response_data(players_stats, 'player_stats'):
                logger.error(f"Player stats data failed validation for {season}")
                if self.strict_validation:
                    return []
            
            logger.info(f"Retrieved stats for {len(players_stats)} players")
            return players_stats
            
        except Exception as e:
            logger.error(f"Error processing player stats for {season}: {e}")
            return []
    
    def get_team_stats(self, season: str = "2023-24") -> List[Dict]:
        """
        Get team stats for a given season
        
        Args:
            season: Season in format "2023-24"
            
        Returns:
            List of team stat dictionaries
        """
        try:
            logger.info(f"Fetching team stats for {season} season...")
            self._wait_between_calls()
            
            # Get team stats
            stats = leaguedashteamstats.LeagueDashTeamStats(season=season)
            df = stats.get_data_frames()[0]
            
            # Convert to list of dictionaries and add metadata
            team_stats = df.to_dict('records')
            
            for team_stat in team_stats:
                team_stat['season'] = season
                team_stat['retrieved_at'] = datetime.now()
            
            logger.info(f"Retrieved stats for {len(team_stats)} teams")
            return team_stats
            
        except Exception as e:
            logger.error(f"Error fetching team stats for {season}: {e}")
            return []
    
    def get_standings(self, season: str = "2023-24") -> List[Dict]:
        """
        Get league standings for a given season
        
        Args:
            season: Season in format "2023-24"
            
        Returns:
            List of team standing dictionaries
        """
        try:
            logger.info(f"Fetching standings for {season} season...")
            self._wait_between_calls()
            
            standings = leaguestandings.LeagueStandings(season=season)
            df = standings.get_data_frames()[0]
            
            # Convert to list of dictionaries and add metadata
            standings_data = df.to_dict('records')
            
            for standing in standings_data:
                standing['season'] = season
                standing['retrieved_at'] = datetime.now()
            
            # Validate response data
            if not self._validate_response_data(standings_data, 'standings'):
                logger.error(f"Standings data failed validation for {season}")
                if self.strict_validation:
                    return []
            
            logger.info(f"Retrieved standings for {len(standings_data)} teams")
            return standings_data
            
        except Exception as e:
            logger.error(f"Error fetching standings for {season}: {e}")
            return []
    
    def get_player_career_stats(self, player_id: str) -> Dict:
        """
        Get career stats for a specific player
        
        Args:
            player_id: NBA player ID
            
        Returns:
            Dictionary with career stats
        """
        try:
            logger.info(f"Fetching career stats for player {player_id}...")
            self._wait_between_calls()
            
            career = playercareerstats.PlayerCareerStats(player_id=player_id)
            
            # Get different data frames
            season_totals = career.get_data_frames()[0]
            career_totals = career.get_data_frames()[1]
            
            result = {
                'player_id': player_id,
                'season_totals': season_totals.to_dict('records'),
                'career_totals': career_totals.to_dict('records'),
                'retrieved_at': datetime.now()
            }
            
            logger.info(f"Retrieved career stats for player {player_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching career stats for player {player_id}: {e}")
            return {}
    
    def get_games_for_date(self, game_date: str = None) -> List[Dict]:
        """
        Get games for a specific date
        
        Args:
            game_date: Date in YYYY-MM-DD format. If None, uses today
            
        Returns:
            List of game dictionaries
        """
        if game_date is None:
            game_date = date.today().strftime('%Y-%m-%d')
        
        try:
            logger.info(f"Fetching games for {game_date}...")
            self._wait_between_calls()
            
            scoreboard = scoreboardv2.ScoreboardV2(game_date=game_date)
            
            # Get game headers
            games_df = scoreboard.get_data_frames()[0]
            games_data = games_df.to_dict('records')
            
            for game in games_data:
                game['game_date'] = game_date
                game['retrieved_at'] = datetime.now()
            
            # Validate response data
            if not self._validate_response_data(games_data, 'games'):
                logger.error(f"Games data failed validation for {game_date}")
                if self.strict_validation:
                    return []
            
            logger.info(f"Retrieved {len(games_data)} games for {game_date}")
            return games_data
            
        except Exception as e:
            logger.error(f"Error fetching games for {game_date}: {e}")
            return []
    
    def find_player_by_name(self, player_name: str) -> Optional[Dict]:
        """
        Find a player by name using fuzzy matching
        
        Args:
            player_name: Player name to search for
            
        Returns:
            Player dictionary if found, None otherwise
        """
        try:
            # Get player using the static data
            player_dict = players.find_players_by_full_name(player_name)
            
            if player_dict:
                logger.info(f"Found player: {player_dict[0]}")
                return player_dict[0]
            else:
                logger.warning(f"No player found with name: {player_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error finding player {player_name}: {e}")
            return None


def main():
    """Example usage of the NBA API client"""
    client = NBAApiClient()
    
    # Get all players and teams
    print("=== NBA API Client Demo ===\n")
    
    # 1. Get basic data
    print("1. Getting all teams...")
    all_teams = client.get_all_teams()
    if all_teams:
        print(f"Found {len(all_teams)} teams")
        print("Sample team:", all_teams[0])
    
    print("\n2. Getting player stats for current season...")
    player_stats = client.get_player_season_stats("2023-24")
    if player_stats:
        print(f"Found stats for {len(player_stats)} players")
        
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(player_stats)
        # Print what columns we have for debugging purposes
        print("Available columns:", df.columns.tolist())
        print("First few rows:")
        print(df.head())
        print("\nTop 10 scorers:")
        top_scorers = df.nlargest(10, 'PTS')[['PLAYER', 'TEAM', 'PTS']]
        print(top_scorers.to_string(index=False))
    
    print("\n3. Getting team standings...")
    standings = client.get_standings("2023-24")
    if standings:
        df_standings = pd.DataFrame(standings)
        print("Eastern Conference standings:")
        east = df_standings[df_standings['Conference'] == 'East'].head(5)
        print(east[['TeamName', 'WINS', 'LOSSES', 'WinPCT']].to_string(index=False))
    
    print("\n4. Finding a specific player...")
    lebron = client.find_player_by_name("LeBron James")
    if lebron:
        print(f"Found: {lebron['full_name']} (ID: {lebron['id']})")
        
        # Get his career stats
        print("Getting LeBron's career stats...")
        career_stats = client.get_player_career_stats(str(lebron['id']))
        if career_stats and career_stats['season_totals']:
            df_career = pd.DataFrame(career_stats['season_totals'])
            print("LeBron's last 3 seasons:")
            recent = df_career.tail(3)[['SEASON_ID', 'TEAM_ABBREVIATION', 'GP', 'PTS', 'REB', 'AST']]
            print(recent.to_string(index=False))


if __name__ == "__main__":
    main()
