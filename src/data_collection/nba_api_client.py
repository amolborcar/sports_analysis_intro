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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NBAApiClient:
    """Client for NBA.com APIs"""
    
    def __init__(self, delay_seconds: float = 1.0):
        """
        Initialize the NBA API client
        
        Args:
            delay_seconds: Delay between API calls to be respectful
        """
        self.delay_seconds = delay_seconds
        logger.info("NBA API Client initialized")
    
    def _wait_between_calls(self):
        """Add delay between API calls"""
        time.sleep(self.delay_seconds)
    
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
        try:
            logger.info(f"Fetching player stats for {season} season...")
            self._wait_between_calls()
            
            # Get league leaders (this gives us per game stats for all players)
            stats = leagueleaders.LeagueLeaders(
                season=season,
                season_type_all_star='Regular Season'
            )
            
            df = stats.get_data_frames()[0]
            
            # Convert to list of dictionaries and add metadata
            players_stats = df.to_dict('records')
            
            for player_stat in players_stats:
                player_stat['season'] = season
                player_stat['retrieved_at'] = datetime.now()
            
            logger.info(f"Retrieved stats for {len(players_stats)} players")
            return players_stats
            
        except Exception as e:
            logger.error(f"Error fetching player stats for {season}: {e}")
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
