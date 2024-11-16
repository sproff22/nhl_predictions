# -*- coding: utf-8 -*-
"""
Module for analyzing NHL games and providing betting recommendations.

Author: Sam Roffman
"""

from nhlpy import NHLClient
import pandas as pd
from datetime import datetime

def fetch_games_for_today(client, date):
    """
    Fetch scheduled games for a given date.

    Parameters
    ----------
    client : nhlpy.NHLClient
        NHL API client.
    date : str
        Date in 'YYYY-MM-DD' format.

    Returns
    -------
    dict
        Dictionary containing schedule information for the given date.
    """
    return client.schedule.get_schedule(date=date)

def prepare_today_games_df(games_data):
    """
    Parse today's games into a DataFrame.

    Parameters
    ----------
    games_data : dict
        Dictionary containing games data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing parsed games information.
    """
    parsed_games = []
    for game in games_data['games']:
        game_info = {
            'date': games_data['date'],
            'home_team': game['homeTeam']['abbrev'],
            'away_team': game['awayTeam']['abbrev'],
            'neutral_site': game['neutralSite'],
            'home_team_odds_fanduel': next(
                (odds['value'] for odds in game['homeTeam'].get('odds', []) if odds['providerId'] == 7), None),
            'home_team_odds_draftkings': next(
                (odds['value'] for odds in game['homeTeam'].get('odds', []) if odds['providerId'] == 9), None),
            'away_team_odds_fanduel': next(
                (odds['value'] for odds in game['awayTeam'].get('odds', []) if odds['providerId'] == 7), None),
            'away_team_odds_draftkings': next(
                (odds['value'] for odds in game['awayTeam'].get('odds', []) if odds['providerId'] == 9), None)
        }
        parsed_games.append(game_info)
    return pd.DataFrame(parsed_games)

def calculate_elo_probabilities(today_games_df, elo_games_df, home_adv, spread_denom):
    """
    Calculate Elo probabilities and update today's games DataFrame.

    Parameters
    ----------
    today_games_df : pd.DataFrame
        DataFrame for today's games.
    elo_games_df : pd.DataFrame
        DataFrame containing Elo ratings for past games.
    home_adv : int
        Home advantage modifier.
    spread_denom : int
        Denominator for calculating Elo probabilities.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with Elo and probability columns.
    """
    elo_games_df = elo_games_df.sort_values(by='date')
    today_games_df = today_games_df.copy()
    today_games_df['home_elo'] = None
    today_games_df['away_elo'] = None
    today_games_df['home_prob'] = None
    today_games_df['away_prob'] = None

    for idx, row in today_games_df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']

        last_away_game = elo_games_df.loc[
            ((elo_games_df['home_team'] == away_team) | (elo_games_df['away_team'] == away_team)) &
            (elo_games_df['date'] <= row['date'])
        ].iloc[-1]

        last_home_game = elo_games_df.loc[
            ((elo_games_df['home_team'] == home_team) | (elo_games_df['away_team'] == home_team)) &
            (elo_games_df['date'] <= row['date'])
        ].iloc[-1]

        away_elo = last_away_game['home_resulting_elo'] if last_away_game['home_team'] == away_team else last_away_game['away_resulting_elo']
        home_elo = last_home_game['home_resulting_elo'] if last_home_game['home_team'] == home_team else last_home_game['away_resulting_elo']

        elo_diff = home_elo - away_elo
        home_prob = 1 / (10 ** (-1 * (elo_diff + home_adv) / spread_denom) + 1)
        away_prob = 1 - home_prob

        today_games_df.at[idx, 'home_elo'] = home_elo
        today_games_df.at[idx, 'away_elo'] = away_elo
        today_games_df.at[idx, 'home_prob'] = home_prob
        today_games_df.at[idx, 'away_prob'] = away_prob

    return today_games_df

def probability_to_american_odds(probability):
    """
    Convert probability to American odds format.

    Parameters
    ----------
    probability : float
        Probability value between 0 and 1.

    Returns
    -------
    float
        American odds equivalent of the probability.
    """
    if probability > 0.5:
        return -(probability / (1 - probability)) * 100
    else:
        return ((1 - probability) / probability) * 100

def apply_betting_odds(today_games_df):
    """
    Calculate and format implied betting odds for today's games.

    Parameters
    ----------
    today_games_df : pd.DataFrame
        DataFrame of today's games with calculated probabilities.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with implied odds columns.
    """
    today_games_df['home_implied_odds'] = today_games_df['home_prob'].apply(probability_to_american_odds).apply(
        lambda x: f"+{int(x)}" if x > 0 else f"{int(x)}")
    today_games_df['away_implied_odds'] = today_games_df['away_prob'].apply(probability_to_american_odds).apply(
        lambda x: f"+{int(x)}" if x > 0 else f"{int(x)}")
    return today_games_df

def suggest_bets(picks_df):
    """
    Suggest bets based on implied and actual odds.

    Parameters
    ----------
    picks_df : pd.DataFrame
        DataFrame with game odds and implied probabilities.
    """
    picks_df['home_team_odds_fanduel'] = pd.to_numeric(picks_df['home_team_odds_fanduel'], errors='coerce')
    picks_df['home_team_odds_draftkings'] = pd.to_numeric(picks_df['home_team_odds_draftkings'], errors='coerce')
    picks_df['away_team_odds_fanduel'] = pd.to_numeric(picks_df['away_team_odds_fanduel'], errors='coerce')
    picks_df['away_team_odds_draftkings'] = pd.to_numeric(picks_df['away_team_odds_draftkings'], errors='coerce')
    picks_df['home_implied_odds'] = pd.to_numeric(picks_df['home_implied_odds'], errors='coerce')
    picks_df['away_implied_odds'] = pd.to_numeric(picks_df['away_implied_odds'], errors='coerce')
    def format_odds(odds):
        return f"+{odds}" if odds > 0 else f"{odds}"

    for _, row in picks_df.iterrows():
        def format_odds(odds):
            return f"+{odds}" if odds > 0 else f"{odds}"
    
        # Print the matchup
        print(f"MATCHUP: {row['away_team']} vs. {row['home_team']}:")
        
        # Initialize a flag to check if any bet suggestion is made
        bet_suggested = False
        
        # Check if odds are unavailable on any platform and print the implied odds
        if pd.isnull(row['home_team_odds_fanduel']):
            print(f"No line available on FanDuel for this game. Implied odds are:")
            print(f"{row['home_team']}: {format_odds(row['home_implied_odds'])}, {row['away_team']}: {format_odds(row['away_implied_odds'])}")
            bet_suggested = True
    
        if pd.isnull(row['home_team_odds_draftkings']):
            print(f"No line available on DraftKings for this game. Implied odds are:")
            print(f"{row['home_team']}: {format_odds(row['home_implied_odds'])}, {row['away_team']}: {format_odds(row['away_implied_odds'])}")
            bet_suggested = True
        
        # Check for favorable home team bet on FanDuel
        if not pd.isnull(row['home_team_odds_fanduel']) and (
           (row['home_implied_odds'] < 0 and row['home_team_odds_fanduel'] > row['home_implied_odds']) or 
           (row['home_implied_odds'] > 0 and row['home_team_odds_fanduel'] > row['home_implied_odds'])):
            print(f"BET {row['home_team']} on FANDUEL at better than {format_odds(row['home_implied_odds'])}")
            bet_suggested = True
        
        # Check for favorable home team bet on DraftKings
        if not pd.isnull(row['home_team_odds_draftkings']) and (
           (row['home_implied_odds'] < 0 and row['home_team_odds_draftkings'] > row['home_implied_odds']) or 
           (row['home_implied_odds'] > 0 and row['home_team_odds_draftkings'] > row['home_implied_odds'])):
            print(f"BET {row['home_team']} on DRAFTKINGS at better than {format_odds(row['home_implied_odds'])}")
            bet_suggested = True
        
        # Check for favorable away team bet on FanDuel
        if not pd.isnull(row['away_team_odds_fanduel']) and (
           (row['away_implied_odds'] < 0 and row['away_team_odds_fanduel'] > row['away_implied_odds']) or 
           (row['away_implied_odds'] > 0 and row['away_team_odds_fanduel'] > row['away_implied_odds'])):
            print(f"BET {row['away_team']} on FANDUEL at better than {format_odds(row['away_implied_odds'])}")
            bet_suggested = True
        
        # Check for favorable away team bet on DraftKings
        if not pd.isnull(row['away_team_odds_draftkings']) and (
           (row['away_implied_odds'] < 0 and row['away_team_odds_draftkings'] > row['away_implied_odds']) or 
           (row['away_implied_odds'] > 0 and row['away_team_odds_draftkings'] > row['away_implied_odds'])):
            print(f"BET {row['away_team']} on DRAFTKINGS at better than {format_odds(row['away_implied_odds'])}")
            bet_suggested = True
        
        # If no bets were suggested, print "No bets."
        if not bet_suggested:
            print("No bets.")
    
        # Skip a line for the next game
        print("\n")

def predict_games_today():
    """
    Main function to fetch games, calculate probabilities, and suggest bets.
    """
    client = NHLClient()
    today = datetime.now().strftime('%Y-%m-%d')
    games_data = fetch_games_for_today(client, today)

    elo_games_df = pd.read_csv('season_elo_results.csv')
    today_games_df = prepare_today_games_df(games_data)

    home_adv = 10
    spread_denom = 400

    today_games_df = calculate_elo_probabilities(today_games_df, elo_games_df, home_adv, spread_denom)
    today_games_df = apply_betting_odds(today_games_df)
    suggest_bets(today_games_df)

    today_games_df.to_csv('predictions_today.csv')

if __name__ == "__main__":
    predict_games_today()


