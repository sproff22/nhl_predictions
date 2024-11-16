# -*- coding: utf-8 -*-
"""
Module for calculating NHL team Elo ratings throughout a season.

Author: Sam Roffman
"""

from nhlpy import NHLClient
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import csv

def load_initial_elo(filepath):
    """
    Load initial Elo ratings from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing initial Elo ratings.

    Returns
    -------
    dict
        Dictionary of teams and their initial Elo ratings.
    """
    return pd.read_csv(filepath).set_index('team')['initial_elo'].to_dict()

def fetch_season_games(client, season_start, season_end):
    """
    Fetch all games in the NHL season within a date range.

    Parameters
    ----------
    client : nhlpy.NHLClient
        NHL API client.
    season_start : datetime
        Start date of the season.
    season_end : datetime
        End date of the season.

    Returns
    -------
    pd.DataFrame
        DataFrame containing season games information.
    """
    current_date = season_start
    season_games = []

    while current_date <= season_end:
        try:
            schedule = client.schedule.get_schedule(date=current_date.strftime('%Y-%m-%d'))
            games_data = schedule.get('games', [])
            if games_data:
                for game in games_data:
                    season_games.append({
                        'date': schedule['date'],
                        'away_team': game['awayTeam']['abbrev'],
                        'away_score': game['awayTeam'].get('score'),
                        'home_team': game['homeTeam']['abbrev'],
                        'home_score': game['homeTeam'].get('score'),
                        'neutral_site': game.get('neutralSite', False),
                        'is_ot_so': game.get('periodDescriptor', {}).get('periodType') in ['OT', 'SO']
                    })
        except Exception:
            pass  # Handle missing data or API issues

        current_date += timedelta(days=1)

    return pd.DataFrame(season_games)

def calculate_elo_ratings(games_df, initial_elo, K, home_adv, spread_denom):
    """
    Calculate Elo ratings for all teams throughout the season.

    Parameters
    ----------
    games_df : pd.DataFrame
        DataFrame of season games.
    initial_elo : dict
        Initial Elo ratings for all teams.
    K : int
        Constant for Elo rating adjustment.
    home_adv : int
        Home team advantage modifier.
    spread_denom : int
        Denominator for calculating Elo spread.

    Returns
    -------
    pd.DataFrame, dict
        Updated games DataFrame with Elo ratings and final team Elo ratings.
    """
    games_df = games_df.sort_values(by='date').reset_index(drop=True)
    team_elo = initial_elo.copy()

    for idx, row in games_df.iterrows():
        away_team = row['away_team']
        home_team = row['home_team']
        away_elo = team_elo.get(away_team, 1500)
        home_elo = team_elo.get(home_team, 1500)

        games_df.at[idx, 'away_starting_elo'] = away_elo
        games_df.at[idx, 'home_starting_elo'] = home_elo

        elo_diff = home_elo - away_elo
        home_prob = 1 / (10 ** (-1 * (elo_diff + home_adv) / spread_denom) + 1)
        away_prob = 1 - home_prob

        games_df.at[idx, 'home_team_prob'] = home_prob
        games_df.at[idx, 'away_team_prob'] = away_prob

        if row['home_score'] > row['away_score']:
            win_flag = 1
            win_prob = home_prob
            margin_of_victory = row['home_score'] - row['away_score']
        else:
            win_flag = 0
            win_prob = away_prob
            margin_of_victory = row['away_score'] - row['home_score']

        MV_mult = 0.6686 * np.log(margin_of_victory) + 0.8048 if margin_of_victory > 1 else 1
        autocorr_adjustment = 2.05 / (abs(elo_diff) * 0.001 + 2.05)

        elo_change = K * MV_mult * (win_flag - win_prob) * autocorr_adjustment
        games_df.at[idx, 'home_resulting_elo'] = home_elo + elo_change
        games_df.at[idx, 'away_resulting_elo'] = away_elo - elo_change

        team_elo[home_team] = games_df.at[idx, 'home_resulting_elo']
        team_elo[away_team] = games_df.at[idx, 'away_resulting_elo']

    games_df['home_team_win'] = (games_df['home_score'] > games_df['away_score']).astype(int)

    return games_df, team_elo

def save_results(elo_games_df, final_elo, games_filepath, rankings_filepath):
    """
    Save results to CSV files.

    Parameters
    ----------
    elo_games_df : pd.DataFrame
        DataFrame of games with calculated Elo ratings.
    final_elo : dict
        Final Elo ratings for all teams.
    games_filepath : str
        Path to save the games data.
    rankings_filepath : str
        Path to save the Elo rankings data.
    """
    elo_games_df.to_csv(games_filepath, index=False)
    with open(rankings_filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Team", "Elo Score"])
        for team, elo in sorted(final_elo.items(), key=lambda item: item[1], reverse=True):
            writer.writerow([team, elo])

def build_elo_model():
    """
    Main function to execute the Elo calculation workflow.
    """
    client = NHLClient()
    initial_elo = load_initial_elo('initial_elo.csv')
    season_start = datetime(2024, 10, 8)
    season_end = datetime.now() - timedelta(days=1)

    games_df = fetch_season_games(client, season_start, season_end)

    K = 6
    home_adv = 10
    spread_denom = 400

    elo_games_df, final_elo = calculate_elo_ratings(games_df, initial_elo, K, home_adv, spread_denom)

    save_results(elo_games_df, final_elo, 'season_elo_results.csv', 'elo_rankings.csv')

if __name__ == "__main__":
    build_elo_model()
