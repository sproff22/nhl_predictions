# -*- coding: utf-8 -*-
"""
Module for generating initial NHL Elo ratings based on standings.

Author: Sam Roffman
"""

from nhlpy import NHLClient
import pandas as pd
from scipy.stats import norm
import numpy as np

def fetch_standings(client, season):
    """
    Fetch NHL standings for a given season.

    Parameters
    ----------
    client : nhlpy.NHLClient
        NHL API client.
    season : str
        Season identifier in the format 'YYYYYYYY' (e.g., '20232024').

    Returns
    -------
    list
        List of team standings data.
    """
    standings = client.standings
    standings_data = standings.get_standings(season=season)
    return standings_data['standings']

def generate_initial_elo(standings_data, mean_elo, std_dev_elo, n_teams):
    """
    Generate initial Elo ratings based on standings.

    Parameters
    ----------
    standings_data : list
        List of standings data with team abbreviations and rankings.
    mean_elo : int
        Mean value for Elo distribution.
    std_dev_elo : int
        Standard deviation for Elo distribution.
    n_teams : int
        Number of teams in the standings.

    Returns
    -------
    pd.DataFrame
        DataFrame containing team abbreviations, rankings, and initial Elo ratings.
    """
    team_names = [team['teamAbbrev']['default'] for team in standings_data]
    rankings = [team['leagueSequence'] for team in standings_data]

    np.random.seed(1)  # Set seed for reproducibility
    elos = norm.rvs(loc=mean_elo, scale=std_dev_elo, size=n_teams)
    elos.sort()  # Sort in ascending order to align low Elo with low rank

    # Create DataFrame and assign Elo ratings in descending order of ranking
    elo_df = pd.DataFrame({
        'team': team_names,
        '2324_ranking': rankings
    })
    elo_df['initial_elo'] = elos[::-1]

    # Replace 'ARI' with 'UTA' as a special case
    elo_df['team'] = elo_df['team'].replace('ARI', 'UTA')

    return elo_df

def save_initial_elo(elo_df, filepath):
    """
    Save the initial Elo ratings DataFrame to a CSV file.

    Parameters
    ----------
    elo_df : pd.DataFrame
        DataFrame containing initial Elo ratings.
    filepath : str
        Path to save the CSV file.
    """
    elo_df.to_csv(filepath, index=False)

def build_initial_elo():
    """
    Main function to generate and save initial NHL Elo ratings.
    """
    client = NHLClient()
    season = '20232024'
    standings_data = fetch_standings(client, season)

    mean_elo = 1500
    std_dev_elo = 30
    n_teams = 32

    initial_elo_df = generate_initial_elo(standings_data, mean_elo, std_dev_elo, n_teams)
    save_initial_elo(initial_elo_df, 'initial_elo.csv')

if __name__ == "__main__":
    build_initial_elo()
