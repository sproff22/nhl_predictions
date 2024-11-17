# NHL Elo Rating Analysis and Game Predictions

## Project Overview

This project is designed to calculate and track Elo ratings for NHL teams throughout a season and use these ratings to predict the outcomes of upcoming games. It also incorporates betting odds to provide insights into potential value bets.

The Elo rating system is a powerful method for evaluating the relative skill levels of players or teams in competitive games. This project leverages the Elo framework to analyze NHL games, calculate win probabilities, and compare implied betting odds to bookmaker lines.

## About the Elo Rating System

The Elo rating system, originally developed for chess by Arpad Elo, is a method for calculating the relative skill levels of players or teams. It updates ratings based on the outcome of a match, rewarding the winner and penalizing the loser. The key factors that determine Elo updates include:

- **Initial Ratings:** Teams start with a predefined rating.
- **Match Outcomes:** The difference in ratings predicts the expected outcome; deviations from the expected result adjust the ratings.
- **Home Advantage:** A constant modifier accounts for the advantage of playing at home.
- **Margin of Victory:** The Elo system can be adjusted to factor in the score difference, providing more granularity.
  
For more on Elo models and their applications in sports, see the FiveThirtyEight sports Elo explanation, which served as inspiration for this project.

## Features

### Initial Elo Rating Calculation:

`initial_elo.py` provides a starting point for the current season (2024 - 2025) by assigning each team an Elo rating based on their previous season`s results. This script uses the following methodology:

1) Sort teams from 1st to 32nd based on the final standings from the previous season.
2) Define a suggested starting rating distribution approximatelty `~ N(1500,30)`. These parameters were chosen based on exploratory data analysis and outside research from other similar projects.
3) Assign each team an Elo Rating within this distribution.

### Game Predictions:

`nhl_elo_model_script.py` iteratively calculates each team's probability to win each game, then calcultes each team's new Elo rating after each game. The calculations are largely derived from FiveThirtyEight's formulas, with adjustments made based on the results of a grid search. 

#### Formula for Home Team Win Probability

The probability of the home team winning is calculated as:

`p̂_home = 1 / (10^(-(elo_diff + home_adv) / spread_denom) + 1)`

Where:
- `p̂_home`: Probability of the home team winning.
- `p̂_away`: `1 - p̂_home`
- `elo_diff`: Elo rating difference between the home and away teams.
- `home_adv`: Home advantage constant.
- `spread_denom`: Scaling factor for the Elo probability spread.

#### Formula for Calculating Elo Change

1. **Margin of Victory Multiplier**:

`MV_mult = 0.6686 * log(margin_of_victory) + 0.8048 if margin_of_victory > 1, 1 otherwise`

2. **Autocorrelation Adjustment**:

`autocorr_adjustment = 2.05 / (abs(elo_diff) * 0.001 + 2.05)`

3. **Elo Change**:

`elo_change = K * MV_mult * (win_flag - win_prob) * autocorr_adjustment`

4. **Resulting Elo**:

`resulting_elo = starting_elo + elo_change`


This iterative process is continued until `yesterday`, which is freshly defined for each user with the datetime package.

### Predictions for Today`s Games, With Gambling Insights

`predict_games_today.py` uses each team's most up-to-date Elo ratings, today`s NHL schedule, and the current moneyline odds from FanDuel and DraftKings to provide win probabilities for each team, as well as if there is value in betting on any game on either of these online sportsbooks. 

## Workflow

### Prerequisites
- Python 3.8 or higher
- Conda environment (recommended for dependency management)
- Dependencies listed in `environment.yml`

### Installation

**1.Clone the repository:**
   ```bash
   git clone https://github.com/sproff22/nhl_predictions
   cd nhl-elo-analysis
   ```
**2. Set up the environment:**

  ```bash
  conda env create -f environment.yml
  conda activate hockey
  ```

### Run the project

Depending on your goals with the project, you can run different combinations of the python files. To run any file in the command line, use:

```bash
python {example_script.py}
```

### Prerequisite: Initial Elo Ratings:

```bash
python initial_elo.py
```

**Resulting File:**

- `initial_elo.csv` (Each team`s starting Elo rating for the beginning of the season)


### Option 1: Build season results (thus far) with Elo ratings, and save current Elo Power Rankings:

**Prerequisite File:** 

- `initial_elo.csv`

```bash
python nhl_elo_model_script.py
```

**Resulting Files:**

- `elo_rankings.csv` (Each team`s current Elo rating this season)
- `season_elo_results.csv` (A log of each game this season, with each team`s starting Elo, predicted win probability, and resulting Elo)
  
### Option 2: Predict today`s games, with printed suggestions in the command line:

**Prerequisite File:** 

- `season_elo_results.csv`

```bash
python predict_games_today.py
```

**Resulting File:**

- `predictions_today.csv` (Contains today`s scheduled games, moneylines on FanDuel and DraftKings, predicted win probabilities, and implied predicted moneylines based on the Elo system)

The suggested bets will be printed in the command line.

### Option 3: Analyze or utilize today`s prediction information any other way

Save and use the resulting file `predictions_today.csv`. 

## Additional Artifacts

Also included in the repository is a notebook file `model_tuning.ipynb` and an html file `model_tuning.html`. The notebook file contains testing and validation code with visualizations and performance metrics to examine model performance through the season thus far. It also contains a cell to perform a grid search on the possible hyperparameters to tune. These hyperparameters can be tuned on the full results from the previous season to allow for the maximum sample size. The corresponding html file is provided as an example of the model analysis, from November of 2024.

## References & Notes

Elo Rating System: [Wikipedia Entry on Elo Ratings](https://en.wikipedia.org/wiki/Elo_rating_system)

Inspiration from FiveThirtyEight: [How Elo Ratings Are Used in Sports](https://projects.fivethirtyeight.com/2023-nhl-predictions/)

NHL Data API: [Powered by the nhlpy Python package for accessing NHL data.
](https://pypi.org/project/nhl-api-py/)

*Note: It is not recommended to bet every game based on a slight discrepency between predicted probabilities and gambling odds. Elo is a purely Time-Series based rating system, which doesn`t take into account day-to-day factors such as injuries, starting goalies, trades, and more. Please gamble responsibly and at your own risk.* 

---
## Acknowledgments
This project draws inspiration from the work of [FiveThirtyEight](https://projects.fivethirtyeight.com/2023-nhl-predictions/), which popularized Elo models in sports analysis.

Special thanks to the developers of the nhlpy library for providing easy access to NHL data through their API.


