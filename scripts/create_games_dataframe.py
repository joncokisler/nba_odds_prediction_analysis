import nba_api.stats.static as static
from nba_api.stats.endpoints import leaguegamefinder
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_games_dataframe():
    from nba_api.stats.static import teams
    nba_teams = teams.get_teams()
    reg_playoff_games = ['002', '004', '005', '006']
    gamefinder = leaguegamefinder.LeagueGameFinder(date_from_nullable='09/09/2015', date_to_nullable='09/09/2020')
    gamefinder2 = leaguegamefinder.LeagueGameFinder(date_from_nullable='09/09/2020', date_to_nullable='03/13/2025')
    all_games = pd.concat([gamefinder.get_data_frames()[0],gamefinder2.get_data_frames()[0]], axis=0)
    all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])
    all_games = all_games[all_games['GAME_ID'].astype(str).str.startswith(('002', '004', '005', '006'))]
    all_games = all_games.dropna(subset=['WL'])
    teams = all_games["TEAM_NAME"].unique().tolist()
    games_sorted = all_games.sort_values(by='GAME_DATE').drop([
        'SEASON_ID', 'TEAM_ID'], axis=1)
    exclude_cols = ['TEAM_NAME', 'GAME_DATE', 'WL', 'GAME_ID']
    cols_to_roll = [col for col in games_sorted.columns if col not in exclude_cols]
    team_ma_A_list = []
    for team in teams:
        sorted_team_vals = games_sorted[games_sorted["TEAM_NAME"] == team].sort_values(by='GAME_DATE')
        re_add_cols = sorted_team_vals[exclude_cols]
        team_ma_A_list.append(sorted_team_vals[cols_to_roll].rolling(5, min_periods=1).mean(numeric_only=True).join(re_add_cols))
    df = pd.concat(team_ma_A_list)
    home_cols = [col for col in df.columns if "_HOME" in col]
    away_cols = [col.replace("_HOME_MA", "_AWAY_MA") for col in home_cols]
    concat_arr = []
    for name, groups in df.sort_values(by='GAME_ID').groupby(by=['GAME_ID']):
        try:
            g = groups.iloc[0].to_frame().transpose().merge(groups.iloc[1].to_frame().transpose(), on="GAME_ID",
                                                            suffixes=("_HOME", "_AWAY"))
            concat_arr.append(g)
        except:
            print("Something went wrong, ski;ping")
    all_games_df = pd.concat(concat_arr)
    all_games_df = all_games_df.set_index('GAME_ID')
    all_games_df.to_csv('all_games.csv')


if __name__ == "__main__":
    create_games_dataframe()
