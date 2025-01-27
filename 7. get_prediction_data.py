#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from helpers import log, check_file_exists, load_csv, save_csv, remove_file_or_dir


# In[2]:


# This notebook lets us get csv with the data we want to predict

# First let's calculate rolling stats for each team that was active and add their defensive and offensive strengths

rolling_windows = [4, 16]
stats_to_roll = ["xG", "deep", "xGA", "deep_allowed"]
team_columns = [
    "id", "short_name", "strength_attack_home", "strength_attack_away",
    "strength_defence_home", "strength_defence_away"
]
input_folder = "Fantasy-Premier-League/data"
output_file = "Fantasy-Premier-League/data/prediction_files/teams-pred.csv"
output_folder = os.path.dirname(output_file)
os.makedirs(output_folder, exist_ok=True)

teams_file = os.path.join(input_folder, "2024-25/teams.csv")
if not check_file_exists(teams_file):
    log(f"Teams file not found: {teams_file}", level="error")
    raise FileNotFoundError(f"{teams_file} not found.")

teams = load_csv(teams_file)

all_team_data = []

for team_id in range(1, 21):
    team_file = os.path.join(input_folder, f"teams/team_{team_id}.csv")
    if not check_file_exists(team_file):
        log(f"Team file not found: {team_file}", level="warning")
        continue

    team_data = load_csv(team_file)
    team_data = team_data.sort_values(by="date")

    latest_row = team_data.iloc[[-1]].copy()
    rolling_stats = {}
    for window in rolling_windows:
        for stat in stats_to_roll:
            rolling_col = f"new_{stat}_rolling_{window}"
            rolling_stats[rolling_col] = team_data[stat].rolling(window, min_periods=1).mean().iloc[-1]

    for col, value in rolling_stats.items():
        latest_row.loc[:, col] = value

    team_info = teams[teams["id"] == team_id]
    if not team_info.empty:
        for col in team_columns:
            latest_row.loc[:, col] = team_info.iloc[0][col]

    latest_row = latest_row[team_columns + list(rolling_stats.keys())]

    all_team_data.append(latest_row)

if all_team_data:
    merged_data = pd.concat(all_team_data, ignore_index=True)
    save_csv(merged_data, output_file)
    log(f"Processed data saved to: {output_file}")
else:
    log("No team data was processed. Exiting.", level="warning")


# In[3]:


# Now let's merge that file with future fixtures to a new one

data_directory = "Fantasy-Premier-League/data/2024-25"
prediction_folder = "Fantasy-Premier-League/data/prediction_files"
fixtures_file = os.path.join(data_directory, "fixtures.csv")
teams_file = os.path.join(prediction_folder, "teams-pred.csv")
output_file = os.path.join(prediction_folder, "fixtures-with-info.csv")
fixtures = load_csv(fixtures_file)
teams_pred = load_csv(teams_file)
if fixtures is None or teams_pred is None:
    log("Required CSV files not found. Exiting.", level="ERROR")
    raise FileNotFoundError("Fixtures or teams-pred file not found.")

unfinished_fixtures = fixtures[~fixtures["finished"]]
next_8_gameweeks = unfinished_fixtures["event"].dropna().unique()[:8]
filtered_fixtures = unfinished_fixtures[unfinished_fixtures["event"].isin(next_8_gameweeks)].copy()

for _, team_row in teams_pred.iterrows():
    team_id = team_row["id"]
    for _, fixture_row in filtered_fixtures.iterrows():
        if fixture_row["team_h"] == team_id:
            team_column_prefix = "team_h"
            attack_column = "strength_attack_home"
            defence_column = "strength_defence_home"
            short_name_column = "short_name_h"
        elif fixture_row["team_a"] == team_id:
            team_column_prefix = "team_a"
            attack_column = "strength_attack_away"
            defence_column = "strength_defence_away"
            short_name_column = "short_name_a"
        else:
            continue

        for stat in ["xG", "deep", "xGA", "deep_allowed"]:
            for window in [4, 16]:
                stat_column = f"new_{stat}_rolling_{window}"
                target_column = f"{team_column_prefix}_{stat}_rolling_{window}"
                filtered_fixtures.loc[fixture_row.name, target_column] = team_row[stat_column]
                
        filtered_fixtures.loc[fixture_row.name, attack_column] = team_row[attack_column]
        filtered_fixtures.loc[fixture_row.name, defence_column] = team_row[defence_column]
        filtered_fixtures.loc[fixture_row.name, short_name_column] = team_row["short_name"]
        
columns_to_keep = [
    "event", "team_h", "team_a", "short_name_h", "short_name_a", 
    "strength_attack_home", "strength_defence_home", 
    "strength_attack_away", "strength_defence_away"
] + [
    f"team_h_{stat}_rolling_{window}" for stat in ["xG", "deep", "xGA", "deep_allowed"] for window in [4, 16]
] + [
    f"team_a_{stat}_rolling_{window}" for stat in ["xG", "deep", "xGA", "deep_allowed"] for window in [4, 16]
]

final_data = filtered_fixtures[columns_to_keep]

# Zapisanie wynikowego pliku
os.makedirs(prediction_folder, exist_ok=True)
save_csv(final_data, output_file)
log(f"Fixtures with info saved to: {output_file}")


# In[4]:


# Once we have that we can make players_pred.csv where we will add current clubs of the
# players and their values, we base on Fantasy-Premier-League/data/master_player_v2.csv and iterate through ones active in
# 2024-24
# player_raw has element_type (1, 2, 3, 4) which corresponds to positions, in the prediction_files folder we want to have players
# for each position
data_directory = "Fantasy-Premier-League/data"
master_players_file = os.path.join(data_directory, "master_player_v2.csv")
players_raw_file = os.path.join(data_directory, "2024-25/players_raw.csv")

master_players = load_csv(master_players_file)
players_raw = load_csv(players_raw_file)

if master_players is None or players_raw is None:
    log("Failed to load required CSV files. Exiting.", level="ERROR")
    exit()
    
merged_data = pd.merge(
    master_players[['24_id', 'Full_Name']],
    players_raw[['id', 'now_cost', 'team', 'element_type']],
    left_on='24_id',
    right_on='id',
    how='left'
).rename(columns={
    'now_cost': 'value',
    'team': 'own_team',
    'element_type': 'position',
    'id': 'Unique_ID'
})
merged_data = merged_data.drop(columns=['24_id'])

position_mapping = {
    1: "goalkeepers",
    2: "defenders",
    3: "midfielders",
    4: "forwards"
}

for position, filename in position_mapping.items():
    position_data = merged_data[merged_data['position'] == position]
    position_output_file = os.path.join(data_directory, "prediction_files", f"{filename}.csv")
    save_csv(position_data, position_output_file)
    log(f"Saved {filename} data to: {position_output_file}")


# In[5]:


# Now we calculate must needed things for goalkeepers

# We iterate through goalkeepers.csv, based on Unique_ID.toInt we find matching folder in data/consolidated_players. The file ends with
# player's unique id
# things we need to calculate
# selected_features = [
#     'bps_rolling_16', 'influence_rolling_4', 'influence_rolling_16', 
#     'clean_sheets_rolling_4'
# ]
# we add them to goalkeepers.csv, we do this just for the newest possible data
data_directory = "Fantasy-Premier-League/data/"
goalkeepers_file = os.path.join(data_directory, "prediction_files/goalkeepers.csv")
consolidated_players_dir = os.path.join(data_directory, "consolidated_players")
output_file = os.path.join(data_directory, "prediction_files/goalkeepers_v2.csv")

goalkeepers = load_csv(goalkeepers_file)
if goalkeepers is None:
    log("Failed to load goalkeepers data. Exiting.", level="ERROR")
    exit()

selected_features = {
    "bps": [16],
    "influence": [4, 16],
    "clean_sheets": [4]
}

for _, row in goalkeepers.iterrows():
    unique_id = int(row["Unique_ID"])
    
    matching_files = [
        f for f in os.listdir(consolidated_players_dir) 
        if f.endswith(f"_{unique_id}.csv")
    ]
    
    if not matching_files:
        log(f"No matching file found for Unique_ID {unique_id}. Setting zeros.", level="WARNING")
        for stat, windows in selected_features.items():
            for window in windows:
                rolling_col = f"{stat}_rolling_{window}"
                goalkeepers.at[row.name, rolling_col] = 0
        goalkeepers.at[row.name, "xMins"] = 0
        continue
    
    player_file = os.path.join(consolidated_players_dir, matching_files[0])
    player_data = load_csv(player_file)
    
    if player_data is None or "kickoff_time" not in player_data.columns:
        log(f"Invalid player data for Unique_ID {unique_id}. Skipping.", level="WARNING")
        continue

    player_data["date"] = pd.to_datetime(player_data["kickoff_time"], errors="coerce")
    player_data["date"] = player_data["date"].dt.tz_localize(None)
    player_data = player_data.sort_values(by="date")

    last_4_matches = player_data.tail(4).copy()
    last_4_matches["filtered_minutes"] = last_4_matches["minutes"]

    zero_indices = last_4_matches[last_4_matches["minutes"] == 0].index
    if len(zero_indices) > 0:
        first_zero_idx = zero_indices[0]
        last_4_matches.at[first_zero_idx, "filtered_minutes"] = pd.NA

    filtered_minutes = last_4_matches["filtered_minutes"].dropna()
    if not filtered_minutes.empty:
        xMins = filtered_minutes.mean()
    else:
        xMins = 0
    goalkeepers.at[row.name, "xMins"] = xMins

    player_data = player_data[player_data["minutes"] >= 60]

    if player_data.empty:
        log(f"Player {unique_id} has no matches with >= 60 minutes. Setting zeros.", level="DEBUG")
        for stat, windows in selected_features.items():
            for window in windows:
                rolling_col = f"{stat}_rolling_{window}"
                goalkeepers.at[row.name, rolling_col] = 0
        continue
    
    latest_row = player_data.iloc[-1]
    for stat, windows in selected_features.items():
        for window in windows:
            rolling_col = f"{stat}_rolling_{window}"
            if stat in player_data.columns:
                latest_value = player_data[stat].rolling(window=window, min_periods=1).mean().iloc[-1]
                goalkeepers.at[row.name, rolling_col] = latest_value
            else:
                log(f"Stat {stat} not found for Unique_ID {unique_id}. Skipping.", level="WARNING")

save_csv(goalkeepers, output_file)
log(f"Goalkeepers with features saved to: {output_file}")


# In[6]:


# Now we merge fixtures to goalkeepers

data_directory = "Fantasy-Premier-League/data/"
fixtures_file = os.path.join(data_directory, "prediction_files/fixtures-with-info.csv")
goalkeepers_file = os.path.join(data_directory, "prediction_files/goalkeepers_v2.csv")
output_file = os.path.join(data_directory, "prediction_files/gk-ready.csv")

fixtures = load_csv(fixtures_file)
goalkeepers = load_csv(goalkeepers_file)

if fixtures is None or goalkeepers is None:
    log("Failed to load required data files. Exiting.", level="ERROR")
    exit()
    
columns_to_include = [
    "def_atk_diff", "opponent_deep_rolling_4", "opponent_xG_rolling_16",
    "own_defense", "team_xGA_rolling_16", "team_deep_allowed_rolling_16",
    "was_home", "event", "short_name_h", "short_name_a", "opponent_team"
]

prediction_data = []

for _, fixture in fixtures.iterrows():
    for team_column, was_home in [("team_h", True), ("team_a", False)]:
        team_id = fixture[team_column]
        opponent_column = "team_h" if team_column == "team_a" else "team_a"
        opponent_id = fixture[opponent_column]
        team_gks = goalkeepers[goalkeepers["own_team"] == team_id].copy()
        team_gks["def_atk_diff"] = (
            fixture[f"strength_defence_{'home' if was_home else 'away'}"]
            - fixture[f"strength_attack_{'home' if not was_home else 'away'}"]
        )
        team_gks["opponent_deep_rolling_4"] = fixture[f"{opponent_column}_deep_rolling_4"]
        team_gks["opponent_xG_rolling_16"] = fixture[f"{opponent_column}_xG_rolling_16"]
        team_gks["own_defense"] = fixture[f"strength_defence_{'home' if was_home else 'away'}"]
        team_gks["team_xGA_rolling_16"] = fixture[f"{team_column}_xGA_rolling_16"]
        team_gks["team_deep_allowed_rolling_16"] = fixture[f"{team_column}_deep_allowed_rolling_16"]
        team_gks["was_home"] = was_home
        team_gks["event"] = fixture["event"]
        team_gks["short_name_h"] = fixture["short_name_h"]
        team_gks["short_name_a"] = fixture["short_name_a"]
        team_gks["opponent_team"] = opponent_id
        
        prediction_data.append(team_gks)

if prediction_data:
    prediction_df = pd.concat(prediction_data, ignore_index=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    save_csv(prediction_df, output_file)
    log(f"Goalkeeper predictions saved to: {output_file}")
else:
    log("No prediction data generated. Exiting.", level="WARNING")


# In[7]:


# Now we calculate xMins for defenders

data_directory = "Fantasy-Premier-League/data/"
defenders_file = os.path.join(data_directory, "prediction_files/defenders.csv")
consolidated_players_dir = os.path.join(data_directory, "consolidated_players")
output_file = os.path.join(data_directory, "prediction_files/defenders_v2.csv")

defenders = load_csv(defenders_file)
if defenders is None:
    log("Failed to load defenders data. Exiting.", level="ERROR")
    exit()

for _, row in defenders.iterrows():
    unique_id = int(row["Unique_ID"])
    
    matching_files = [
        f for f in os.listdir(consolidated_players_dir) 
        if f.endswith(f"_{unique_id}.csv")
    ]
    
    if not matching_files:
        log(f"No matching file found for Unique_ID {unique_id}. Setting xMins to 0.", level="WARNING")
        defenders.at[row.name, "xMins"] = 0
        continue
    
    player_file = os.path.join(consolidated_players_dir, matching_files[0])
    player_data = load_csv(player_file)
    
    if player_data is None or "kickoff_time" not in player_data.columns:
        log(f"Invalid player data for Unique_ID {unique_id}. Skipping.", level="WARNING")
        defenders.at[row.name, "xMins"] = 0
        continue

    player_data["date"] = pd.to_datetime(player_data["kickoff_time"], errors="coerce")
    player_data["date"] = player_data["date"].dt.tz_localize(None)
    player_data = player_data.sort_values(by="date")

    # Calculate xMins
    last_4_matches = player_data.tail(4).copy()
    last_4_matches["filtered_minutes"] = last_4_matches["minutes"]

    zero_indices = last_4_matches[last_4_matches["minutes"] == 0].index
    if len(zero_indices) > 0:
        first_zero_idx = zero_indices[0]
        last_4_matches.at[first_zero_idx, "filtered_minutes"] = pd.NA

    filtered_minutes = last_4_matches["filtered_minutes"].dropna()
    if not filtered_minutes.empty:
        xMins = filtered_minutes.mean()
    else:
        xMins = 0
    defenders.at[row.name, "xMins"] = xMins

save_csv(defenders, output_file)
log(f"Defenders with xMins saved to: {output_file}")


# In[8]:


# Now we merge fixtures to defenders

data_directory = "Fantasy-Premier-League/data/"
fixtures_file = os.path.join(data_directory, "prediction_files/fixtures-with-info.csv")
defenders_file = os.path.join(data_directory, "prediction_files/defenders_v2.csv")
output_file = os.path.join(data_directory, "prediction_files/def-ready.csv")

fixtures = load_csv(fixtures_file)
defenders = load_csv(defenders_file)

if fixtures is None or defenders is None:
    log("Failed to load required data files. Exiting.", level="ERROR")
    exit()

columns_to_include = [
    "atk_def_diff", "def_atk_diff", "opponent_xG_rolling_4", "opponent_deep_rolling_4",
    "opponent_xGA_rolling_4", "opponent_deep_allowed_rolling_4", "team_xG_rolling_4", 
    "team_xGA_rolling_4", "opponent_xGA_rolling_16", "opponent_deep_allowed_rolling_16", 
    "opponent_xG_rolling_16", "opponent_deep_rolling_16", "opponent_defense", "own_defense", 
    "team_xG_rolling_16", "team_deep_rolling_16", "team_xGA_rolling_16", "team_deep_allowed_rolling_16", 
    "was_home", "event", "short_name_h", "short_name_a", "opponent_team"
]

prediction_data = []

for _, fixture in fixtures.iterrows():
    for team_column, was_home in [("team_h", True), ("team_a", False)]:
        team_id = fixture[team_column]
        opponent_column = "team_h" if team_column == "team_a" else "team_a"
        opponent_id = fixture[opponent_column]
        team_defs = defenders[defenders["own_team"] == team_id].copy()
        
        # Calculating features
        team_defs["atk_def_diff"] = (
            fixture[f"strength_attack_{'home' if was_home else 'away'}"]
            - fixture[f"strength_defence_{'home' if not was_home else 'away'}"]
        )
        team_defs["def_atk_diff"] = (
            fixture[f"strength_defence_{'home' if was_home else 'away'}"]
            - fixture[f"strength_attack_{'home' if not was_home else 'away'}"]
        )
        team_defs["opponent_xG_rolling_4"] = fixture[f"{opponent_column}_xG_rolling_4"]
        team_defs["opponent_deep_rolling_4"] = fixture[f"{opponent_column}_deep_rolling_4"]
        team_defs["opponent_xGA_rolling_4"] = fixture[f"{opponent_column}_xGA_rolling_4"]
        team_defs["opponent_deep_allowed_rolling_4"] = fixture[f"{opponent_column}_deep_allowed_rolling_4"]
        team_defs["team_xG_rolling_4"] = fixture[f"{team_column}_xG_rolling_4"]
        team_defs["team_xGA_rolling_4"] = fixture[f"{team_column}_xGA_rolling_4"]
        team_defs["opponent_xGA_rolling_16"] = fixture[f"{opponent_column}_xGA_rolling_16"]
        team_defs["opponent_deep_allowed_rolling_16"] = fixture[f"{opponent_column}_deep_allowed_rolling_16"]
        team_defs["opponent_xG_rolling_16"] = fixture[f"{opponent_column}_xG_rolling_16"]
        team_defs["opponent_deep_rolling_16"] = fixture[f"{opponent_column}_deep_rolling_16"]
        team_defs["opponent_defense"] = fixture[f"strength_defence_{'home' if not was_home else 'away'}"]
        team_defs["own_defense"] = fixture[f"strength_defence_{'home' if was_home else 'away'}"]
        team_defs["team_xG_rolling_16"] = fixture[f"{team_column}_xG_rolling_16"]
        team_defs["team_deep_rolling_16"] = fixture[f"{team_column}_deep_rolling_16"]
        team_defs["team_xGA_rolling_16"] = fixture[f"{team_column}_xGA_rolling_16"]
        team_defs["team_deep_allowed_rolling_16"] = fixture[f"{team_column}_deep_allowed_rolling_16"]
        team_defs["was_home"] = was_home
        team_defs["event"] = fixture["event"]
        team_defs["short_name_h"] = fixture["short_name_h"]
        team_defs["short_name_a"] = fixture["short_name_a"]
        team_defs["opponent_team"] = opponent_id
        
        prediction_data.append(team_defs)

if prediction_data:
    prediction_df = pd.concat(prediction_data, ignore_index=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    save_csv(prediction_df, output_file)
    log(f"Defender predictions saved to: {output_file}")
else:
    log("No prediction data generated. Exiting.", level="WARNING")


# In[9]:


# Now we calculate must needed things for midfielders

data_directory = "Fantasy-Premier-League/data/"
midfielders_file = os.path.join(data_directory, "prediction_files/midfielders.csv")
consolidated_players_dir = os.path.join(data_directory, "consolidated_players")
output_file = os.path.join(data_directory, "prediction_files/midfielders_v2.csv")

midfielders = load_csv(midfielders_file)
if midfielders is None:
    log("Failed to load midfielders data. Exiting.", level="ERROR")
    exit()

selected_features = {
    "shots": [4, 16],
    "key_passes": [16],
    "expected_goals": [4, 16],
    "expected_assists": [4],
    "goals_scored": [4, 16],
    "ict_index": [4, 16],
    "influence": [16],
    "creativity": [16],
    "threat": [4, 16],
    "assists": [4, 16],
    "total_points": [4, 16]
}

for _, row in midfielders.iterrows():
    unique_id = int(row["Unique_ID"])
    
    matching_files = [
        f for f in os.listdir(consolidated_players_dir) 
        if f.endswith(f"_{unique_id}.csv")
    ]
    
    if not matching_files:
        log(f"No matching file found for Unique_ID {unique_id}. Setting zeros.", level="WARNING")
        for stat, windows in selected_features.items():
            for window in windows:
                rolling_col = f"{stat}_rolling_{window}"
                midfielders.at[row.name, rolling_col] = 0
        midfielders.at[row.name, "xMins"] = 0
        continue
    
    player_file = os.path.join(consolidated_players_dir, matching_files[0])
    player_data = load_csv(player_file)
    
    if player_data is None or "kickoff_time" not in player_data.columns:
        log(f"Invalid player data for Unique_ID {unique_id}. Setting zeros.", level="WARNING")
        for stat, windows in selected_features.items():
            for window in windows:
                rolling_col = f"{stat}_rolling_{window}"
                midfielders.at[row.name, rolling_col] = 0
        midfielders.at[row.name, "xMins"] = 0
        continue

    player_data["date"] = pd.to_datetime(player_data["kickoff_time"], errors="coerce")
    player_data["date"] = player_data["date"].dt.tz_localize(None)
    player_data = player_data.sort_values(by="date")

    # Calculate xMins
    last_4_matches = player_data.tail(4).copy()
    last_4_matches["filtered_minutes"] = last_4_matches["minutes"]

    zero_indices = last_4_matches[last_4_matches["minutes"] == 0].index
    if len(zero_indices) > 0:
        first_zero_idx = zero_indices[0]
        last_4_matches.at[first_zero_idx, "filtered_minutes"] = pd.NA

    filtered_minutes = last_4_matches["filtered_minutes"].dropna()
    if not filtered_minutes.empty:
        xMins = filtered_minutes.mean()
    else:
        xMins = 0
    midfielders.at[row.name, "xMins"] = xMins

    # Filter rows with at least 60 minutes
    player_data = player_data[player_data["minutes"] >= 60]

    if player_data.empty:
        log(f"Player {unique_id} has no matches with >= 60 minutes. Setting zeros.", level="DEBUG")
        for stat, windows in selected_features.items():
            for window in windows:
                rolling_col = f"{stat}_rolling_{window}"
                midfielders.at[row.name, rolling_col] = 0
        continue

    # Calculate rolling stats
    for stat, windows in selected_features.items():
        for window in windows:
            rolling_col = f"{stat}_rolling_{window}"
            if stat in player_data.columns:
                latest_value = player_data[stat].rolling(window=window, min_periods=1).mean().iloc[-1]
                midfielders.at[row.name, rolling_col] = latest_value
            else:
                log(f"Stat {stat} not found for Unique_ID {unique_id}. Setting zero for {rolling_col}.", level="WARNING")
                midfielders.at[row.name, rolling_col] = 0

save_csv(midfielders, output_file)
log(f"Midfielders with features saved to: {output_file}")


# In[10]:


# Now we merge fixtures to midfielders

data_directory = "Fantasy-Premier-League/data/"
fixtures_file = os.path.join(data_directory, "prediction_files/fixtures-with-info.csv")
midfielders_file = os.path.join(data_directory, "prediction_files/midfielders_v2.csv")
output_file = os.path.join(data_directory, "prediction_files/mid-ready.csv")

fixtures = load_csv(fixtures_file)
midfielders = load_csv(midfielders_file)

if fixtures is None or midfielders is None:
    log("Failed to load required data files. Exiting.", level="ERROR")
    exit()

columns_to_include = [
    "atk_def_diff", "def_atk_diff", "opponent_xGA_rolling_4", "team_xG_rolling_4", 
    "team_deep_rolling_4", "opponent_xGA_rolling_16", "opponent_deep_allowed_rolling_16", 
    "own_attack", "opponent_defense", "own_defense", "team_xG_rolling_16", "team_deep_rolling_16", 
    "was_home", "event", "short_name_h", "short_name_a", "opponent_team"
]

prediction_data = []

for _, fixture in fixtures.iterrows():
    for team_column, was_home in [("team_h", True), ("team_a", False)]:
        team_id = fixture[team_column]
        opponent_column = "team_h" if team_column == "team_a" else "team_a"
        opponent_id = fixture[opponent_column]
        team_mids = midfielders[midfielders["own_team"] == team_id].copy()
        
        # Calculating features
        team_mids["atk_def_diff"] = (
            fixture[f"strength_attack_{'home' if was_home else 'away'}"]
            - fixture[f"strength_defence_{'home' if not was_home else 'away'}"]
        )
        team_mids["def_atk_diff"] = (
            fixture[f"strength_defence_{'home' if was_home else 'away'}"]
            - fixture[f"strength_attack_{'home' if not was_home else 'away'}"]
        )
        team_mids["opponent_xGA_rolling_4"] = fixture[f"{opponent_column}_xGA_rolling_4"]
        team_mids["team_xG_rolling_4"] = fixture[f"{team_column}_xG_rolling_4"]
        team_mids["team_deep_rolling_4"] = fixture[f"{team_column}_deep_rolling_4"]
        team_mids["opponent_xGA_rolling_16"] = fixture[f"{opponent_column}_xGA_rolling_16"]
        team_mids["opponent_deep_allowed_rolling_16"] = fixture[f"{opponent_column}_deep_allowed_rolling_16"]
        team_mids["own_attack"] = fixture[f"strength_attack_{'home' if was_home else 'away'}"]
        team_mids["opponent_defense"] = fixture[f"strength_defence_{'home' if not was_home else 'away'}"]
        team_mids["own_defense"] = fixture[f"strength_defence_{'home' if was_home else 'away'}"]
        team_mids["team_xG_rolling_16"] = fixture[f"{team_column}_xG_rolling_16"]
        team_mids["team_deep_rolling_16"] = fixture[f"{team_column}_deep_rolling_16"]
        team_mids["was_home"] = was_home
        team_mids["event"] = fixture["event"]
        team_mids["short_name_h"] = fixture["short_name_h"]
        team_mids["short_name_a"] = fixture["short_name_a"]
        team_mids["opponent_team"] = opponent_id
        
        prediction_data.append(team_mids)

if prediction_data:
    prediction_df = pd.concat(prediction_data, ignore_index=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    save_csv(prediction_df, output_file)
    log(f"Midfielder predictions saved to: {output_file}")
else:
    log("No prediction data generated. Exiting.", level="WARNING")


# In[11]:


data_directory = "Fantasy-Premier-League/data/"
forwards_file = os.path.join(data_directory, "prediction_files/forwards.csv")
consolidated_players_dir = os.path.join(data_directory, "consolidated_players")
output_file = os.path.join(data_directory, "prediction_files/forwards_v2.csv")

forwards = load_csv(forwards_file)
if forwards is None:
    log("Failed to load forwards data. Exiting.", level="ERROR")
    exit()

selected_features = {
    "shots": [4],
    "expected_goals": [4],
    "expected_assists": [16],
    "goals_scored": [4, 16],
    "bps": [16],
    "ict_index": [4, 16],
    "influence": [4],
    "threat": [16],
    "assists": [4, 16],
    "total_points": [16]
}

for _, row in forwards.iterrows():
    unique_id = int(row["Unique_ID"])
    
    matching_files = [
        f for f in os.listdir(consolidated_players_dir) 
        if f.endswith(f"_{unique_id}.csv")
    ]
    
    if not matching_files:
        log(f"No matching file found for Unique_ID {unique_id}. Setting zeros.", level="WARNING")
        for stat, windows in selected_features.items():
            for window in windows:
                rolling_col = f"{stat}_rolling_{window}"
                forwards.at[row.name, rolling_col] = 0
        forwards.at[row.name, "xMins"] = 0
        continue
    
    player_file = os.path.join(consolidated_players_dir, matching_files[0])
    player_data = load_csv(player_file)
    
    if player_data is None or "kickoff_time" not in player_data.columns:
        log(f"Invalid player data for Unique_ID {unique_id}. Setting zeros.", level="WARNING")
        for stat, windows in selected_features.items():
            for window in windows:
                rolling_col = f"{stat}_rolling_{window}"
                forwards.at[row.name, rolling_col] = 0
        forwards.at[row.name, "xMins"] = 0
        continue

    player_data["date"] = pd.to_datetime(player_data["kickoff_time"], errors="coerce")
    player_data["date"] = player_data["date"].dt.tz_localize(None)
    player_data = player_data.sort_values(by="date")

    # Calculate xMins
    last_4_matches = player_data.tail(4).copy()
    last_4_matches["filtered_minutes"] = last_4_matches["minutes"]

    zero_indices = last_4_matches[last_4_matches["minutes"] == 0].index
    if len(zero_indices) > 0:
        first_zero_idx = zero_indices[0]
        last_4_matches.at[first_zero_idx, "filtered_minutes"] = pd.NA

    filtered_minutes = last_4_matches["filtered_minutes"].dropna()
    if not filtered_minutes.empty:
        xMins = filtered_minutes.mean()
    else:
        xMins = 0
    forwards.at[row.name, "xMins"] = xMins

    # Filter rows with at least 60 minutes
    player_data = player_data[player_data["minutes"] >= 60]

    if player_data.empty:
        log(f"Player {unique_id} has no matches with >= 60 minutes. Setting zeros.", level="DEBUG")
        for stat, windows in selected_features.items():
            for window in windows:
                rolling_col = f"{stat}_rolling_{window}"
                forwards.at[row.name, rolling_col] = 0
        continue

    # Calculate rolling stats
    for stat, windows in selected_features.items():
        for window in windows:
            rolling_col = f"{stat}_rolling_{window}"
            if stat in player_data.columns:
                latest_value = player_data[stat].rolling(window=window, min_periods=1).mean().iloc[-1]
                forwards.at[row.name, rolling_col] = latest_value
            else:
                log(f"Stat {stat} not found for Unique_ID {unique_id}. Setting zero for {rolling_col}.", level="DEBUG")
                forwards.at[row.name, rolling_col] = 0

save_csv(forwards, output_file)
log(f"Forwards with features saved to: {output_file}")


# In[12]:


# Now we merge fixtures to forwards

data_directory = "Fantasy-Premier-League/data/"
fixtures_file = os.path.join(data_directory, "prediction_files/fixtures-with-info.csv")
forwards_file = os.path.join(data_directory, "prediction_files/forwards_v2.csv")
output_file = os.path.join(data_directory, "prediction_files/fwd-ready.csv")

fixtures = load_csv(fixtures_file)
forwards = load_csv(forwards_file)

if fixtures is None or forwards is None:
    log("Failed to load required data files. Exiting.", level="ERROR")
    exit()

columns_to_include = [
    "own_attack", "opponent_defense", "opponent_xGA_rolling_4", 
    "opponent_deep_allowed_rolling_4", "opponent_deep_allowed_rolling_16", 
    "team_xG_rolling_4", "team_xG_rolling_16", 
    "was_home", "event", "short_name_h", "short_name_a", "opponent_team"
]

prediction_data = []

for _, fixture in fixtures.iterrows():
    for team_column, was_home in [("team_h", True), ("team_a", False)]:
        team_id = fixture[team_column]
        opponent_column = "team_h" if team_column == "team_a" else "team_a"
        opponent_id = fixture[opponent_column]
        team_fwds = forwards[forwards["own_team"] == team_id].copy()
        
        # Calculating features
        team_fwds["own_attack"] = fixture[f"strength_attack_{'home' if was_home else 'away'}"]
        team_fwds["opponent_defense"] = fixture[f"strength_defence_{'home' if not was_home else 'away'}"]
        team_fwds["opponent_xGA_rolling_4"] = fixture[f"{opponent_column}_xGA_rolling_4"]
        team_fwds["opponent_deep_allowed_rolling_4"] = fixture[f"{opponent_column}_deep_allowed_rolling_4"]
        team_fwds["opponent_deep_allowed_rolling_16"] = fixture[f"{opponent_column}_deep_allowed_rolling_16"]
        team_fwds["team_xG_rolling_4"] = fixture[f"{team_column}_xG_rolling_4"]
        team_fwds["team_xG_rolling_16"] = fixture[f"{team_column}_xG_rolling_16"]
        team_fwds["was_home"] = was_home
        team_fwds["event"] = fixture["event"]
        team_fwds["short_name_h"] = fixture["short_name_h"]
        team_fwds["short_name_a"] = fixture["short_name_a"]
        team_fwds["opponent_team"] = opponent_id
        
        prediction_data.append(team_fwds)

if prediction_data:
    prediction_df = pd.concat(prediction_data, ignore_index=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    save_csv(prediction_df, output_file)
    log(f"Forward predictions saved to: {output_file}")
else:
    log("No prediction data generated. Exiting.", level="WARNING")


# In[ ]:




