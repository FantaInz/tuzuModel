#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from helpers import log, check_file_exists, load_csv, save_csv, remove_file_or_dir, calculate_season_average_until_gw


# In[2]:


# This notebook lets us get csv with the data we want to predict
data_directory = "Fantasy-Premier-League/data/2024-25"
fixtures_file = os.path.join(data_directory, "fixtures.csv")
teams_file = os.path.join(data_directory, "teams.csv")

fixtures = load_csv(fixtures_file)
teams = load_csv(teams_file)

if fixtures is None or teams is None:
    log("Failed to load required CSV files. Exiting.", level="ERROR")
    exit()

unfinished_fixtures = fixtures[~fixtures["finished"]]
next_6_gameweeks = unfinished_fixtures["event"].dropna().unique()[:6]
filtered_fixtures = unfinished_fixtures[unfinished_fixtures["event"].isin(next_6_gameweeks)]

team_columns = [
    "id", "short_name", "strength_attack_home", "strength_attack_away",
    "strength_defence_home", "strength_defence_away"
]
team_data = teams[team_columns]

filtered_fixtures = filtered_fixtures.merge(
    team_data,
    left_on="team_a",
    right_on="id",
    how="left"
).rename(columns={
    "short_name": "short_name_a",
    "strength_attack_away": "strength_attack_a",
    "strength_defence_away": "strength_defense_a"
}).drop(columns=["strength_attack_home", "strength_defence_home"])

filtered_fixtures = filtered_fixtures.merge(
    team_data,
    left_on="team_h",
    right_on="id",
    how="left",
    suffixes=("", "_home")
).rename(columns={
    "short_name": "short_name_h",
    "strength_attack_home": "strength_attack_h",
    "strength_defence_home": "strength_defense_h"
})

columns_to_keep = [
    "team_a", "team_h", "strength_attack_h", "strength_attack_a",
    "strength_defense_h", "strength_defense_a", "short_name_h", 
    "short_name_a", "event"
]
final_data = filtered_fixtures[columns_to_keep].rename(columns={"event": "gameweek"})

output_file = os.path.join(data_directory, "processed_data", "filtered_fixtures.csv")
save_csv(final_data, output_file)


# In[3]:


# Once we have that we can make players_with_clubs.csv where we will add current clubs of the
# players and their values
data_directory = "Fantasy-Premier-League/data/2024-25/"
processed_players_file = os.path.join(data_directory, "processed_data", "processed_players.csv")
players_raw_file = os.path.join(data_directory, "players_raw.csv")

processed_players = load_csv(processed_players_file)
players_raw = load_csv(players_raw_file)

if processed_players is None or players_raw is None:
    log("Failed to load player data. Exiting.", level="ERROR")
    exit()

merged_data = pd.merge(
    processed_players,
    players_raw[['id', 'now_cost', 'team', 'penalties_order']],
    on='id',
    how='left'
).rename(columns={'now_cost': 'value', 'team': 'team_id'})

output_file = os.path.join(data_directory, "processed_data", "players_with_clubs.csv")
save_csv(merged_data, output_file)


# In[4]:


# Let's split the data into positions now because it will make using different models easier for us later

data_directory = "Fantasy-Premier-League/data/2024-25/"
merged_file = os.path.join(data_directory, "processed_data", "players_with_clubs.csv")
merged_data = load_csv(merged_file)

if merged_data is None:
    log("Failed to load merged player data. Exiting.", level="ERROR")
    exit()

position_folders = ["GK", "DEF", "MID", "FWD"]

for position in position_folders:
    position_data = merged_data[merged_data['position'] == position]
    folder_path = os.path.join(data_directory, "processed_data", position)
    os.makedirs(folder_path, exist_ok=True)

    output_file = os.path.join(folder_path, f"{position}_with_clubs.csv")
    save_csv(position_data, output_file)


# In[5]:


def add_features(data_directory, positions):
    """
    Add features required for predictions to player data for each position.
    """
    for position in positions:
        folder_path = os.path.join(data_directory, "processed_data", position)
        position_file = os.path.join(folder_path, f"{position}_with_clubs.csv")

        # Check if position file exists
        if not check_file_exists(position_file):
            log(f"Position file {position_file} does not exist. Skipping.", level="WARNING")
            continue

        # Load the position-specific data
        players_with_clubs = load_csv(position_file)
        if players_with_clubs is None:
            log(f"Failed to load {position_file}. Skipping.", level="WARNING")
            continue

        updated_data = []

        # Process each player's data
        for _, player in players_with_clubs.iterrows():
            player_id = player["id"]
            player_folder = find_player_folder(data_directory, player_id)

            if not player_folder:
                log(f"Folder for player ID {player_id} not found. Skipping.", level="WARNING")
                continue

            gw_data = load_gw_data(player_folder)
            gw_data = gw_data.sort_values(by=["gameweek"]).reset_index(drop=True)
            if gw_data is None or gw_data.empty:
                log(f"GW file not found or empty for player ID {player_id}. Skipping.", level="WARNING")
                continue
            general_features = calculate_general_features(gw_data)
            if position == "GK":
                position_features = calculate_gk_features(gw_data)
            else:
                position_features = calculate_nongk_features(gw_data)

            updated_player = {**player.to_dict(), **position_features, **general_features}
            updated_data.append(updated_player)

        # Save the updated data
        save_updated_data(folder_path, position, updated_data)


def calculate_gk_features(gw_data):
    """
    Calculate features required for goalkeeper prediction.
    """
    rolling_periods = [3, 5, 8, 12]
    required_columns = ["saves", "opponent_xg", "total_points", "minutes", "clean_sheets", "opponent_deep"]

    # Ensure required columns are present
    for column in required_columns:
        if column not in gw_data.columns:
            gw_data[column] = 0  # Initialize missing columns with 0

    # Fill NaN values in existing columns
    gw_data = gw_data.fillna(0)

    gk_features = {}

    for period in rolling_periods:
        for feature in required_columns:
            if feature == "opponent_xg" or feature == "opponent_deep":
                gk_features[f"rolling_{feature}_{period}"] = (
                    gw_data.groupby("unique_id")[feature]
                    .rolling(window=period, min_periods=1)
                    .mean()
                    .iloc[-1] if not gw_data.empty else 0
                )
            else:
                # Perform rolling calculation and reset index to flatten MultiIndex
                rolling_data = (
                    gw_data[gw_data["minutes"] > 0]  # Filter for non-zero appearances
                    .groupby("unique_id")[feature]
                    .rolling(window=period, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)  # Drop the groupby index
                )
                gk_features[f"rolling_{feature}_{period}"] = (
                    rolling_data.iloc[-1] if not rolling_data.empty else 0
                )

    # gk_features["avg_penalties_saved"] = (
    #     gw_data.loc[gw_data["minutes"] > 0, "penalties_saved"].expanding().mean().iloc[-1]
    #     if not gw_data.loc[gw_data["minutes"] > 0].empty else 0
    # )

    # gk_features["avg_yellow_cards"] = (
    #     gw_data.loc[gw_data["minutes"] > 0, "yellow_cards"].expanding().mean().iloc[-1]
    #     if not gw_data.loc[gw_data["minutes"] > 0].empty else 0
    # )

    return gk_features

def calculate_nongk_features(gw_data):
    """
    Calculate features required for defender prediction.
    """
    rolling_periods = [3, 5, 8, 12]
    required_columns = [
        "opponent_xg", "total_points", "shots", 
        "expected_assists", "expected_goals", "ict_index", "influence", 
        "threat", "yellow_cards", "opponent_deep", "team_deep", "bps", "adjusted_xg", "key_passes"
    ]

    for column in required_columns:
        if column not in gw_data.columns:
            gw_data[column] = 0 

    gw_data = gw_data.fillna(0)
    plr_features = {}

    for period in rolling_periods:
        for feature in required_columns:
            if feature == "opponent_xg" or feature == "opponent_deep" or feature == "team_deep":
                plr_features[f"rolling_{feature}_{period}"] = (
                    gw_data.groupby("unique_id")[feature]
                    .rolling(window=period, min_periods=1)
                    .mean()
                    .iloc[-1] if not gw_data.empty else 0
                )
            else:
                rolling_data = (
                    gw_data[gw_data["minutes"] > 0]  # Filter for non-zero appearances
                    .groupby("unique_id")[feature]
                    .rolling(window=period, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)  # Drop the groupby index
                )
                plr_features[f"rolling_{feature}_{period}"] = (
                    rolling_data.iloc[-1] if not rolling_data.empty else 0
                )

    # Calculate average yellow cards (only when minutes > 0)
    plr_features["avg_yellow_cards"] = (
        gw_data.loc[gw_data["minutes"] > 0, "yellow_cards"].expanding().mean().iloc[-1]
        if not gw_data.loc[gw_data["minutes"] > 0].empty else 0
    )

    plr_features["average_season_goals"] = (
        gw_data[gw_data["minutes"] > 0]
        .groupby(["unique_id", "season"])["goals_scored"]
        .expanding()
        .mean()
        .iloc[-1] if not gw_data[gw_data["minutes"] > 0].empty else 0
    )

    plr_features["average_season_assists"] = (
        gw_data[gw_data["minutes"] > 0]
        .groupby(["unique_id", "season"])["assists"]
        .expanding()
        .mean()
        .iloc[-1] if not gw_data[gw_data["minutes"] > 0].empty else 0
    )


    return plr_features


def calculate_general_features(gw_data):
    """
    Calculate features required for non-goalkeeper players.
    """
    # Ensure pred_minutes is calculated
    if "pred_minutes" not in gw_data.columns:
        gw_data["pred_minutes"] = gw_data["minutes"].rolling(window=5, min_periods=1).mean()

    general_features = {}
    general_features["pred_minutes"] = gw_data["pred_minutes"].iloc[-1] if not gw_data.empty else 0
    return general_features

def find_player_folder(data_directory, player_id):
    """
    Locate the folder for a specific player by ID.
    """
    players_dir = os.path.join(data_directory, "players")
    return next(
        (os.path.join(players_dir, folder) for folder in os.listdir(players_dir)
         if folder.endswith(f"_{player_id}")),
        None
    )

def load_gw_data(player_folder):
    """
    Load the player's gameweek data (gw.csv) from their folder.
    """
    gw_file = os.path.join(player_folder, "gw.csv")
    if not check_file_exists(gw_file, log_missing=True):
        return None
    return load_csv(gw_file)

def save_updated_data(folder_path, position, updated_data):
    """
    Save the updated data for a position to a CSV file.
    """
    if not updated_data:
        log(f"No data to save for position {position}. Skipping.", level="INFO")
        return

    output_file = os.path.join(folder_path, f"{position}_with_features.csv")
    save_csv(pd.DataFrame(updated_data), output_file)
    log(f"Updated data saved to {output_file}.", level="INFO")

data_directory = "Fantasy-Premier-League/data/2024-25"
positions = ["GK", "DEF", "MID", "FWD"]
add_features(data_directory, positions)


# In[6]:


# def calculate_xg_prediction_features(gw_data):
#     """
#     Calculate features required for xG prediction from gameweek data.
#     """
#     # Calculate pred_minutes
#     gw_data["pred_minutes"] = gw_data["minutes"].rolling(window=5, min_periods=1).apply(
#         lambda x: x[x != 0].iloc[-4:].mean() if len(x[x != 0]) > 0 else 0, raw=False
#     )
    
#     cumulative_npxg = gw_data["npxG"].sum()
#     cumulative_npg = gw_data["npg"].sum()

#     # Create a filtered DataFrame for calculations requiring minutes >= 60
#     gw_data_filtered = gw_data.loc[gw_data["minutes"] >= 60].copy()
#     if gw_data_filtered.empty:  # Early exit if filtered data is empty
#         return {
#             "pred_minutes": gw_data["pred_minutes"].iloc[-1] if not gw_data.empty else 0,
#             "rolling_adjxg_5": 0,
#             "rolling_shots_5": 0,
#             "season_mean_bonus": 0,
#             "average_adjxg_season": 0,
#             "average_shots_season": 0,
#             "cumulative_npxg": cumulative_npxg,
#             "cumulative_npg": cumulative_npg,
#         }
#     gw_data_filtered["rolling_adjxg_5"] = gw_data_filtered["adjusted_xg"].rolling(window=5, min_periods=1).mean()
#     gw_data_filtered["rolling_shots_5"] = gw_data_filtered["shots"].rolling(window=5, min_periods=1).mean()
#     gw_data_filtered["season_mean_bonus"] = gw_data_filtered["bonus"].expanding(min_periods=1).mean()
#     gw_data_filtered["average_adjxg_season"] = gw_data_filtered["adjusted_xg"].expanding(min_periods=1).mean()
#     gw_data_filtered["average_shots_season"] = gw_data_filtered["shots"].expanding(min_periods=1).mean()

#     return {
#         "pred_minutes": gw_data["pred_minutes"].iloc[-1] if not gw_data.empty else 0,
#         "rolling_adjxg_5": gw_data_filtered["rolling_adjxg_5"].iloc[-1] if not gw_data_filtered.empty else 0,
#         "rolling_shots_5": gw_data_filtered["rolling_shots_5"].iloc[-1] if not gw_data_filtered.empty else 0,
#         "season_mean_bonus": gw_data_filtered["season_mean_bonus"].iloc[-1] if not gw_data_filtered.empty else 0,
#         "average_adjxg_season": gw_data_filtered["average_adjxg_season"].iloc[-1] if not gw_data_filtered.empty else 0,
#         "average_shots_season": gw_data_filtered["average_shots_season"].iloc[-1] if not gw_data_filtered.empty else 0,
#         "cumulative_npxg": cumulative_npxg,
#         "cumulative_npg": cumulative_npg
#     }

# def calculate_xa_prediction_features(gw_data):
#     """
#     Calculate assist-related features.
#     """
#     cumulative_xa = gw_data["expected_assists"].sum()
#     cumulative_assists = gw_data["assists"].sum()
    
#     gw_data_filtered = gw_data.loc[gw_data["minutes"] >= 60].copy()

#     if gw_data_filtered.empty:  # Early exit if filtered data is empty
#         return {
#             "rolling_xa_5": 0,
#             "rolling_key_passes_5": 0,
#             "average_xa_per_game": 0,
#             "average_key_passes_per_game": 0,
#             "cumulative_xa": cumulative_xa,
#             "cumulative_assists": cumulative_assists,
#         }

    
#     gw_data_filtered["rolling_xa_5"] = gw_data_filtered["expected_assists"].rolling(window=5, min_periods=1).mean()
#     gw_data_filtered["rolling_key_passes_5"] = gw_data_filtered["key_passes"].rolling(window=5, min_periods=1).mean()
#     gw_data_filtered["average_xa_season"] = gw_data_filtered["expected_assists"].expanding(min_periods=1).mean()
#     gw_data_filtered["average_key_passes_season"] = gw_data_filtered["key_passes"].expanding(min_periods=1).mean()
    
#     return {
#         "rolling_xa_5": gw_data_filtered["rolling_xa_5"].iloc[-1] if not gw_data_filtered.empty else 0,
#         "rolling_key_passes_5": gw_data_filtered["rolling_key_passes_5"].iloc[-1] if not gw_data_filtered.empty else 0,
#         "average_xa_season": gw_data_filtered["average_xa_season"].iloc[-1] if not gw_data_filtered.empty else 0,
#         "average_key_passes_season": gw_data_filtered["average_key_passes_season"].iloc[-1] if not gw_data_filtered.empty else 0,
#         "cumulative_xa": cumulative_xa,
#         "cumulative_assists": cumulative_assists
#     }


# In[7]:


def merge_with_fixtures(data_directory, fixtures_file, positions):
    """
    Merge player data with fixtures for each position and save the final merged data.
    """
    fixtures = pd.read_csv(fixtures_file)
    position_mapping = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}

    required_prefixes = ["rolling_", "average_", "cumulative_", "avg_"]
    required_columns = [
        "id", "first_name", "second_name", "own_team", "opponent_team",
        "own_short_name", "opponent_short_name", "own_attack", "opponent_attack",
        "own_defense", "opponent_defense", "was_home", "season", "POS",
        "gameweek", "value", "pred_minutes", 'penalties_order'
    ]

    for position in positions:
        folder_path = os.path.join(data_directory, "processed_data", position)
        position_file = os.path.join(folder_path, f"{position}_with_features.csv")

        if not os.path.exists(position_file):
            print(f"{position_file} not found. Skipping.")
            continue

        # Load player data
        players = pd.read_csv(position_file)

        # Process and merge data for home and away fixtures
        home_merge = process_fixtures_merge(players, fixtures, "team_h", "team_a", was_home=1)
        away_merge = process_fixtures_merge(players, fixtures, "team_a", "team_h", was_home=0)

        # Concatenate home and away data
        merged = pd.concat([home_merge, away_merge], ignore_index=True)

        # Add additional columns
        merged["season"] = 24  # Fixed season value
        merged["POS"] = position_mapping[position]  # Position mapping

        all_columns = merged.columns
        dynamic_columns_to_keep = [
            col for col in all_columns
            if col in required_columns or any(col.startswith(prefix) for prefix in required_prefixes)
        ]

        final_data = merged[dynamic_columns_to_keep]

        # Save the merged data
        output_file = os.path.join(folder_path, f"{position}_final.csv")
        final_data.to_csv(output_file, index=False)
        log(f"Saved merged data for {position} to {output_file}", level="INFO")

def process_fixtures_merge(players, fixtures, own_key, opponent_key, was_home):
    """
    Helper function to process the merge between players and fixtures.
    """
    merged = players.merge(
        fixtures,
        left_on="team_id",
        right_on=own_key,
        how="inner"
    )
    merged["was_home"] = was_home

    # Rename columns for consistency
    column_mapping = {
        own_key: "own_team",
        opponent_key: "opponent_team",
        f"short_name_{own_key[-1]}": "own_short_name",
        f"short_name_{opponent_key[-1]}": "opponent_short_name",
        f"strength_attack_{own_key[-1]}": "own_attack",
        f"strength_defense_{own_key[-1]}": "own_defense",
        f"strength_attack_{opponent_key[-1]}": "opponent_attack",
        f"strength_defense_{opponent_key[-1]}": "opponent_defense"
    }
    merged.rename(columns=column_mapping, inplace=True)
    return merged


# Main Execution
data_directory = "Fantasy-Premier-League/data/2024-25/"
fixtures_file = os.path.join(data_directory, "processed_data", "filtered_fixtures.csv")
positions = ["GK", "DEF", "MID", "FWD"]
merge_with_fixtures(data_directory, fixtures_file, positions)


# In[ ]:




