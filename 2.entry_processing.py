#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
import re

import pandas as pd
from rapidfuzz import fuzz
from datetime import datetime, timezone

from helpers import log, check_file_exists, load_csv, save_csv, remove_file_or_dir, filter_directory


# In[2]:


def check_for_duplicate_players_by_name(data_directory, seasons):
    """
    Check for duplicate player names across multiple seasons.

    Args:
        data_directory (str): Path to the base data directory.
        seasons (list): List of season names to check (e.g., ["2020-21", "2021-22"]).

    Returns:
        dict: A dictionary where keys are season names and values are DataFrames
              of duplicate player names with ids.
    """
    duplicate_players = {}
    
    for season in seasons:
        season_path = os.path.join(data_directory, season)
        player_idlist_path = os.path.join(season_path, "player_idlist.csv")
        
        if not check_file_exists(player_idlist_path):
            log(f"player_idlist.csv not found for season {season}. Skipping.", level="WARNING")
            continue

        player_idlist = load_csv(player_idlist_path)
        if player_idlist is None:
            continue

        duplicates = player_idlist.duplicated(subset=['first_name', 'second_name'], keep=False)
        duplicate_names = player_idlist.loc[duplicates, ['id', 'first_name', 'second_name']].drop_duplicates()
        
        if not duplicate_names.empty:
            duplicate_players[season] = duplicate_names
            log(f"Found duplicate names in season {season}:")
            for _, row in duplicate_names.iterrows():
                log(f"    ID: {row['id']} - {row['first_name']} {row['second_name']}", level="INFO")
        else:
            log(f"No duplicate names found in season {season}.")
            
    return duplicate_players

data_directory = "Fantasy-Premier-League/data"
seasons = ["2022-23", "2023-24", "2024-25"]

duplicates_by_name = check_for_duplicate_players_by_name(data_directory, seasons)


# In[3]:


# Here we get rid of the Ben Davies that's no longer in the game
def remove_players_and_folders(data_directory, players_to_remove):
    """
    Remove players from player_idlist.csv and their associated folders.

    Args:
        data_directory (str): Base directory containing the season data.
        players_to_remove (list): List of dictionaries with player details to remove.

    Example Player Format:
        {"season": "2020-21", "first_name": "Ben", "second_name": "Davies", "id": 653}
    """
    for player in players_to_remove:
            season = player["season"]
            player_id = player["id"]
            first_name = player["first_name"]
            second_name = player["second_name"]

            season_path = os.path.join(data_directory, season)
            player_idlist_path = os.path.join(season_path, "player_idlist.csv")
            players_folder_path = os.path.join(season_path, "players")
            player_folder_name = f"{first_name}_{second_name}_{player_id}"
            player_folder_path = os.path.join(players_folder_path, player_folder_name)

            if not check_file_exists(player_idlist_path):
                log(f"player_idlist.csv not found for season {season}. Skipping.", level="WARNING")
                continue
                
            player_idlist = load_csv(player_idlist_path)
            if player_idlist is None:
                continue

            matching_rows = player_idlist[
                (player_idlist["id"] == player_id) &
                (player_idlist["first_name"] == first_name) &
                (player_idlist["second_name"] == second_name)
            ]

            if not matching_rows.empty:
                player_idlist.drop(matching_rows.index, inplace=True)
                save_csv(player_idlist, player_idlist_path)
                log(f"Removed {first_name} {second_name} (ID: {player_id}) from {player_idlist_path}.")
            else:
                log(f"Player {first_name} {second_name} (ID: {player_id}) not found in {player_idlist_path}. Skipping.", level="WARNING")

            remove_file_or_dir(player_folder_path)

data_directory = "Fantasy-Premier-League/data"
players_to_remove = [
        {"season": "2022-23", "first_name": "Ben", "second_name": "Davies", "id": 499},
]

remove_players_and_folders(data_directory, players_to_remove)


# In[4]:


# First let's get each player position and id for the season
def process_season_data(data_directory, seasons, output_folder_name="processed_data"):
    """
    Process player data for each season by merging player ID lists with cleaned data.

    Args:
        data_directory (str): Base directory containing the season data.
        seasons (list): List of season names to process (e.g., ["2020-21", "2021-22"]).
        output_folder_name (str): Name of the folder to store the processed files.
    Returns:
        None
    """
    for season in seasons:
        season_path = os.path.join(data_directory, season)
        player_idlist_path = os.path.join(season_path, "player_idlist.csv")
        cleaned_players_path = os.path.join(season_path, "cleaned_players.csv")

        output_folder_path = os.path.join(season_path, output_folder_name)
        os.makedirs(output_folder_path, exist_ok=True)

        output_file_path = os.path.join(output_folder_path, "processed_players.csv")
        
        if not os.path.isdir(season_path):
            log(f"Season directory {season_path} does not exist. Skipping.", level="WARNING")
            continue

        if not (check_file_exists(player_idlist_path) and check_file_exists(cleaned_players_path)):
            log(f"Missing required files in {season_path}: player_idlist.csv or cleaned_players.csv", level="WARNING")
            continue
        
        player_idlist_df = load_csv(player_idlist_path)
        cleaned_players_df = load_csv(cleaned_players_path)

        if player_idlist_df is None or cleaned_players_df is None:
            continue

        try:
            # Merge the datasets
            merged_df = pd.merge(
                player_idlist_df[['id', 'first_name', 'second_name']],
                cleaned_players_df[['first_name', 'second_name', 'element_type']],
                on=['first_name', 'second_name'],
                how='inner'
            )

            merged_df.rename(columns={'element_type': 'position'}, inplace=True)

            save_csv(merged_df, output_file_path)
            log(f"Processed and saved: {output_file_path}")
        except Exception as e:
            log(f"Error processing season {season}: {e}", level="ERROR")

data_directory = "Fantasy-Premier-League/data"
seasons = ["2024-25", "2023-24", "2022-23"]

process_season_data(data_directory, seasons)


# In[5]:


# This is used to create a new file with player ids across seasons
def merge_player_ids(data_directory, seasons, output_file="master_player_list.csv"):
    """
    Merge player IDs across seasons into a consolidated master list.

    Args:
        data_directory (str): Base directory containing the season data.
        seasons (list): List of season names to process (e.g., ["2020-21", "2021-22"]).
        output_file (str): Name of the output file for the master player list.

    Returns:
        pd.DataFrame: The consolidated DataFrame of player IDs.
    """
    player_data = {}
    next_unique_id = 1
    main_season = "2024-25"
    main_season_path = os.path.join(data_directory, main_season, "processed_data", "processed_players.csv")

    # Process the main season first. We want to keep these id's since we will use same ones for predictions. 
    if check_file_exists(main_season_path):
        main_processed_df = load_csv(main_season_path)
        if main_processed_df is not None:
            for _, row in main_processed_df.iterrows():
                full_name = f"{row['first_name']} {row['second_name']}"
                if full_name not in player_data:
                    player_data[full_name] = {
                        "First_Name": row['first_name'],
                        "Last_Name": row['second_name'],
                        "Unique_ID": row['id'],
                        "24_id": row['id']
                    }
                next_unique_id = max(next_unique_id, row['id'] + 1)
    else:
        log(f"Missing processed_players.csv for main season: {main_season}", level="WARNING")

    # Process all other seasons
    for season in seasons:
        if season == main_season:
            continue

        season_path = os.path.join(data_directory, season, "processed_data", "processed_players.csv")
        if check_file_exists(season_path):
            processed_df = load_csv(season_path)
            if processed_df is not None:
                season_short = season[:4][-2:]
                for _, row in processed_df.iterrows():
                    full_name = f"{row['first_name']} {row['second_name']}"
                    if full_name not in player_data:
                        player_data[full_name] = {
                            "First_Name": row['first_name'],
                            "Last_Name": row['second_name'],
                            "Unique_ID": next_unique_id
                        }
                        next_unique_id += 1
                    # Add season-specific ID
                    player_data[full_name][f"{season_short}_id"] = row['id']
        else:
            log(f"Missing processed_players.csv for season: {season}", level="WARNING")

    # Convert player data to DataFrame
    consolidated_df = pd.DataFrame.from_dict(player_data, orient='index').reset_index(drop=True)

    # Ensure all ID columns are integers
    id_columns = [col for col in consolidated_df.columns if col.endswith("_id")]
    consolidated_df[id_columns] = consolidated_df[id_columns].fillna(-1).astype(int)

    season_columns = sorted(
        [col for col in consolidated_df.columns if col.endswith("_id")],
        reverse=True
    )
    # Reorder columns
    other_columns = [col for col in consolidated_df.columns if col not in season_columns]
    consolidated_df = consolidated_df[other_columns + season_columns]
    
    # Save the consolidated DataFrame
    output_path = os.path.join(data_directory, output_file)
    save_csv(consolidated_df, output_path)
    log(f"Consolidated player data saved to {output_path}")

    return consolidated_df


data_directory = "Fantasy-Premier-League/data"
seasons = ["2022-23", "2023-24", "2024-25"]

master_player_list = merge_player_ids(data_directory, seasons, output_file="master_player_v1.csv")


# In[6]:


# # Script below was used to get duplicate candidates (players can have name changes across seasons). After finding them I checked it manually

# consolidated_file = "master_player_v1.csv"
# consolidated_path = os.path.join(data_directory, consolidated_file)
# consolidated_df = pd.read_csv(consolidated_path)

# SIMILARITY_THRESHOLD = 80
# FIRST_NAME_SIMILARITY_THRESHOLD = 40

# def is_pair(player_a, player_b, id_columns):
#     # Check for ID overlaps across seasons
#     id_overlap = any(
#         player_a[col] != -1 and player_b[col] != -1
#         for col in id_columns
#     )
#     if id_overlap:
#         return False
    
#     # Now we check the names similarity
#     first_name_similarity = fuzz.partial_ratio(player_a["First_Name"], player_b["First_Name"])
#     if first_name_similarity <= FIRST_NAME_SIMILARITY_THRESHOLD:
#         return False
    
#     # Check if one last name is contained in the other
#     if player_a["Last_Name"] in player_b["Last_Name"] or player_b["Last_Name"] in player_a["Last_Name"]:
#         return True

#     # Calculate string similarity only if containment check fails
#     similarity = fuzz.partial_ratio(player_a["Last_Name"], player_b["Last_Name"])
#     return similarity >= SIMILARITY_THRESHOLD

# # Identify all ID columns
# id_columns = [col for col in consolidated_df.columns if col.endswith("_id")]
# pairs = []
# for i, player_a in consolidated_df.iterrows():
#     for j, player_b in consolidated_df.iterrows():
#         if i >= j:  # Avoid duplicate and self-pairs
#             continue
#         if is_pair(player_a, player_b, id_columns):
#             pairs.append({
#                 "Player_A_First_Name": player_a["First_Name"],
#                 "Player_A_Last_Name": player_a["Last_Name"],
#                 "Player_B_First_Name": player_b["First_Name"],
#                 "Player_B_Last_Name": player_b["Last_Name"],
#             })

# # Convert pairs to a DataFrame
# created_csv_dir = "created_csv"
# pairs_df = pd.DataFrame(pairs)
# if not pairs_df.empty:
#     output_path = os.path.join(created_csv_dir, "players_with_potential_renames.csv")
#     pairs_df.to_csv(output_path, index=False)
#     print(f"Potential renamed players saved to {output_path}")
# else:
#     print("No potential renamed players found.")


# In[7]:


# After manual checking the candidates now we have to update master_player_list.csv 
# (new file will me master_player_verified).
# We might get some warnings here but we can safely ignore them (one player can have multiple rows and deleting one of them causes errors
# but we still merge the data to the one with lowest ID.
def update_master_player_list(data_directory, master_player_file, verified_renames_file, output_file="master_player_v2.csv"):
    """
    Update the master player list by merging duplicate player data (same players with different names across seasons)
    and saving the verified list.

    Args:
        data_directory (str): Base directory containing the data.
        master_player_file (str): Name of the master player list file (e.g., "master_player_list.csv").
        verified_renames_file (str): Path to the verified renames CSV file.
        output_file (str): Name of the output file for the updated master list.

    Returns:
        None
    """
    master_player_path = os.path.join(data_directory, master_player_file)
    output_path = os.path.join(data_directory, output_file)

    # Load the verified renames and master player list
    verified_renames = load_csv(verified_renames_file)
    master_player_list = load_csv(master_player_path)

    if verified_renames is None or master_player_list is None:
        log("Failed to load required files. Exiting update process.", level="ERROR")
        return

    # Iterate through verified renames and process each pair. The way finding duplicate candidates work player_b will have greater
    # ID than player_a
    for _, row in verified_renames.iterrows():
        player_a_first = row["Player_A_First_Name"]
        player_a_last = row["Player_A_Last_Name"]
        player_b_first = row["Player_B_First_Name"]
        player_b_last = row["Player_B_Last_Name"]

        # Find rows for Player A and Player B
        player_a_row = master_player_list[
            (master_player_list["First_Name"] == player_a_first) &
            (master_player_list["Last_Name"] == player_a_last)
        ]

        player_b_row = master_player_list[
            (master_player_list["First_Name"] == player_b_first) &
            (master_player_list["Last_Name"] == player_b_last)
        ]

        # Check for missing players
        if player_a_row.empty or player_b_row.empty:
            log(f"Warning: Missing player in master_player_list. Player A: {player_a_first} {player_a_last}, "
                f"Player B: {player_b_first} {player_b_last}", level="DEBUG")
            continue

        player_a_id = player_a_row["Unique_ID"].iloc[0]
        player_b_id = player_b_row["Unique_ID"].iloc[0]

        # Update the smaller ID row with non -1 season data from the larger ID row
        for season in ["24_id", "23_id", "22_id"]:
            if player_b_row[season].iloc[0] != -1:
                master_player_list.loc[master_player_list["Unique_ID"] == player_a_id, season] = player_b_row[season].iloc[0]

        # Remove Player B
        master_player_list = master_player_list[master_player_list["Unique_ID"] != player_b_id]
        log(f"Merged Player B (ID: {player_b_id}) into Player A (ID: {player_a_id}). Removed Player B.", level="DEBUG")

    # Save the updated master player list
    save_csv(master_player_list, output_path)
    log(f"Updated master player list saved to {output_path}")

data_directory = "Fantasy-Premier-League/data"
master_player_file = "master_player_v1.csv"
verified_renames_file = "created_csv/verified_renames_2022.csv"

update_master_player_list(data_directory, master_player_file, verified_renames_file)


# In[8]:


# With the new list we can move on to renaming player folders to have the new ids and their positions
# We also update the processed_players

def update_processed_players_and_rename_folders(data_directory, seasons, master_player_file):
    """
    Update processed players' IDs and names, and rename player folders.

    Args:
        data_directory (str): Base directory containing the data.
        seasons (list): List of seasons to process (e.g., ["2020-21", "2021-22"]).
        master_player_file (str): Name of the master player list file (e.g., "master_player_verified.csv").

    Returns:
        None
    """
    master_player_path = os.path.join(data_directory, master_player_file)
    master_player_verified = load_csv(master_player_path)

    if master_player_verified is None:
        log(f"Failed to load {master_player_file}. Exiting process.", level="ERROR")
        return

    for season in seasons:
        season_path = os.path.join(data_directory, season)
        players_folder_path = os.path.join(season_path, "players")
        processed_file = os.path.join(season_path, "processed_data", "processed_players.csv")
        verified_file = os.path.join(season_path, "processed_data", "processed_players_v2.csv")

        # Special case for "2024-25". We just copy here.
        if season == "2024-25":
            log(f"Processing special case: {season}")
            if check_file_exists(processed_file):
                try:
                    shutil.copy(processed_file, verified_file)
                    log(f"Copied {processed_file} to {verified_file}")
                except Exception as e:
                    log(f"Error copying {processed_file} to {verified_file}: {e}", level="ERROR")
            else:
                log(f"Processed players file not found for season: {season}", level="WARNING")
            continue

        # Process other seasons
        season_short = season[:4][-2:]  # Extract short season ID (e.g., "21" for "2021-22")
        season_id_column = f"{season_short}_id"

        if not check_file_exists(processed_file):
            log(f"Processed players file not found for season: {season}", level="WARNING")
            continue

        processed_players = load_csv(processed_file)
        if processed_players is None:
            continue

        # Rename player folders and update processed players
        if check_file_exists(players_folder_path):
            player_folders = [
                f for f in os.listdir(players_folder_path)
                if os.path.isdir(os.path.join(players_folder_path, f))
            ]
        else:
            log(f"Players folder not found for season {season}. Skipping folder renames.", level="WARNING")
            player_folders = []

        for index, row in processed_players.iterrows():
            current_id = row["id"]
            matching_row = master_player_verified[master_player_verified[season_id_column] == current_id]

            if matching_row.empty:
                log(f"No matching row in master list for player ID {current_id} in season {season}.", level="WARNING")
                continue

            # Get new player data
            new_data = matching_row.iloc[0]
            new_unique_id = new_data["Unique_ID"]
            new_first_name = new_data["First_Name"]
            new_last_name = new_data["Last_Name"]

            # Update processed_players DataFrame
            processed_players.at[index, "id"] = new_unique_id
            processed_players.at[index, "first_name"] = new_first_name
            processed_players.at[index, "second_name"] = new_last_name

            # Rename player folder
            matching_folder = next(
                (f for f in player_folders if f.endswith(f"_{current_id}")), None
            )
            if matching_folder:
                old_path = os.path.join(players_folder_path, matching_folder)
                new_folder_name = f"{new_first_name}_{new_last_name}_{new_unique_id}"
                new_path = os.path.join(players_folder_path, new_folder_name)

                if not check_file_exists(new_path):  # Avoid overwriting existing folders
                    try:
                        shutil.move(old_path, new_path)
                        log(f"Renamed folder {matching_folder} to {new_folder_name}", level="DEBUG")
                        player_folders.remove(matching_folder)  # Remove processed folder
                    except Exception as e:
                        log(f"Error renaming {old_path} to {new_path}: {e}", level="ERROR")
                else:
                    log(f"Destination folder {new_path} already exists. Skipping rename.", level="DEBUG")

        save_csv(processed_players, verified_file)
        log(f"Updated processed players saved for season {season} to {verified_file}")

data_directory = "Fantasy-Premier-League/data"
seasons = ["2022-23", "2023-24", "2024-25"]
master_player_file = "master_player_v2.csv"
update_processed_players_and_rename_folders(data_directory, seasons, master_player_file)


# In[9]:


# We have to rename some files because of their wrong names in the understat folder
def rename_files(directory):
    """
    Renames files in the specified directory by replacing '&#039;' with an apostrophe (').

    Parameters:
    - directory (str): The path to the directory containing the files to rename.
    """
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the filename contains '&#039;'
        if "&#039;" in filename:
            # Replace '&#039;' with apostrophe
            new_filename = filename.replace("&#039;", "'")
            # Define full paths
            src = os.path.join(directory, filename)
            dst = os.path.join(directory, new_filename)
            try:
                os.rename(src, dst)
                log(f"Renamed: '{filename}' -> '{new_filename}'", "DEBUG")
            except Exception as e:
                log(f"Failed to rename '{filename}'. Reason: {e}", "ERROR")

data_dir = "Fantasy-Premier-League/data"
seasons = ["2022-23", "2023-24", "2024-25"]
for season in seasons:
    understat_dir = os.path.join(data_dir, season, "understat")
    
    if os.path.isdir(understat_dir):
        log(f"Processing season '{season}' in '{understat_dir}'", "INFO")
        rename_files(understat_dir)
    else:
        log(f"Understat directory for season '{season}' does not exist at '{understat_dir}'", "ERROR")


# In[10]:


# We have disparity between understat ids and fpl ids. For season 2022-23 we can use the existing file, for the other seasons we will
# have to create our own
data_directory = "Fantasy-Premier-League/data"
# Add unique_id to the 2022-23 mapping. We also need to rename Kante since he is caussing issues
id_dict_path_22 = os.path.join(data_directory, "2022-23", "id_dict.csv")
master_player_path = os.path.join(data_directory, "master_player_v2.csv")
id_dict = load_csv(id_dict_path_22)
id_dict['Understat_Name'] = id_dict['Understat_Name'].replace(
        "N&#039;Golo Kanté", "N'Golo Kanté"
    )
master_player = load_csv(master_player_path)

# Merge id_dict with master_player to add Unique_ID
merged_data = id_dict.merge(master_player[['Unique_ID', '22_id']], 
                                left_on='FPL_ID', 
                                right_on='22_id', 
                                how='left')
merged_data = merged_data.drop(columns=['22_id'])
save_csv(merged_data, id_dict_path_22)


# In[11]:


# This was used to help create the mappings between understat ids and fpl ids.

# Functions to help with creating id maps between fpl and understat for 2022-23 and 2023-24
# data_dir = "Fantasy-Premier-League/data"
# output_dir = "created_csv"
# master_file = os.path.join(data_dir, "master_player_v2.csv")
# id_dict_file = os.path.join(data_dir, "2022-23", "id_dict.csv") 

# master_df = load_csv(master_file)
# master_df['Full_Name'] = master_df['First_Name'].str.strip() + ' ' + master_df['Last_Name'].str.strip()

# id_dict_df = load_csv(id_dict_file)
# unique_to_understat = id_dict_df.set_index('Unique_ID')['Understat_ID'].to_dict()
# master_df['Understat_ID'] = master_df['Unique_ID'].map(unique_to_understat)

# season_mapping = {
#     "2023-24": {"id_col": "23_id", "output_file": "players_2023-24.csv"},
#     "2024-25": {"id_col": "24_id", "output_file": "players_2024-25.csv"}
# }

# for season, info in season_mapping.items():
#     id_col = info["id_col"]
#     output_file = info["output_file"]
#     output_path = os.path.join(output_dir, output_file)
    
#     # Filter players who played in the season
#     season_df = master_df[master_df[id_col] != -1][['Unique_ID', 'Full_Name', id_col]].copy()

#     # Now, extract Understat_IDs from understat folder
#     understat_dir = os.path.join(data_dir, season, "understat")
#     if not os.path.isdir(understat_dir):
#         log(f"Understat directory for season '{season}' not found at '{understat_dir}'. Skipping Understat_ID mapping.", "ERROR")
#         continue

#     # Extract Full_Name and Understat_ID from filenames
#     understat_files = [f for f in os.listdir(understat_dir) if f.endswith('.csv') and not f.startswith("understat")]
#     understat_data = []

#     for file in understat_files:
#         name_part, uid_part = os.path.splitext(file)[0].rsplit('_', 1)
#         full_name = name_part.replace('_', ' ').strip()
#         full_name = full_name.strip()
#         understat_data.append({"Full_Name": full_name, "Understat_ID_filename": uid_part})
#     if not understat_data:
#         log(f"No valid Understat files found for season '{season}'.", "WARNING")
#         continue

#     understat_df = pd.DataFrame(understat_data)

#     # Merge Understat_ID into season_df
#     merged_df = pd.merge(season_df, understat_df, on="Full_Name", how="left")
#     merged_df['Understat_ID'] = merged_df['Understat_ID'].fillna(merged_df['Understat_ID_filename'])
#     merged_df.drop(columns=['Understat_ID_filename'], inplace=True)

#     # Let's try to match on substrings here
#     unmatched = merged_df[merged_df['Understat_ID'].isna()]
#     for idx, row in unmatched.iterrows():
#         master_full_name = row['Full_Name']
#         possible_matches = understat_df[understat_df['Full_Name'].apply(lambda x: x.lower() in master_full_name.lower())]
#         if len(possible_matches) == 1:
#             matched_id = possible_matches['Understat_ID'].values[0]
#             merged_df.at[idx, 'Understat_ID'] = matched_id
#             log(f"Mapped Understat_ID '{matched_id}' to '{master_full_name}' via substring match.", "INFO")
#         elif len(possible_matches) > 1:
#             log(f"Multiple Understat_IDs matched for '{master_full_name}'. Cannot assign.", "INFO")
#         else:
#             log(f"No Understat_ID found for '{master_full_name}'.", "INFO")

#     merged_df['Understat_ID'] = merged_df['Understat_ID'].astype('Int64')
    
#     save_csv(merged_df, output_path)
#     log(f"Updated '{output_file}' with Understat_IDs.", "INFO")

# understat_map_file = os.path.join(output_dir, "understat_map_2023-24.csv")
# understat_map_df = load_csv(understat_map_file)
# players_2024_25_file = os.path.join(output_dir, "players_2024-25.csv")
# players_2024_25_df = load_csv(players_2024_25_file)
# updated_players_2024_25_df = pd.merge(
#     players_2024_25_df,
#     understat_map_df[['Unique_ID', 'Understat_ID']],
#     on='Unique_ID',
#     how='left',
#     suffixes=('', '_map')
# )
# updated_players_2024_25_df['Understat_ID'] = updated_players_2024_25_df['Understat_ID'].fillna(updated_players_2024_25_df['Understat_ID_map'])
# updated_players_2024_25_df.drop(columns=['Understat_ID_map'], inplace=True)
# updated_players_2024_25_df['Understat_ID'] = updated_players_2024_25_df['Understat_ID'].astype('Int64')
# save_csv(updated_players_2024_25_df, players_2024_25_file)


# In[12]:


# Now let's extract useful understat data for each player
data_dir = "Fantasy-Premier-League/data"
created_dir = "created_csv"
players_dir_template = os.path.join(data_dir, "{season}", "players")
understat_dir_template = os.path.join(data_dir, "{season}", "understat")

season_input_map = {
    "2022-23": os.path.join(data_dir, "2022-23", "id_dict.csv"),
    "2023-24": os.path.join(created_dir, "understat_map_2023-24.csv"),
    "2024-25": os.path.join(created_dir, "understat_map_2024-25.csv"),
}

# Main function to update GW files
def update_gw_files(input_file, season):
    mapping = load_csv(input_file)
    if mapping.empty or not {"Understat_ID", "Unique_ID"}.issubset(mapping.columns):
        log(f"Invalid input file: {input_file}", "ERROR")
        return

    # Get all player folders
    players_dir = players_dir_template.format(season=season)
    understat_dir = understat_dir_template.format(season=season)
    player_folders = [
        folder for folder in os.listdir(players_dir) if os.path.isdir(os.path.join(players_dir, folder))
    ]
    understat_files = [
        file for file in os.listdir(understat_dir) if file.endswith(".csv")
    ]

    for _, row in mapping.iterrows():
        understat_id = str(row["Understat_ID"])
        unique_id = str(row["Unique_ID"])

        matching_folder = next((folder for folder in player_folders if folder.endswith(f"_{unique_id}")), None)
        if not matching_folder:
            log(f"Player folder for Unique_ID '{unique_id}' not found in season '{season}'", "DEBUG")
            continue

        # Find the matching Understat file
        matching_understat_file = next((file for file in understat_files if file.endswith(f"_{understat_id}.csv")), None)
        if not matching_understat_file:
            log(f"Understat file for ID '{understat_id}' not found in season '{season}'", "DEBUG")
            continue

        # Construct file paths
        understat_file = os.path.join(understat_dir, matching_understat_file)
        gw_file = os.path.join(players_dir, matching_folder, "gw.csv")

        if not os.path.exists(gw_file):
            log(f"GW file missing for player folder '{matching_folder}' in season '{season}'", "WARNING")
            continue

        understat_data = load_csv(understat_file)
        gw_data = load_csv(gw_file)

        if understat_data.empty or gw_data.empty:
            log(f"Empty data for player {unique_id} or understat ID {understat_id}", "WARNING")
            continue

        # Merge and update GW file
        try:
            understat_data["date"] = pd.to_datetime(understat_data["date"]).dt.date
            gw_data["kickoff_time"] = pd.to_datetime(gw_data["kickoff_time"], utc=True).dt.date 
            merged = pd.merge(
                gw_data,
                understat_data[["date", "shots", "xG", "xA", "key_passes", "npg", "npxG"]],
                left_on="kickoff_time",
                right_on="date",
                how="left"
            )

            merged["shots"] = merged["shots"].fillna(0).astype(int)
            merged["expected_goals"] = merged["xG"].fillna(merged["expected_goals"])
            merged["expected_assists"] = merged["xA"].fillna(merged["expected_assists"])
            merged["key_passes"] = merged["key_passes"].fillna(0).astype(int)
            merged["npg"] = merged["npg"].fillna(0).astype(int)
            merged["npxG"] = merged["npxG"].fillna(0)
            merged.drop(columns=["date", "xG", "xA"], inplace=True)

            save_csv(merged, gw_file)
            log(f"Updated GW file for player folder '{matching_folder}' in season '{season}'", "DEBUG")
        except Exception as e:
            log(f"Error updating GW file for player folder '{matching_folder}': {e}", "ERROR")

# Process seasons
seasons = ["2022-23", "2023-24", "2024-25"]

for season in seasons:
    input_file = season_input_map.get(season)
    if input_file and os.path.exists(input_file):
        log(f"Processing season {season}")
        update_gw_files(input_file, season)
    else:
        log(f"Skipping season {season}: Input file missing", "WARNING")


# In[13]:


# Now we will start manipulating player's data. First let's add to each player row his position
# (1 - GK, 2 - DEF, 3 - MID, 4 - FWD) and add it to the folder's name as well
# Let's also add the Full Name of the player and his unique id.

def update_player_folders_and_gw_data(season_path):
    """
    Update player folders with positions, unique IDs, and full names. 
    Also, update gw.csv files with these attributes.

    Args:
        season_path (str): Path to the season folder.

    Returns:
        None
    """
    position_mapping = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    
    processed_players_path = os.path.join(season_path, "processed_data", "processed_players_v2.csv")
    players_folder_path = os.path.join(season_path, "players")
    
    if not check_file_exists(processed_players_path):
        log(f"{processed_players_path} not found. Skipping season.", level="WARNING")
        return
    
    processed_players = load_csv(processed_players_path)
    if processed_players is None:
        log(f"Failed to load {processed_players_path}. Skipping season.", level="ERROR")
        return
    
    for _, row in processed_players.iterrows():
        player_id = row['id']
        first_name = row['first_name']
        second_name = row['second_name']
        position_str = row['position']
        
        pos_numeric = position_mapping.get(position_str, 0)  # Default to 0 if unknown
        if pos_numeric == 0:
            log(f"Unknown position for player {first_name} {second_name} (ID: {player_id}). Skipping.", level="WARNING")
            continue
        
        old_folder_name = f"{first_name}_{second_name}_{player_id}"
        old_folder_path = os.path.join(players_folder_path, old_folder_name)
        if not check_file_exists(old_folder_path):
            log(f"Folder {old_folder_name} not found. Skipping.", level="WARNING")
            continue

        new_folder_name = f"{position_str}_{old_folder_name}"
        new_folder_path = os.path.join(players_folder_path, new_folder_name)
        
        try:
            if not check_file_exists(new_folder_path):  # Avoid overwriting existing folders
                shutil.move(old_folder_path, new_folder_path)
                log(f"Renamed folder: {old_folder_name} -> {new_folder_name}", level="DEBUG")
            else:
                log(f"Destination folder {new_folder_name} already exists. Skipping rename.", level="WARNING")
        except Exception as e:
            log(f"Error renaming {old_folder_name} to {new_folder_name}: {e}", level="ERROR")
            continue
        
        gw_file_path = os.path.join(new_folder_path, "gw.csv")
        
        if check_file_exists(gw_file_path):
            try:
                gw_data = load_csv(gw_file_path)
                if gw_data is not None:
                    # Update element to unique_id and rename column
                    if 'element' in gw_data.columns:
                        gw_data.rename(columns={'element': 'unique_id'}, inplace=True)
                        gw_data['unique_id'] = player_id
                    
                    # Add POS and Full Name
                    gw_data['POS'] = pos_numeric
                    gw_data['Full Name'] = f"{first_name} {second_name}"
                    
                    # Save updated gw.csv
                    save_csv(gw_data, gw_file_path)
                    log(f"Updated POS and unique_id in {gw_file_path}", level="DEBUG")
            except Exception as e:
                log(f"Error updating {gw_file_path}: {e}", level="ERROR")
        else:
            log(f"{gw_file_path} not found. Skipping GW data update.", level="WARNING")

data_directory = "Fantasy-Premier-League/data"
seasons = ["2022-23", "2023-24", "2024-25"]

for season in seasons:
    season_path = os.path.join(data_directory, season)
    log(f"Processing season: {season_path}", level="INFO")
    update_player_folders_and_gw_data(season_path)


# In[14]:


# Now let's get a list of team ids from each season, similarly as we did we players. Let's also add a column for team's name in understat
# database
def generate_master_team_list(data_directory, seasons, output_file="master_team_list_v2.csv"):
    """
    Generate a master team list with unique IDs and seasonal IDs.

    Args:
        data_directory (str): Base directory containing the season data.
        seasons (list): List of season names (e.g., ["2023-24", "2022-23", ...]).
        output_file (str): Name of the output file for the master team list.

    Returns:
        pd.DataFrame: The consolidated DataFrame of team IDs across seasons.
    """
    team_data = {}
    next_unique_id = 1

    understat_mapping = {
        "Man City": "Manchester City",
        "Man Utd": "Manchester United",
        "Newcastle": "Newcastle United",
        "Nott'm Forest": "Nottingham Forest",
        "Spurs": "Tottenham",
        "Wolves": "Wolverhampton Wanderers",
        "Sheffield Utd": "Sheffield United",
    }

    # Process 2024-25 season first to assign Unique IDs
    main_season = "2024-25"
    main_season_path = os.path.join(data_directory, main_season, "teams.csv")

    if check_file_exists(main_season_path):
        main_season_df = load_csv(main_season_path)
        if main_season_df is not None:
            for _, row in main_season_df.iterrows():
                team_name = row["name"]
                team_id = row["id"]
                if team_name not in team_data:
                    team_data[team_name] = {
                        "Name": team_name,
                        "Unique_Id": next_unique_id,
                        "24_Id": team_id,
                    }
                    next_unique_id += 1
    else:
        log(f"teams.csv not found for main season: {main_season}", level="ERROR")
        return None

    # Process other seasons
    for season in seasons:
        if season == main_season:
            continue

        season_short = season[:4][-2:]  # Extract year
        season_path = os.path.join(data_directory, season, "teams.csv")

        if check_file_exists(season_path):
            season_df = load_csv(season_path)
            if season_df is not None:
                for _, row in season_df.iterrows():
                    team_name = row["name"]
                    team_id = row["id"]
                    if team_name not in team_data:
                        # Add new team with placeholder for Unique_Id
                        team_data[team_name] = {"Name": team_name, "Unique_Id": next_unique_id}
                        next_unique_id += 1
                    team_data[team_name][f"{season_short}_Id"] = team_id
        else:
            log(f"Skipping season {season}. teams.csv not found.", level="WARNING")

    # Map the names, if name not in mapping then use the normal one
    for team_name, team_info in team_data.items():
        understat_name = understat_mapping.get(team_name, team_name)
        team_info["Understat_Team_Name"] = understat_name

    # Create DataFrame from team_data dictionary
    consolidated_df = pd.DataFrame.from_dict(team_data, orient="index")
    consolidated_df = consolidated_df.fillna(-1)  # Fill missing season IDs with -1
    consolidated_df = consolidated_df.astype({col: int for col in consolidated_df.columns if col.endswith("_Id")})

    output_path = os.path.join(data_directory, output_file)
    save_csv(consolidated_df, output_path)

    log(f"Master team list saved to {output_path}")
    return consolidated_df
    
data_directory = "Fantasy-Premier-League/data"
seasons = ["2024-25", "2023-24", "2022-23"]
generate_master_team_list(data_directory, seasons)


# In[15]:


# Let's make use of understat data. In order to do that we have to update team ids in fixtures.csv and then we can add some stats for each
# game
def update_fixtures_file_ids(data_directory, seasons, master_team_list):
    """
    Update the team_a and team_h columns in the fixtures.csv files for each season based on the master team list.

    Args:
        data_directory (str): Base directory containing the season data.
        seasons (list): List of season names to update (e.g., ["2023-24", "2022-23"]).
        master_team_list (pd.DataFrame): DataFrame containing the master team list with Unique_Id mapping.
    """
    for season in seasons:
        season_path = os.path.join(data_directory, season)
        fixtures_file_path = os.path.join(season_path, "fixtures.csv")

        if check_file_exists(fixtures_file_path):
            try:
                fixtures_data = load_csv(fixtures_file_path)

                if "team_a" in fixtures_data.columns and "team_h" in fixtures_data.columns:
                    # Determine the correct ID column for the season
                    season_year = season.split("-")[0]
                    id_column = f"{season_year[-2:]}_Id"

                    if id_column in master_team_list.columns:
                        # Create a mapping from seasonal IDs to Unique IDs
                        id_mapping = master_team_list.set_index(id_column)["Unique_Id"].to_dict()

                        # Map team_a and team_h IDs using the master team list
                        fixtures_data["team_a"] = fixtures_data["team_a"].map(id_mapping)
                        fixtures_data["team_h"] = fixtures_data["team_h"].map(id_mapping)

                        # Save the updated fixtures data back to the file
                        save_csv(fixtures_data, fixtures_file_path)
                        log(f"Updated team_a and team_h in {fixtures_file_path}")
                    else:
                        log(f"ID column {id_column} not found in master team list. Skipping season {season}.", level="WARNING")
            except Exception as e:
                log(f"Error updating {fixtures_file_path}: {e}", level="ERROR")

def update_fixtures_with_understat(data_directory, seasons, master_team_list):
    """
    Update fixtures.csv with xG, deep, and npxG data from Understat.

    Args:
        data_directory (str): Base directory containing the season data.
        seasons (list): List of season names to update (e.g., ["2023-24", "2022-23"]).
        master_team_list (pd.DataFrame): DataFrame containing the master team list
    """
    
    # Create a mapping from team names to get folders easier
    master_team_list["Understat_Name"] = master_team_list["Understat_Team_Name"].str.replace(" ", "_")

    for season in seasons:
        season_short = season[:4][-2:]
        log(f"Processing season: {season}", level="INFO")
        season_path = os.path.join(data_directory, season)
        fixtures_file_path = os.path.join(season_path, "fixtures.csv")

        if not check_file_exists(fixtures_file_path):
            log(f"fixtures.csv not found for season {season}. Skipping.", level="WARNING")
            continue

        fixtures_data = load_csv(fixtures_file_path)
        if fixtures_data is None or "kickoff_time" not in fixtures_data.columns:
            log(f"Invalid fixtures.csv for season {season}. Skipping.", level="WARNING")
            continue

        # Convert kickoff_time to datetime for matching
        fixtures_data["kickoff_time"] = pd.to_datetime(fixtures_data["kickoff_time"], utc=True).dt.date
        for _, team_row in master_team_list.iterrows():
            if team_row[f"{season_short}_Id"] == -1:
                continue
            understat_name = team_row["Understat_Name"]
            understat_file_path = os.path.join(season_path, "understat", f"understat_{understat_name}.csv")
            
            if not check_file_exists(understat_file_path):
                log(f"Understat file not found: {understat_file_path}. Skipping team {understat_name}.", level="WARNING")
                continue

            understat_data = load_csv(understat_file_path)
            if understat_data is None or "date" not in understat_data.columns:
                log(f"Invalid understat file: {understat_file_path}. Skipping team {understat_name}.", level="WARNING")
                continue

            # Convert date for matching
            understat_data["date"] = pd.to_datetime(understat_data["date"]).dt.date

            for match_date, date_filtered_fixtures in fixtures_data.groupby("kickoff_time"):
                understat_rows = understat_data[understat_data["date"] == match_date]

                for _, fixture in date_filtered_fixtures.iterrows():
                    if fixture["team_a"] == team_row["Unique_Id"]:
                        team_column = "team_a"
                        understat_row = understat_rows[understat_rows["h_a"] == "a"]
                    elif fixture["team_h"] == team_row["Unique_Id"]:
                        team_column = "team_h"
                        understat_row = understat_rows[understat_rows["h_a"] == "h"]
                    else:
                        continue

                    # If matches found, update the fixture
                    if not understat_row.empty:
                        understat_row = understat_row.iloc[0]
                        fixtures_data.at[fixture.name, f"{team_column}_xg"] = understat_row["xG"]
                        fixtures_data.at[fixture.name, f"{team_column}_deep"] = understat_row["deep"]
                        fixtures_data.at[fixture.name, f"{team_column}_npxg"] = understat_row["npxG"]

        save_csv(fixtures_data, fixtures_file_path)
        log(f"Updated fixtures.csv for season {season}", level="INFO")

data_directory = "Fantasy-Premier-League/data"
seasons = ["2024-25", "2023-24", "2022-23"]
master_team_list_file = "master_team_list_v2.csv"
master_team_list_path = os.path.join(data_directory, master_team_list_file)
if not check_file_exists(master_team_list_path):
    log("Master team list file not found. Aborting process.", level="ERROR")
else:
    master_team_list = load_csv(master_team_list_path)

    if master_team_list is not None:
        update_fixtures_file_ids(data_directory, seasons, master_team_list)

        # Step 2: Update fixtures.csv with Understat data
        update_fixtures_with_understat(data_directory, seasons, master_team_list)
    else:
        log("Failed to load master team list. Aborting process.", level="ERROR")


# In[16]:


# Once we have that we have to go into each season folder (except 2024-25) and update:
# - the opponent_team in gw.csv for each player
# - team in players_raw.csv
# - id in teams.csv
def update_team_ids_in_seasons(data_directory, seasons, master_team_list_file):
    """
    Update team IDs in all relevant files across seasons based on the master team list.

    Args:
        data_directory (str): Base directory containing the season data.
        seasons (list): List of season names to update (e.g., ["2023-24", "2022-23"]).
        master_team_list_file (str): Path to the master team list CSV file.
    """

    master_team_list_path = os.path.join(data_directory, master_team_list_file)
    if not check_file_exists(master_team_list_path):
        log("Master team list file not found. Aborting updates.", level="ERROR")
        return

    master_team_list = load_csv(master_team_list_path)
    if master_team_list is None:
        log("Failed to load master team list. Aborting updates.", level="ERROR")
        return

    # Process each season
    for season in seasons:
        log(f"Processing season: {season}", level="INFO")
        season_path = os.path.join(data_directory, season)

        # Generate mapping for the current season: {this_season_team_id -> unique_id}
        season_short = season[:4][-2:]
        id_column = f"{season_short}_Id"
        if id_column not in master_team_list.columns:
            log(f"Column {id_column} not found in master team list. Skipping season {season}.", level="WARNING")
            continue
        
        id_mapping = (
            master_team_list.set_index(id_column)["Unique_Id"]
            .dropna()
            .to_dict()
        )
        id_mapping = {k: v for k, v in id_mapping.items() if k != -1}
        
        players_folder = os.path.join(season_path, "players")
        if check_file_exists(players_folder):
            for player_folder in os.listdir(players_folder):
                gw_file_path = os.path.join(players_folder, player_folder, "gw.csv")
                if check_file_exists(gw_file_path):
                    try:
                        gw_data = load_csv(gw_file_path)
                        if "opponent_team" in gw_data.columns:
                            gw_data["opponent_team"] = gw_data["opponent_team"].map(id_mapping)
                            save_csv(gw_data, gw_file_path)
                            log(f"Updated opponent_team in {gw_file_path}", level="DEBUG")
                    except Exception as e:
                        log(f"Error updating {gw_file_path}: {e}", level="ERROR")

        # Update players_raw.csv
        players_raw_file_path = os.path.join(season_path, "players_raw.csv")
        if check_file_exists(players_raw_file_path):
            try:
                players_raw_data = load_csv(players_raw_file_path)
                if "team" in players_raw_data.columns:
                    players_raw_data["team"] = players_raw_data["team"].map(id_mapping)
                    save_csv(players_raw_data, players_raw_file_path)
                    log(f"Updated team in {players_raw_file_path}")
            except Exception as e:
                log(f"Error updating {players_raw_file_path}: {e}", level="ERROR")

        # Update teams.csv
        teams_file_path = os.path.join(season_path, "teams.csv")
        if check_file_exists(teams_file_path):
            try:
                teams_data = load_csv(teams_file_path)
                if "id" in teams_data.columns:
                    teams_data["id"] = teams_data["id"].map(id_mapping).fillna(teams_data["id"])
                    save_csv(teams_data, teams_file_path)
                    log(f"Updated id in {teams_file_path}")
            except Exception as e:
                log(f"Error updating {teams_file_path}: {e}", level="ERROR")

    log("All seasons updated successfully.", level="INFO")

data_directory = "Fantasy-Premier-League/data"
seasons = ["2023-24", "2022-23"]
master_team_list_file = "master_team_list_v2.csv"

update_team_ids_in_seasons(data_directory, seasons, master_team_list_file)


# In[17]:


# Player data doesn't have his own team id (and they can switch teams during season so we have to add that).
# For that we have to use fixtures.csv. Let's also rename round to gameweek
def add_own_team_to_players(data_directory, season, fixtures_file):
    """
    Adds the 'own_team' column to gw.csv files for each player and renames 'round' to 'gameweek'.

    Args:
        data_directory (str): Base directory containing the season data.
        season (str): Season name (e.g., "2022-23").
        fixtures_file (str): Name of the fixtures file (e.g., "fixtures.csv").

    Returns:
        None
    """
    fixtures_path = os.path.join(data_directory, season, fixtures_file)
    if not check_file_exists(fixtures_path):
        return

    fixtures = load_csv(fixtures_path)
    if fixtures is None:
        return

    players_folder = os.path.join(data_directory, season, "players")
    if not check_file_exists(players_folder):
        return

    for player_folder in os.listdir(players_folder):
        player_folder_path = os.path.join(players_folder, player_folder)
        gw_file_path = os.path.join(player_folder_path, "gw.csv")

        if check_file_exists(gw_file_path):
            try:
                gw_data = load_csv(gw_file_path)
                if gw_data is None:
                    continue

                # Ensure required columns exist
                if 'fixture' not in gw_data.columns or 'was_home' not in gw_data.columns:
                    log(f"Missing columns in {gw_file_path}. Skipping.", level="WARNING")
                    continue

                # Merge fixture data and compute 'own_team'
                gw_data = gw_data.merge(
                    fixtures[['id', 'team_a', 'team_h']],
                    left_on='fixture',
                    right_on='id',
                    how='left'
                )
                gw_data['own_team'] = gw_data.apply(
                    lambda x: x['team_h'] if x['was_home'] else x['team_a'], axis=1
                )

                # Rename 'round' to 'gameweek'
                if 'round' in gw_data.columns:
                    gw_data.rename(columns={'round': 'gameweek'}, inplace=True)

                # Drop unnecessary columns
                columns_to_drop = ['id', 'team_a', 'team_h']
                gw_data.drop(columns=[col for col in columns_to_drop if col in gw_data.columns], inplace=True)

                # Save the updated file
                save_csv(gw_data, gw_file_path)
                log(f"Updated {gw_file_path} with 'own_team' and 'gameweek'.", level="DEBUG")

            except Exception as e:
                log(f"Error processing {gw_file_path}: {e}", level="ERROR")

data_directory = "Fantasy-Premier-League/data"
seasons = ["2022-23", "2023-24", "2024-25"]
fixtures_file = "fixtures.csv"

for season in seasons:
    log(f"Processing season: {season}", level="INFO")
    add_own_team_to_players(data_directory, season, fixtures_file)


# In[18]:


# Now we can add short team names for player and opponent and also strengths of both teams from teams.csv
def add_team_strengths_and_short_names(data_directory, season, teams_file):
    """
    Add team strengths and short names for players and opponents in gw.csv files.

    Args:
        data_directory (str): Base directory containing the season data.
        season (str): Season name (e.g., "2022-23").
        teams_file (str): Name of the teams.csv file.
    """
    teams_path = os.path.join(data_directory, season, teams_file)
    if not check_file_exists(teams_path):
        return

    # Load teams.csv and index by team ID
    teams_data = load_csv(teams_path)
    if teams_data is None:
        return
    teams_data = teams_data.set_index("id")

    players_folder = os.path.join(data_directory, season, "players")
    if not check_file_exists(players_folder):
        return

    # Iterate through each player folder and update gw.csv
    for player_folder in os.listdir(players_folder):
        player_folder_path = os.path.join(players_folder, player_folder)
        gw_file_path = os.path.join(player_folder_path, "gw.csv")

        if check_file_exists(gw_file_path):
            try:
                gw_data = load_csv(gw_file_path)
                if gw_data is None:
                    continue

                # Merge opponent team data
                gw_data = gw_data.merge(
                    teams_data[[
                        'short_name', 'strength_attack_home', 'strength_attack_away',
                        'strength_defence_home', 'strength_defence_away'
                    ]],
                    left_on='opponent_team',
                    right_index=True,
                    how='left'
                ).rename(columns={
                    'short_name': 'opponent_short_name',
                    'strength_attack_home': 'opponent_attack_home',
                    'strength_attack_away': 'opponent_attack_away',
                    'strength_defence_home': 'opponent_defence_home',
                    'strength_defence_away': 'opponent_defence_away'
                })

                # Add opponent strengths
                gw_data['opponent_attack'] = gw_data.apply(
                    lambda x: x['opponent_attack_away'] if x['was_home'] else x['opponent_attack_home'], axis=1
                )
                gw_data['opponent_defense'] = gw_data.apply(
                    lambda x: x['opponent_defence_away'] if x['was_home'] else x['opponent_defence_home'], axis=1
                )

                # Merge own team data
                gw_data = gw_data.merge(
                    teams_data[[
                        'short_name', 'strength_attack_home', 'strength_attack_away',
                        'strength_defence_home', 'strength_defence_away'
                    ]],
                    left_on='own_team',
                    right_index=True,
                    how='left'
                ).rename(columns={
                    'short_name': 'own_short_name',
                    'strength_attack_home': 'own_attack_home',
                    'strength_attack_away': 'own_attack_away',
                    'strength_defence_home': 'own_defence_home',
                    'strength_defence_away': 'own_defence_away'
                })

                # Add own team strengths
                gw_data['own_attack'] = gw_data.apply(
                    lambda x: x['own_attack_home'] if x['was_home'] else x['own_attack_away'], axis=1
                )
                gw_data['own_defense'] = gw_data.apply(
                    lambda x: x['own_defence_home'] if x['was_home'] else x['own_defence_away'], axis=1
                )

                # Drop temporary columns
                gw_data.drop(columns=[
                    'opponent_attack_home', 'opponent_attack_away',
                    'opponent_defence_home', 'opponent_defence_away',
                    'own_attack_home', 'own_attack_away',
                    'own_defence_home', 'own_defence_away'
                ], inplace=True)

                # Save updated gw.csv
                save_csv(gw_data, gw_file_path)
                log(f"Updated {gw_file_path} with team strengths and short names.", level="DEBUG")
            except Exception as e:
                log(f"Error processing {gw_file_path}: {e}", level="ERROR")

data_directory = "Fantasy-Premier-League/data"
seasons = ["2022-23", "2023-24", "2024-25"]
teams_file = "teams.csv"

for season in seasons:
    log(f"Processing season: {season}", level="INFO")
    add_team_strengths_and_short_names(data_directory, season, teams_file)


# In[19]:


# This isn't used for now as we use newer season data
# def add_crowds_to_gw(data_directory, season):
#     """
#     Add a 'crowds' column to gw.csv for each player to indicate if there were crowds at the stadium.

#     Args:
#         data_directory (str): Base directory containing the season data.
#         season (str): Season name (e.g., "2020-21").

#     Returns:
#         None
#     """
#     # Define the period without crowds (COVID restrictions)
#     no_crowds_start = datetime(2020, 3, 15, tzinfo=timezone.utc)
#     no_crowds_end = datetime(2021, 6, 17, tzinfo=timezone.utc)

#     players_folder = os.path.join(data_directory, season, "players")
#     if not check_file_exists(players_folder):
#         return

#     # Iterate through each player's folder
#     for player_folder in os.listdir(players_folder):
#         player_folder_path = os.path.join(players_folder, player_folder)
#         gw_file_path = os.path.join(player_folder_path, "gw.csv")

#         if check_file_exists(gw_file_path):
#             try:
#                 gw_data = load_csv(gw_file_path)
#                 if gw_data is None or "kickoff_time" not in gw_data.columns:
#                     log(f"'kickoff_time' not found in {gw_file_path}. Skipping.", level="WARNING")
#                     continue

#                 # Convert kickoff_time to UTC and add 'crowds' column
#                 gw_data['kickoff_time'] = pd.to_datetime(gw_data['kickoff_time']).dt.tz_convert('UTC')
#                 gw_data['crowds'] = gw_data['kickoff_time'].apply(
#                     lambda x: 0 if no_crowds_start <= x <= no_crowds_end else 1
#                 )

#                 save_csv(gw_data, gw_file_path)
#                 log(f"Updated {gw_file_path} with crowds column.")
#             except Exception as e:
#                 log(f"Error processing {gw_file_path}: {e}", level="ERROR")

# data_directory = "Fantasy-Premier-League/data"
# seasons = ["2022-23", "2023-24", "2024-25"]

# for season in seasons:
#     log(f"Processing season: {season}", level="INFO")
#     add_crowds_to_gw(data_directory, season)


# In[20]:


# Let's add season number
def add_season_to_gw(data_directory, season):
    """
    Add the season year as a column to gw.csv files for each player.

    Args:
        data_directory (str): Base directory containing the season data.
        season (str): Season name (e.g., "2020-21").

    Returns:
        None
    """
    # Extract the start year from the season folder (e.g., "2021-22" -> "2021")
    try:
        season_year = int(season.split("-")[0])
    except ValueError:
        log(f"Invalid season format: {season}. Skipping.", level="ERROR")
        return

    players_folder = os.path.join(data_directory, season, "players")
    if not check_file_exists(players_folder):
        return

    for player_folder in os.listdir(players_folder):
        player_folder_path = os.path.join(players_folder, player_folder)
        gw_file_path = os.path.join(player_folder_path, "gw.csv")

        if check_file_exists(gw_file_path):
            try:
                gw_data = load_csv(gw_file_path)
                if gw_data is None:
                    continue

                gw_data["season"] = season_year

                save_csv(gw_data, gw_file_path)
                log(f"Updated {gw_file_path} with season column.", level="DEBUG")
            except Exception as e:
                log(f"Error processing {gw_file_path}: {e}", level="ERROR")

data_directory = "Fantasy-Premier-League/data"
seasons = ["2022-23", "2023-24", "2024-25"]

for season in seasons:
    log(f"Processing season: {season}", level="INFO")
    add_season_to_gw(data_directory, season)


# In[21]:


# Penalties are not goods for predictions, let's value them less
def add_adjusted_xg_to_gw(data_directory, season, penalty_adjustment_factor=0.5):
    """
    Add adjusted xG to gw.csv files for each player.

    Args:
        data_directory (str): Base directory containing the season data.
        season (str): Season name (e.g., "2020-21").
        penalty_adjustment_factor (float): The adjustment factor for penalties. Defaults to 0.5.

    Returns:
        None
    """
    players_folder = os.path.join(data_directory, season, "players")
    if not check_file_exists(players_folder):
        log(f"Players folder not found for season {season}. Skipping.", level="ERROR")
        return

    for player_folder in os.listdir(players_folder):
        player_folder_path = os.path.join(players_folder, player_folder)
        gw_file_path = os.path.join(player_folder_path, "gw.csv")

        if check_file_exists(gw_file_path):
            try:
                gw_data = load_csv(gw_file_path)
                if gw_data is None:
                    continue

                if "expected_goals" not in gw_data.columns or "npxG" not in gw_data.columns:
                    # Some players will have these missing since gw.csv has data for players with 0 minutes and those are excluded from 
                    # understat
                    log(f"Missing required columns in {gw_file_path}. Skipping.", level="DEBUG")
                    continue

                # Calculate adjusted xG
                gw_data["adjusted_xg"] = gw_data["npxG"] + penalty_adjustment_factor * (gw_data["expected_goals"] - gw_data["npxG"])

                save_csv(gw_data, gw_file_path)
                log(f"Updated {gw_file_path} with adjusted xG.", level="DEBUG")
            except Exception as e:
                log(f"Error processing {gw_file_path}: {e}", level="ERROR")

data_directory = "Fantasy-Premier-League/data"
seasons = ["2022-23", "2023-24", "2024-25"]

for season in seasons:
    log(f"Processing season: {season}", level="INFO")
    add_adjusted_xg_to_gw(data_directory, season)


# In[22]:


# Let's add the player's number in penalty kick order
def add_penalty_kick_order_to_gw(data_directory, season):
    """
    Add penalty kick order to gw.csv files for each player.

    Args:
        data_directory (str): Base directory containing the season data.
        season (str): Season name (e.g., "2020-21").

    Returns:
        None
    """
    season_path = os.path.join(data_directory, season)
    players_raw_path = os.path.join(season_path, "players_raw.csv")
    players_folder = os.path.join(season_path, "players")

    if not check_file_exists(players_raw_path):
        log(f"players_raw.csv not found for season {season}. Skipping.", level="ERROR")
        return

    if not check_file_exists(players_folder):
        log(f"Players folder not found for season {season}. Skipping.", level="ERROR")
        return

    try:
        # Load players_raw.csv
        players_raw = load_csv(players_raw_path)
        if players_raw is None or "penalties_order" not in players_raw.columns:
            log(f"Missing penalties_order column in players_raw.csv for season {season}. Skipping.", level="WARNING")
            return

        # Iterate through player folders
        for player_folder in os.listdir(players_folder):
            player_folder_path = os.path.join(players_folder, player_folder)
            gw_file_path = os.path.join(player_folder_path, "gw.csv")

            if check_file_exists(gw_file_path):
                try:
                    gw_data = load_csv(gw_file_path)
                    if gw_data is None:
                        continue

                    # Extract player ID from the folder name
                    player_id = int(player_folder.split("_")[-1])

                    # Get penalty order for this player
                    penalty_order = players_raw.loc[players_raw["id"] == player_id, "penalties_order"].values
                    penalty_order = penalty_order[0] if len(penalty_order) > 0 else None

                    # Add penalty order to gw.csv
                    gw_data["penalties_order"] = penalty_order

                    save_csv(gw_data, gw_file_path)
                    log(f"Updated {gw_file_path} with penalties_order.", level="DEBUG")
                except Exception as e:
                    log(f"Error processing {gw_file_path}: {e}", level="ERROR")
    except Exception as e:
        log(f"Error processing players_raw.csv for season {season}: {e}", level="ERROR")

data_directory = "Fantasy-Premier-League/data"
seasons = ["2022-23", "2023-24", "2024-25"]

for season in seasons:
    log(f"Processing season: {season}", level="INFO")
    add_penalty_kick_order_to_gw(data_directory, season)


# In[23]:


def add_team_stats_to_gw(data_directory, season):
    """
    Add xG, npxg, deep (of both teams) to gw.csv files for each player.

    Args:
        data_directory (str): Base directory containing the season data.
        season (str): Season name (e.g., "2020-21").

    Returns:
        None
    """
    season_path = os.path.join(data_directory, season)
    fixtures_path = os.path.join(season_path, "fixtures.csv")
    players_folder = os.path.join(season_path, "players")

    if not check_file_exists(fixtures_path):
        log(f"fixtures.csv not found for season {season}. Skipping.", level="ERROR")
        return

    if not check_file_exists(players_folder):
        log(f"Players folder not found for season {season}. Skipping.", level="ERROR")
        return

    required_columns = [
        "team_h_xg", "team_h_deep", "team_h_npxg", 
        "team_a_xg", "team_a_deep", "team_a_npxg"
    ]

    try:
        fixtures = load_csv(fixtures_path)
        if fixtures is None or not all(column in fixtures.columns for column in required_columns):
            log(f"Missing required columns in fixtures.csv for season {season}. Skipping.", level="WARNING")
            return

        for player_folder in os.listdir(players_folder):
            player_folder_path = os.path.join(players_folder, player_folder)
            gw_file_path = os.path.join(player_folder_path, "gw.csv")

            if check_file_exists(gw_file_path):
                try:
                    gw_data = load_csv(gw_file_path)
                    if gw_data is None:
                        continue

                    for idx, row in gw_data.iterrows():
                        fixture_id = row.get("fixture")
                        was_home = row.get("was_home")

                        if pd.isna(fixture_id) or pd.isna(was_home):
                            continue

                        fixture_row = fixtures[fixtures["id"] == fixture_id]
                        if not fixture_row.empty:
                            fixture_row = fixture_row.iloc[0]
                            if was_home:
                                gw_data.at[idx, "team_xg"] = fixture_row["team_h_xg"]
                                gw_data.at[idx, "team_deep"] = fixture_row["team_h_deep"]
                                gw_data.at[idx, "team_npxg"] = fixture_row["team_h_npxg"]
                                gw_data.at[idx, "opponent_xg"] = fixture_row["team_a_xg"]
                                gw_data.at[idx, "opponent_deep"] = fixture_row["team_a_deep"]
                                gw_data.at[idx, "opponent_npxg"] = fixture_row["team_a_npxg"]     
                            else:
                                gw_data.at[idx, "team_xg"] = fixture_row["team_a_xg"]
                                gw_data.at[idx, "team_deep"] = fixture_row["team_a_deep"]
                                gw_data.at[idx, "team_npxg"] = fixture_row["team_a_npxg"]
                                gw_data.at[idx, "opponent_xg"] = fixture_row["team_h_xg"]
                                gw_data.at[idx, "opponent_deep"] = fixture_row["team_h_deep"]
                                gw_data.at[idx, "opponent_npxg"] = fixture_row["team_h_npxg"]

                    # Save updated gw.csv
                    save_csv(gw_data, gw_file_path)
                    log(f"Updated {gw_file_path} with team stats.", level="DEBUG")
                except Exception as e:
                    log(f"Error processing {gw_file_path}: {e}", level="ERROR")
    except Exception as e:
        log(f"Error processing fixtures.csv for season {season}: {e}", level="ERROR")


data_directory = "Fantasy-Premier-League/data"
seasons = ["2022-23", "2023-24", "2024-25"]

for season in seasons:
    log(f"Processing season: {season}", level="INFO")
    add_team_stats_to_gw(data_directory, season)


# In[24]:


# Finally - let's make csv with players from each position for each season
def process_season_for_positions(data_directory, season):
    """
    Process a single season to create position-specific CSVs for players.

    Args:
        data_directory (str): Base directory containing the season data.
        season (str): Season name (e.g., "2022-23").

    Returns:
        None
    """
    players_folder = os.path.join(data_directory, season, "players")
    if not check_file_exists(players_folder, log_missing=True):
        return

    processed_data_folder = os.path.join(data_directory, season, "processed_data")
    os.makedirs(processed_data_folder, exist_ok=True)

    # DataFrames for each position
    position_dfs = {
        "DEF": pd.DataFrame(),
        "MID": pd.DataFrame(),
        "FWD": pd.DataFrame(),
        "GK": pd.DataFrame(),
    }

    # Iterate through player directories
    for player_folder in os.listdir(players_folder):
        # Extract position from folder name
        position = player_folder.split("_")[0]

        # Read the player's gw.csv
        player_folder_path = os.path.join(players_folder, player_folder)
        gw_file_path = os.path.join(player_folder_path, "gw.csv")

        if check_file_exists(gw_file_path):
            try:
                # Load gw.csv and add it to the respective position DataFrame
                gw_data = load_csv(gw_file_path)
                if gw_data is not None and position in position_dfs:
                    position_dfs[position] = pd.concat(
                        [position_dfs[position], gw_data], ignore_index=True
                    )
            except Exception as e:
                log(f"Error processing {gw_file_path}: {e}", level="ERROR")

    # Save combined CSV for each position in the processed_data folder
    for position, df in position_dfs.items():
        if not df.empty:
            output_file_path = os.path.join(processed_data_folder, f"{position}_players.csv")
            save_csv(df, output_file_path)
            log(f"Saved {position} players for season {season} to {output_file_path}.")

data_directory = "Fantasy-Premier-League/data"
seasons = ["2022-23", "2023-24", "2024-25"]

for season in seasons:
    log(f"Processing season: {season}", level="INFO")
    process_season_for_positions(data_directory, season)


# In[ ]:




