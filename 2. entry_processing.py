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
    Check for duplicate player names across multiple seasons in players_raw.csv.

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
        players_raw_path = os.path.join(season_path, "players_raw.csv")
        
        if not check_file_exists(players_raw_path):
            log(f"players_raw.csv not found for season {season}. Skipping.", level="WARNING")
            continue

        players_raw = load_csv(players_raw_path)
        if players_raw is None:
            continue

        duplicates = players_raw.duplicated(subset=['first_name', 'second_name'], keep=False)
        duplicate_names = players_raw.loc[duplicates, ['id', 'first_name', 'second_name']].drop_duplicates()
        
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


# We remove the duplicates
def remove_players_from_raw_and_folders(data_directory, players_to_remove):
    """
    Remove players from players_raw.csv and their associated folders.

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
        players_raw_path = os.path.join(season_path, "players_raw.csv")
        players_folder_path = os.path.join(season_path, "players")
        player_folder_name = f"{first_name}_{second_name}_{player_id}"
        player_folder_path = os.path.join(players_folder_path, player_folder_name)

        if not check_file_exists(players_raw_path):
            log(f"players_raw.csv not found for season {season}. Skipping.", level="WARNING")
            continue
        players_raw = load_csv(players_raw_path)
        if players_raw is None:
            continue

        # Find and remove the player from players_raw.csv
        matching_rows = players_raw[
            (players_raw["id"] == player_id) &
            (players_raw["first_name"] == first_name) &
            (players_raw["second_name"] == second_name)
        ]

        if not matching_rows.empty:
            players_raw.drop(matching_rows.index, inplace=True)
            save_csv(players_raw, players_raw_path)
            log(f"Removed {first_name} {second_name} (ID: {player_id}) from {players_raw_path}.")
        else:
            log(f"Player {first_name} {second_name} (ID: {player_id}) not found in {players_raw_path}. Skipping.", level="WARNING")

        # Remove player folder
        remove_file_or_dir(player_folder_path)

data_directory = "Fantasy-Premier-League/data"
players_to_remove = [
        {"season": "2022-23", "first_name": "Ben", "second_name": "Davies", "id": 499},
]

remove_players_from_raw_and_folders(data_directory, players_to_remove)


# In[4]:


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
    main_season_path = os.path.join(data_directory, main_season, "players_raw.csv")

    # Process the main season first. We want to keep these ids since we will use same ones for predictions. 
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

        season_path = os.path.join(data_directory, season, "players_raw.csv")
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


# In[5]:


# # Script below was used to get duplicate candidates (players can have name changes across seasons). 
# After finding them I checked it manually

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


# In[6]:


# After manual checking the candidates now we have to update master_player_list.csv 
# We might get some warnings here but we can safely ignore them (one player can have multiple rows and deleting one of them causes errors
# but we still merge the data to the one with lowest ID)
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


# In[7]:


# With the new list we can move on to renaming player folders to have the new ids and their positions
# We also update the processed_players

def update_players_and_rename_folders(data_directory, seasons, master_player_file):
    """
    Update player IDs and names in players_raw.csv and rename player folders to Pos_Name_UniqueId.

    Args:
        data_directory (str): Base directory containing the data.
        seasons (list): List of seasons to process.
        master_player_file (str): Master player list file name.

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
        players_raw_file = os.path.join(season_path, "players_raw.csv")

        if not check_file_exists(players_raw_file):
            log(f"players_raw.csv not found for season: {season}. Skipping.", level="WARNING")
            continue

        players_raw = load_csv(players_raw_file)
        season_short = season[:4][-2:]  # Extract short season ID (e.g., "21" for "2021-22")
        season_id_column = f"{season_short}_id"

        # Update names and add unique id's, rename folders
        for index, row in players_raw.iterrows():
            current_id = row["id"]
            matching_row = master_player_verified[master_player_verified[season_id_column] == current_id]

            if matching_row.empty:
                log(f"No matching row for player ID {current_id} in season {season}.", level="WARNING")
                continue

            # Get new data
            new_data = matching_row.iloc[0]
            unique_id = new_data["Unique_ID"]
            first_name = new_data["First_Name"]
            last_name = new_data["Last_Name"]
            element_type = row["element_type"]
            position = ["GK", "DEF", "MID", "FWD"][element_type - 1]

            # Update player data in the DataFrame
            players_raw.at[index, "unique_id"] = unique_id
            players_raw.at[index, "first_name"] = first_name
            players_raw.at[index, "second_name"] = last_name

            old_folder_name = f"{row['first_name']}_{row['second_name']}_{current_id}"
            matching_folder = next(
                (f for f in os.listdir(players_folder_path) if f == old_folder_name),
                None
            )

            if matching_folder:
                old_path = os.path.join(players_folder_path, matching_folder)
                new_folder_name = f"{position}_{first_name}_{last_name}_{unique_id}"
                new_path = os.path.join(players_folder_path, new_folder_name)

                if not check_file_exists(new_path):
                    try:
                        shutil.move(old_path, new_path)
                        log(f"Renamed folder {matching_folder} to {new_folder_name}", level="DEBUG")
                    except Exception as e:
                        log(f"Error renaming {old_path} to {new_path}: {e}", level="ERROR")
                else:
                    log(f"Destination folder {new_path} already exists. Skipping rename for ID {current_id}.", level="WARNING")
            else:
                log(f"No matching folder found for player ID {current_id} in season {season}.", level="WARNING")

        save_csv(players_raw, players_raw_file)
        log(f"Updated players_raw.csv saved for season {season}")

data_directory = "Fantasy-Premier-League/data"
seasons = ["2022-23", "2023-24", "2024-25"]
master_player_file = "master_player_v2.csv"
update_players_and_rename_folders(data_directory, seasons, master_player_file)


# In[8]:


# We have to rename some files because of their wrong names in the understat folder
def rename_files(directory):
    """
    Renames files in the specified directory by replacing '&#039;' with an apostrophe (').

    Parameters:
    - directory (str): The path to the directory containing the files to rename.
    """
    for filename in os.listdir(directory):
        if "&#039;" in filename:
            new_filename = filename.replace("&#039;", "'")
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


# In[9]:


# We have disparity between understat ids and fpl ids. For season 2022-23 we can use the existing file, for the other seasons we will
# have to create our own
def clean_html_entities_in_id_dict(data_directory, season, column_to_clean="Understat_Name", replacements=None):
    """
    Clean multiple HTML entities in the specified column of id_dict.csv.

    Args:
        data_directory (str): Base directory containing season data.
        season (str): Season folder name (e.g., "2022-23").
        column_to_clean (str): Column name to clean in id_dict.csv. Default is "Understat_Name".
        replacements (dict): Dictionary where keys are HTML entities to replace, 
                             and values are their replacements. Default is None.

    Returns:
        None
    """
    if replacements is None:
        replacements = {"&#039;": "'"}

    id_dict_path = os.path.join(data_directory, season, "id_dict.csv")
    id_dict = load_csv(id_dict_path)

    if column_to_clean in id_dict.columns:
        for entity, replacement in replacements.items():
            id_dict[column_to_clean] = id_dict[column_to_clean].str.replace(entity, replacement, regex=False)

        save_csv(id_dict, id_dict_path)
        log(f"Cleaned entities {list(replacements.keys())} in column '{column_to_clean}' for season {season}.", "INFO")
    else:
        log(f"Column '{column_to_clean}' not found in {id_dict_path}. No changes made.", "WARNING")

data_directory = "Fantasy-Premier-League/data"
season = "2022-23"
clean_html_entities_in_id_dict(data_directory, season)


# In[10]:


# Let's add understat ids from 2022-23 to master list

data_directory = "Fantasy-Premier-League/data"
id_dict_path_22 = os.path.join(data_directory, "2022-23", "id_dict.csv")
master_player_path = os.path.join(data_directory, "master_player_v2.csv")

id_dict = load_csv(id_dict_path_22)
master_player = load_csv(master_player_path)

# Merge id_dict with master_player to add Unique_ID
merged_data = master_player.merge(
    id_dict[['Understat_ID', 'FPL_ID']],
    left_on='22_id',
    right_on='FPL_ID',
    how='left'
)
merged_data = merged_data.drop(columns=['FPL_ID'])
save_csv(merged_data, master_player_path)
log(f"Added Understat_ID to {master_player_path}.", level="INFO")


# In[11]:


# Now we can add these understat ids that can be merged by full names
def add_missing_understat_ids(data_directory, master_player_file, seasons):
    """
    Add missing Understat_IDs to the master player list by matching Full_Name with Understat filenames.

    Args:
        data_directory (str): Base directory containing season data.
        master_player_file (str): Path to the master player file.
        seasons (list): List of seasons to process (e.g., ["2023-24", "2024-25"]).

    Returns:
        None
    """
    master_player_path = os.path.join(data_directory, master_player_file)
    master_df = load_csv(master_player_path)

    if master_df is None:
        log(f"Failed to load {master_player_file}. Exiting.", level="ERROR")
        return

    master_df['Full_Name'] = master_df['First_Name'].str.strip() + ' ' + master_df['Last_Name'].str.strip()

    for season in seasons:
        understat_dir = os.path.join(data_directory, season, "understat")
        
        if not os.path.isdir(understat_dir):
            log(f"Understat directory for season '{season}' not found. Skipping.", level="ERROR")
            continue

        # Extract Full_Name and Understat_ID from filenames
        understat_files = [f for f in os.listdir(understat_dir) if f.endswith('.csv') and not f.startswith("understat")]
        understat_data = []

        for file in understat_files:
            name_part, uid_part = os.path.splitext(file)[0].rsplit('_', 1)
            full_name = name_part.replace('_', ' ').strip()
            understat_data.append({"Full_Name": full_name, "Understat_ID": uid_part})
            log(f"Extracted: Full_Name='{full_name}', Understat_ID='{uid_part}' from file '{file}'", level="DEBUG")

        if not understat_data:
            log(f"No valid Understat files found for season '{season}'.", level="WARNING")
            continue

        understat_df = pd.DataFrame(understat_data)

        unmatched_players = master_df[master_df['Understat_ID'].isna()]
        log(unmatched_players, level="DEBUG")
        merged_df = pd.merge(
            unmatched_players.loc[:, unmatched_players.columns != 'Understat_ID'], 
            understat_df, 
            on="Full_Name", 
            how="left"
        )
        log(merged_df, level="DEBUG")
        
        for _, row in merged_df.iterrows():
            unique_id = row['Unique_ID']
            understat_id = row.get('Understat_ID', None)
            if pd.notna(understat_id):
                master_df.loc[master_df['Unique_ID'] == unique_id, 'Understat_ID'] = int(understat_id)
                log(f"Added Understat_ID {understat_id} for player {row['Full_Name']} (Unique_ID {unique_id}) from season {season}.", level="DEBUG")

    save_csv(master_df, master_player_path)
    log(f"Updated master player file with missing Understat_IDs for seasons {seasons}.", level="INFO")

data_directory = "Fantasy-Premier-League/data"
master_player_file = "master_player_v2.csv"
seasons = ["2023-24", "2024-25"]

add_missing_understat_ids(data_directory, master_player_file, seasons)


# In[12]:


# This was used to help map understat players that werent mapped by name
# def generate_id_mapping_file(data_directory, master_player_file, season, output_dir):
#     """
#     Generate a mapping file for players missing Understat_ID for manual verification.

#     Args:
#         data_directory (str): Base directory containing season data.
#         master_player_file (str): Path to the master player file.
#         season (str): Season to process (e.g., "2023-24").
#         output_dir (str): Directory where the generated file will be saved.

#     Returns:
#         None
#     """
#     master_player_path = os.path.join(data_directory, master_player_file)
#     master_df = load_csv(master_player_path)

#     if master_df is None:
#         log(f"Failed to load {master_player_file}. Exiting.", level="ERROR")
#         return

#     master_df['Full_Name'] = master_df['First_Name'].str.strip() + ' ' + master_df['Last_Name'].str.strip()

#     id_col = f"{season[:4][-2:]}_id"
#     season_players = master_df[(master_df[id_col] != -1) & (master_df['Understat_ID'].isna())]

#     if season_players.empty:
#         log(f"No players missing Understat_ID for season {season}.", level="INFO")
#         return

#     output_file = os.path.join(output_dir, f"missing_understat_{season}.csv")
#     os.makedirs(output_dir, exist_ok=True)
#     season_players[['Unique_ID', 'Full_Name', id_col]].to_csv(output_file, index=False)
#     log(f"Generated file for manual verification: {output_file}")

# data_directory = "Fantasy-Premier-League/data"
# master_player_file = "master_player_v2.csv"
# output_dir = "created_csv"
# season = "2023-24"

# generate_id_mapping_file(data_directory, master_player_file, season, output_dir)


# In[13]:


# We now add the 2023-24 mappings to master list, match by 23_id, file is created_csv/understat_map_2023-24.csv
data_directory = "Fantasy-Premier-League/data"
master_player_path = os.path.join(data_directory, "master_player_v2.csv")
understat_map_path = "created_csv/understat_map_2023-24.csv"

# Load data
master_player = pd.read_csv(master_player_path)
understat_map = pd.read_csv(understat_map_path)
updated_master = master_player.merge(
    understat_map[['23_id', 'Understat_ID']],
    on='23_id',
    how='left',
    suffixes=('', '_new')
)
updated_master['Understat_ID'] = updated_master['Understat_ID'].fillna(updated_master['Understat_ID_new']).astype('Int64')
updated_master.drop(columns=['Understat_ID_new'], inplace=True)
save_csv(updated_master, master_player_path)
log(f"Updated master player list with new Understat_ID mappings from {understat_map_path}.")


# In[14]:


# # This was used to get players with no understat id from season 2024-25
# data_directory = "Fantasy-Premier-League/data"
# master_player_file = "master_player_v2.csv"
# output_dir = "created_csv"
# season = "2024-25"

# generate_id_mapping_file(data_directory, master_player_file, season, output_dir)


# In[15]:


# We now add the 2024-25 mappings to master list
data_directory = "Fantasy-Premier-League/data"
master_player_path = os.path.join(data_directory, "master_player_v2.csv")
understat_map_path = "created_csv/understat_map_2024-25.csv"

# Load data
master_player = pd.read_csv(master_player_path)
understat_map = pd.read_csv(understat_map_path)
updated_master = master_player.merge(
    understat_map[['24_id', 'Understat_ID']],
    on='24_id',
    how='left',
    suffixes=('', '_new')
)
updated_master['Understat_ID'] = updated_master['Understat_ID'].fillna(updated_master['Understat_ID_new']).astype('Int64')
updated_master.drop(columns=['Understat_ID_new'], inplace=True)
save_csv(updated_master, master_player_path)
log(f"Updated master player list with new Understat_ID mappings from {understat_map_path}.")


# In[16]:


# Let's add some stats from understat to each player's gw.csv and also the unique id of each player and his full name
def update_gw_files_with_understat(data_directory, master_file, seasons, dry_run=False):
    """
    Update GW files using master player file and Understat

    Args:
        data_directory (str): Base directory containing season data.
        master_file (str): Path to the master player file.
        seasons (list): List of seasons to process.
        dry_run (bool): If True, no changes will be made to files.

    Returns:
        None
    """
    master_path = os.path.join(data_directory, master_file)
    master_df = load_csv(master_path)
    master_df["Understat_ID"] = master_df["Understat_ID"].fillna(-1).astype(int)

    if master_df.empty or not {"Unique_ID", "Understat_ID", "Full_Name"}.issubset(master_df.columns):
        log(f"Master file '{master_file}' is invalid or missing required columns.", "ERROR")
        return

    for season in seasons:
        season_id_col = f"{season[:4][-2:]}_id"
        players_dir = os.path.join(data_directory, season, "players")
        understat_dir = os.path.join(data_directory, season, "understat")

        if not os.path.isdir(players_dir) or not os.path.isdir(understat_dir):
            log(f"Missing directories for season {season}. Skipping.", "WARNING")
            continue

        player_folders = [
            folder for folder in os.listdir(players_dir) if os.path.isdir(os.path.join(players_dir, folder))
        ]
        understat_files = [
            file for file in os.listdir(understat_dir) if file.endswith(".csv")
        ]

        for _, player_row in master_df.iterrows():
            if player_row[season_id_col] == -1:
                log(f"Player '{player_row['Full_Name']}' not active in season '{season}' (ID is -1). Skipping.", "DEBUG")
                continue

            unique_id = str(player_row["Unique_ID"])
            understat_id = str(player_row["Understat_ID"])
            full_name = player_row["Full_Name"]

            matching_folder = next((folder for folder in player_folders if folder.endswith(f"_{unique_id}")), None)
            if not matching_folder:
                log(f"Player folder for Unique_ID '{unique_id}' not found in season '{season}'.", "INFO")
                continue

            gw_file = os.path.join(players_dir, matching_folder, "gw.csv")
            if not os.path.exists(gw_file):
                log(f"GW file missing for player folder '{matching_folder}' in season '{season}'.", "WARNING")
                continue

            gw_data = load_csv(gw_file)

            if gw_data.empty:
                log(f"Empty GW data for player {unique_id}. Skipping.", "WARNING")
                continue

            # Default values
            gw_data["Full_Name"] = full_name
            gw_data["Unique_ID"] = unique_id
            gw_data["shots"] = 0
            gw_data["expected_goals"] = 0
            gw_data["expected_assists"] = 0
            gw_data["key_passes"] = 0
            gw_data["npg"] = 0
            gw_data["npxG"] = 0

            matching_understat_file = next((file for file in understat_files if file.endswith(f"_{understat_id}.csv")), None)

            if matching_understat_file:
                understat_file = os.path.join(understat_dir, matching_understat_file)
                understat_data = load_csv(understat_file)

                if not understat_data.empty:
                    understat_data["date"] = pd.to_datetime(understat_data["date"]).dt.date
                    gw_data["kickoff_time"] = pd.to_datetime(gw_data["kickoff_time"], utc=True).dt.date
                    merged = pd.merge(
                        gw_data,
                        understat_data[["date", "shots", "xG", "xA", "key_passes", "npg", "npxG"]],
                        left_on="kickoff_time",
                        right_on="date",
                        how="left",
                        suffixes=("", "_understat")
                    )
                    # Update values from Understat
                    gw_data["shots"] = merged["shots_understat"].fillna(0).astype(int)
                    gw_data["expected_goals"] = merged["xG"].fillna(0)
                    gw_data["expected_assists"] = merged["xA"].fillna(0)
                    gw_data["key_passes"] = merged["key_passes_understat"].fillna(0).astype(int)
                    gw_data["npg"] = merged["npg_understat"].fillna(0).astype(int)
                    gw_data["npxG"] = merged["npxG_understat"].fillna(0)

            if dry_run:
                log(f"[Dry Run] Would update GW file for player '{full_name}' (Unique_ID {unique_id}) in season '{season}'.", "INFO")
            else:
                save_csv(gw_data, gw_file)
                log(f"Updated GW file for player folder '{matching_folder}' in season '{season}' with Full_Name and Understat data.", "DEBUG")

data_directory = "Fantasy-Premier-League/data"
master_file = "master_player_v2.csv"
seasons = ["2022-23", "2023-24", "2024-25"]

update_gw_files_with_understat(data_directory, master_file, seasons, dry_run=False)


# In[17]:


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


# In[18]:


# Let's merge understat files for teams across seasons and calculate some rolling stats using them
def merge_understat_team_data(data_directory, seasons, master_team_list, output_dir="teams"):
    """
    Merge understat data for each team across all seasons and compute rolling averages.

    Args:
        data_directory (str): Base directory containing the season data.
        seasons (list): List of season names (e.g., ["2023-24", "2022-23", ...]).
        master_team_list (pd.DataFrame): DataFrame containing the master team list with Unique_Id and Understat_Team_Name.
        output_dir (str): Directory to save merged team files.

    Returns:
        None
    """
    output_path = os.path.join(data_directory, output_dir)
    os.makedirs(output_path, exist_ok=True)

    for _, team_row in master_team_list.iterrows():
        unique_id = team_row["Unique_Id"]
        understat_name = team_row["Understat_Team_Name"].replace(" ", "_")

        team_data = []

        for season in seasons:
            season_short = season[:4][-2:]
            if team_row[f"{season_short}_Id"] == -1:
                # Skip teams not active in this season
                continue

            understat_file_path = os.path.join(data_directory, season, "understat", f"understat_{understat_name}.csv")
            if not os.path.exists(understat_file_path):
                log(f"Understat file not found: {understat_file_path}. Skipping {understat_name} in {season}.", level="WARNING")
                continue

            season_data = load_csv(understat_file_path)
            if season_data is None or season_data.empty:
                log(f"Invalid or empty Understat file: {understat_file_path}. Skipping.", level="WARNING")
                continue

            # Convert date to ensure consistency across seasons
            season_data["date"] = pd.to_datetime(season_data["date"]).dt.date
            team_data.append(season_data)

        if not team_data:
            log(f"No data available for team {understat_name} across seasons. Skipping.", level="INFO")
            continue

        # Combine data from all seasons and sort by date
        merged_data = pd.concat(team_data, ignore_index=True)
        merged_data.sort_values("date", inplace=True)

        # Compute rolling averages for selected stats
        rolling_windows = [4, 16]
        stats_to_roll = ["xG", "deep", "xGA", "deep_allowed"]

        for window in rolling_windows:
            for stat in stats_to_roll:
                rolling_col = f"{stat}_rolling_{window}"
                merged_data[rolling_col] =  merged_data[stat].shift(1).rolling(window, min_periods=1).mean()
                merged_data[rolling_col] = merged_data[rolling_col].fillna(merged_data[stat].head(5).mean())
                
        output_file_path = os.path.join(output_path, f"team_{unique_id}.csv")
        save_csv(merged_data, output_file_path)
        log(f"Saved merged data with rolling averages for team {understat_name} to {output_file_path}.", level="DEBUG")

data_directory = "Fantasy-Premier-League/data"
seasons = ["2024-25", "2023-24", "2022-23"]
master_team_list_file = "master_team_list_v2.csv"
output_dir = "teams"
master_team_list_path = os.path.join(data_directory, master_team_list_file)
master_team_list = load_csv(master_team_list_path)

merge_understat_team_data(data_directory, seasons, master_team_list, output_dir=output_dir)


# In[19]:


# Let's make use of understat data. In order to do that we have to update team ids in fixtures.csv
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

data_directory = "Fantasy-Premier-League/data"
seasons = ["2024-25", "2023-24", "2022-23"]
master_team_list_file = "master_team_list_v2.csv"
master_team_list_path = os.path.join(data_directory, master_team_list_file)
master_team_list = load_csv(master_team_list_path)
update_fixtures_file_ids(data_directory, seasons, master_team_list)


# In[20]:


def update_fixtures_with_rolling_stats(data_directory, seasons, master_team_list):
    """
    Update fixtures.csv with rolling stats from merged Understat data.

    Args:
        data_directory (str): Base directory containing the season data.
        seasons (list): List of season names to update (e.g., ["2023-24", "2022-23"]).
        master_team_list (pd.DataFrame): DataFrame containing the master team list.
    """
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
            unique_id = team_row["Unique_Id"]
            rolling_stats_file = os.path.join(data_directory, "teams", f"team_{unique_id}.csv")

            if not check_file_exists(rolling_stats_file):
                log(f"Rolling stats file not found for team {team_row['Understat_Team_Name']}. Skipping.", level="WARNING")
                continue

            rolling_stats_data = load_csv(rolling_stats_file)
            if rolling_stats_data is None or "date" not in rolling_stats_data.columns:
                log(f"Invalid rolling stats file: {rolling_stats_file}. Skipping.", level="WARNING")
                continue

            # Convert date for matching
            rolling_stats_data["date"] = pd.to_datetime(rolling_stats_data["date"]).dt.date

            for match_date, date_filtered_fixtures in fixtures_data.groupby("kickoff_time"):
                rolling_stats_row = rolling_stats_data[rolling_stats_data["date"] == match_date]

                for _, fixture in date_filtered_fixtures.iterrows():
                    if fixture["team_a"] == unique_id:
                        team_column = "team_a"
                    elif fixture["team_h"] == unique_id:
                        team_column = "team_h"
                    else:
                        continue

                    if not rolling_stats_row.empty:
                        rolling_stats_row = rolling_stats_row.iloc[0]
                        for column in rolling_stats_row.index:
                            if "rolling" in column:
                                fixtures_data.at[fixture.name, f"{team_column}_{column}"] = rolling_stats_row[column]

        save_csv(fixtures_data, fixtures_file_path)
        log(f"Updated fixtures.csv for season {season}", level="INFO")
        
data_directory = "Fantasy-Premier-League/data"
seasons = ["2024-25", "2023-24", "2022-23"]
master_team_list_file = "master_team_list_v2.csv"
master_team_list_path = os.path.join(data_directory, master_team_list_file)
master_team_list = load_csv(master_team_list_path)

if master_team_list is not None and not master_team_list.empty:
    update_fixtures_with_rolling_stats(data_directory, seasons, master_team_list)
else:
    log("Master team list is missing or invalid. Cannot update fixtures.", level="ERROR")


# In[21]:


# Once we have that we have to go into each season folder and update:
# - opponent_team in gw.csv for each player (except 24-25)
# - id in teams.csv (except 24-25)

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


# In[22]:


# Players don't have their own teams in gw.csv so we add them from fixtures.csv
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

                if 'fixture' not in gw_data.columns or 'was_home' not in gw_data.columns:
                    log(f"Missing columns in {gw_file_path}. Skipping.", level="WARNING")
                    continue

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

                columns_to_drop = ['id', 'team_a', 'team_h']
                gw_data.drop(columns=[col for col in columns_to_drop if col in gw_data.columns], inplace=True)

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


# In[23]:


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


# In[24]:


# Add rolling stats in player gw.csv from fixtures files

def add_team_stats_to_gw(data_directory, season):
    """
    Add rolling stats (of both teams) to gw.csv files for each player.

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

    try:
        fixtures = load_csv(fixtures_path)
        if fixtures is None:
            log(f"Failed to load fixtures.csv for season {season}. Skipping.", level="ERROR")
            return

        rolling_columns = [col for col in fixtures.columns if "rolling" in col]

        if not rolling_columns:
            log(f"No rolling stats found in fixtures.csv for season {season}. Skipping.", level="WARNING")
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

                            for col in rolling_columns:
                                if was_home:
                                    if "team_h" in col:
                                        gw_data.at[idx, f"team_{col[7:]}"] = fixture_row[col]
                                    elif "team_a" in col:
                                        gw_data.at[idx, f"opponent_{col[7:]}"] = fixture_row[col]
                                else:
                                    if "team_a" in col:
                                        gw_data.at[idx, f"team_{col[7:]}"] = fixture_row[col]
                                    elif "team_h" in col:
                                        gw_data.at[idx, f"opponent_{col[7:]}"] = fixture_row[col]

                    save_csv(gw_data, gw_file_path)
                    log(f"Updated {gw_file_path} with all rolling team stats.", level="DEBUG")
                except Exception as e:
                    log(f"Error processing {gw_file_path}: {e}", level="ERROR")
    except Exception as e:
        log(f"Error processing fixtures.csv for season {season}: {e}", level="ERROR")


data_directory = "Fantasy-Premier-League/data"
seasons = ["2022-23", "2023-24", "2024-25"]

for season in seasons:
    log(f"Processing season: {season}", level="INFO")
    add_team_stats_to_gw(data_directory, season)


# In[25]:


# Let's add season number and position for player
def add_season_and_position_to_gw(data_directory, season):
    """
    Add the season year and position as columns to gw.csv files for each player.

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

        # Extract position from the folder name (before the first "_")
        position = player_folder.split("_")[0]

        if check_file_exists(gw_file_path):
            try:
                gw_data = load_csv(gw_file_path)
                if gw_data is None:
                    continue

                # Add season and position columns
                gw_data["season"] = season_year
                gw_data["position"] = position

                save_csv(gw_data, gw_file_path)
                log(f"Updated {gw_file_path} with season and position columns.", level="DEBUG")
            except Exception as e:
                log(f"Error processing {gw_file_path}: {e}", level="ERROR")

data_directory = "Fantasy-Premier-League/data"
seasons = ["2022-23", "2023-24", "2024-25"]

for season in seasons:
    log(f"Processing season: {season}", level="INFO")
    add_season_and_position_to_gw(data_directory, season)


# In[26]:


def consolidate_player_data(data_directory, master_file, seasons, output_directory="consolidated_players"):
    """
    Consolidate gw.csv data for each player across seasons into a single CSV file.

    Args:
        data_directory (str): Base directory containing the season data.
        master_file (str): Path to the master player file (e.g., master_player_v2.csv).
        seasons (list): List of seasons to process (e.g., ["2022-23", "2023-24"]).
        output_directory (str): Directory to save the consolidated player CSVs.

    Returns:
        None
    """
    master_path = os.path.join(data_directory, master_file)
    master_data = load_csv(master_path)

    if master_data is None or master_data.empty:
        log("Master player file is missing or empty. Aborting.", level="ERROR")
        return

    output_path = os.path.join(data_directory, output_directory)
    os.makedirs(output_path, exist_ok=True)

    # Process each player in the master file
    for _, player_row in master_data.iterrows():
        first_name = player_row["First_Name"]
        last_name = player_row["Last_Name"]
        unique_id = player_row["Unique_ID"]

        consolidated_data = []

        for season in seasons:
            season_short = season[:4][-2:]
            season_player_id = player_row.get(f"{season_short}_id", -1)

            # Skip players not in the season
            if season_player_id == -1:
                continue

            season_path = os.path.join(data_directory, season, "players")
            if not check_file_exists(season_path):
                log(f"Players folder not found for season {season}. Skipping.", level="WARNING")
                continue

            player_folder = next(
                (folder for folder in os.listdir(season_path) if folder.endswith(f"_{unique_id}")), None
            )
            if not player_folder:
                log(f"Player folder for {first_name} {last_name} (Unique_ID {unique_id}) not found in {season}.", level="WARNING")
                continue

            gw_file_path = os.path.join(season_path, player_folder, "gw.csv")
            if not check_file_exists(gw_file_path):
                log(f"gw.csv not found for {first_name} {last_name} (Unique_ID {unique_id}) in {season}.", level="WARNING")
                continue

            try:
                gw_data = load_csv(gw_file_path)
                if gw_data is not None:
                    consolidated_data.append(gw_data)
            except Exception as e:
                log(f"Error processing gw.csv for {first_name} {last_name} (Unique_ID {unique_id}) in {season}: {e}", level="ERROR")

        # Save consolidated data for the player
        if consolidated_data:
            player_filename = f"{first_name}_{last_name}_{unique_id}.csv".replace(" ", "_")
            output_file_path = os.path.join(output_path, player_filename)
            consolidated_df = pd.concat(consolidated_data, ignore_index=True)
            save_csv(consolidated_df, output_file_path)
            log(f"Consolidated data saved for {first_name} {last_name} (Unique_ID {unique_id}) to {output_file_path}.", level="DEBUG")


data_directory = "Fantasy-Premier-League/data"
master_file = "master_player_v2.csv"
seasons = ["2022-23", "2023-24", "2024-25"]

consolidate_player_data(data_directory, master_file, seasons)


# In[27]:


def generate_sorted_training_data(data_directory, seasons, output_directory="training_data"):
    """
    Generate position-specific training data by consolidating and sorting gw.csv data for all seasons.

    Args:
        data_directory (str): Base directory containing the season data.
        seasons (list): List of season names (e.g., ["2022-23", "2023-24"]).
        output_directory (str): Directory to save the consolidated training data.

    Returns:
        None
    """
    output_path = os.path.join(data_directory, output_directory)
    os.makedirs(output_path, exist_ok=True)

    position_dfs = {
        "DEF": pd.DataFrame(),
        "MID": pd.DataFrame(),
        "FWD": pd.DataFrame(),
        "GK": pd.DataFrame(),
    }

    for season in seasons:
        log(f"Processing season: {season}", level="INFO")
        players_folder = os.path.join(data_directory, season, "players")
        if not check_file_exists(players_folder, log_missing=True):
            continue

        for player_folder in os.listdir(players_folder):
            position = player_folder.split("_")[0]

            player_folder_path = os.path.join(players_folder, player_folder)
            gw_file_path = os.path.join(player_folder_path, "gw.csv")

            if check_file_exists(gw_file_path):
                try:
                    gw_data = load_csv(gw_file_path)
                    if gw_data is not None and position in position_dfs:
                        position_dfs[position] = pd.concat(
                            [position_dfs[position], gw_data], ignore_index=True
                        )
                except Exception as e:
                    log(f"Error processing {gw_file_path}: {e}", level="ERROR")

    for position, df in position_dfs.items():
        if not df.empty:
            df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True, format="mixed")
            df.sort_values(by=["Unique_ID", "season", "kickoff_time"], inplace=True)

            output_file_path = os.path.join(output_path, f"{position}_players.csv")
            save_csv(df, output_file_path)
            log(f"Saved sorted training data for {position} to {output_file_path}.")

data_directory = "Fantasy-Premier-League/data"
seasons = ["2022-23", "2023-24", "2024-25"]

generate_sorted_training_data(data_directory, seasons)


# In[ ]:




