import os
import shutil

import pandas as pd
import xgboost as xgb
import numpy as np
from git import Repo

# ========================
# Logging Helpers
# ========================

def log(message, level="INFO", verbosity="INFO"):
    """
    Log messages with a specified log level, given the verbosity level.
    
    Args:
        message (str): The message to log.
        level (str): The level of the log message ("ERROR", "WARNING", "INFO", "DEBUG").
        verbosity (str): The minimum verbosity level for messages to be logged.
    """
    levels = ["ERROR", "WARNING", "INFO", "DEBUG"]
    if levels.index(level) <= levels.index(verbosity):
        print(f"{level}: {message}")

# ========================
# File and Directory Helpers
# ========================

def check_file_exists(file_path, log_missing=False):
    """
    Check if a file or directory exists.

    Args:
        file_path (str): Path to the file or directory.
        log_missing (bool): Whether to log a message if the file is missing.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    exists = os.path.exists(file_path)
    if not exists and log_missing:
        log(f"File not found: {file_path}", level="WARNING")
    return exists

def remove_file_or_dir(path):
    """
    Remove a file or directory if it exists.
    """
    try:
        if os.path.isfile(path):
            os.remove(path)
            log(f"Deleted file: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            log(f"Deleted directory: {path}")
    except Exception as e:
        log(f"Unable to delete {path}: {e}", level="ERROR")

def filter_directory(directory, keep_files=None, keep_dirs=None):
    """
    Remove all files and subdirectories in a directory except those specified.

    Args:
        directory (str): Path to the directory to filter.
        keep_files (list): List of filenames to keep.
        keep_dirs (list): List of subdirectory names to keep.
    """
    if not os.path.exists(directory):
        log(f"Directory {directory} does not exist. Skipping.", level="WARNING")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if (keep_files and item in keep_files) or (keep_dirs and item in keep_dirs):
            continue

        remove_file_or_dir(item_path)

# ========================
# Data Helpers
# ========================

def load_csv(file_path):
    """Load a CSV file and handle exceptions."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        log(f"Error loading CSV {file_path}: {e}", level="ERROR")
        return None

def save_csv(df, file_path):
    """Save a DataFrame to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
        log(f"Saved CSV: {file_path}")
    except Exception as e:
        log(f"Error saving CSV {file_path}: {e}", level="ERROR")

def combine_position_data(data_directory, seasons, positions, output_file_name):
    """
    Combines data from specified positions for given seasons into a single dataset for predictions.

    Args:
        data_directory (str): Base directory containing the season data.
        seasons (list): List of season names (e.g., ["2022-23", "2023-24"]).
        positions (list): List of positions to combine (e.g., ["DEF", "MID", "FWD"]).
        output_file_name (str): Name of the combined output file (e.g., "training_data.csv").

    Returns:
        None
    """
    training_data_folder = os.path.join(data_directory, "training_data")
    os.makedirs(training_data_folder, exist_ok=True)

    combined_data = []

    for season in seasons:
        season_folder = os.path.join(data_directory, season)
        if not check_file_exists(season_folder):
            log(f"Skipping season {season} as folder does not exist.", level="WARNING")
            continue

        # Iterate through position files
        for position in positions:
            position_file = f"{position}_players.csv"
            position_file_path = os.path.join(season_folder, "processed_data", position_file)

            if check_file_exists(position_file_path):
                position_data = load_csv(position_file_path)
                if position_data is not None:
                    combined_data.append(position_data)
                    log(f"Loaded data from {position_file} for season {season}.")
            else:
                log(f"{position_file} not found for season {season}. Skipping.", level="WARNING")

    # Combine all data into a single DataFrame
    if combined_data:
        try:
            combined_df = pd.concat(combined_data, ignore_index=True)
            if {"unique_id", "season", "gameweek"}.issubset(combined_df.columns):
                combined_df.sort_values(by=["unique_id", "season", "gameweek"], inplace=True)
            else:
                log("One or more required columns ('unique_id', 'season', 'gameweek') missing. Sorting skipped.", level="WARNING")
            output_file_path = os.path.join(training_data_folder, output_file_name)
            save_csv(combined_df, output_file_path)
            log(f"Combined data saved to {output_file_path}.")
        except Exception as e:
            log(f"Error combining data: {e}", level="ERROR")
    else:
        log("No data found for the specified positions in the given seasons.", level="WARNING")

def calculate_season_average_until_gw(data, value_column, group_columns, current_column="gameweek"):
    """
    Calculate the season average of a specific value (e.g., xG) until the current gameweek.
    
    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        value_column (str): The column to calculate the average for (e.g., 'expected_goals').
        group_columns (list): The columns to group by (e.g., ['unique_id', 'season']).
        current_column (str): The column to compare for gameweeks (default is 'gameweek').

    Returns:
        pd.Series: A Series containing the average values until the current gameweek.
    """
    # Sort the data by the specified grouping columns and the current column
    data = data.sort_values(by=group_columns + [current_column])

    # Calculate expanding mean for each group and shift to exclude the current gameweek
    averages = (
        data.groupby(group_columns)[value_column]
        .expanding()
        .mean()
        .shift()
        .reset_index(level=group_columns, drop=True)
    )

    # Fill NaN values for the first gameweek with 0
    return averages.fillna(0)

# ========================
# Prediction Helpers
# ========================

def load_and_merge_prediction_data(data_directory, positions):
    """Load and merge data for specified positions."""
    merged_data = pd.DataFrame()
    for position in positions:
        position_file = os.path.join(data_directory, "processed_data", position, f"{position}_final.csv")
        if os.path.exists(position_file):
            position_data = pd.read_csv(position_file)
            if position_data is not None:
                merged_data = pd.concat([merged_data, position_data], ignore_index=True)
                print(f"Merged {position_file}")
        else:
            print(f"File {position_file} not found. Skipping.")
    return merged_data

def preprocess_prediction_data(data):
    """Perform data preprocessing and feature engineering."""
    # Rename "id" to "unique_id" for consistency
    data.rename(columns={"id": "unique_id"}, inplace=True)

    # Feature engineering
    data['was_home'] = data['was_home'].astype(int)
    data["_unique_id_copy"] = data["unique_id"]
    data["_pos_copy"] = data["POS"]

    # Sort for rolling and cumulative calculations
    data = data.sort_values(by=["unique_id", "season", "gameweek"])

    # One-hot encoding for categorical columns
    dummy_columns = ["POS", "was_home", "unique_id", "own_team", "opponent_team"]
    data = pd.get_dummies(data, columns=dummy_columns)

    data["unique_id"] = data["_unique_id_copy"]
    data["POS"] = data["_pos_copy"]
    data.drop(columns=["_unique_id_copy"], inplace=True)

    return data

def make_predictions(data, model_path, prediction_column):
    """Load model, generate predictions, and add to data."""
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    print(f"Loaded model from {model_path}")

    # Ensure only trained features are used
    trained_feature_names = model.get_booster().feature_names
    prediction_data = data.reindex(columns=trained_feature_names, fill_value=0)

    # Generate predictions
    predictions = model.predict(prediction_data)
    predictions = np.clip(predictions, a_min=0, a_max=None)  # Clamp negative predictions to 0
    data[prediction_column] = predictions
    return data

def save_predictions(data, output_columns, prediction_column, output_file):
    """Format and save predictions to a CSV file."""
    output_data = data[output_columns + [prediction_column]]
    output_data.loc[:, "gameweek"] = output_data["gameweek"].astype(int)  # Ensure gameweek is an integer
    output_data = output_data.pivot(index=["unique_id", "first_name", "second_name"], columns="gameweek", values=prediction_column)
    output_data.reset_index(inplace=True)

    # Rename columns for clarity
    output_data.columns = [
        f"gw_{col}_{prediction_column}" if isinstance(col, int) else col for col in output_data.columns
    ]

    # Sort by gameweek columns
    gameweek_columns = sorted(
        [col for col in output_data.columns if col.startswith("gw_")],
        key=lambda x: int(x.split('_')[1])  # Extract the gameweek number for proper sorting
    )
    output_data = output_data.sort_values(by=gameweek_columns, ascending=False)

    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# ========================
# Git Helpers
# ========================

def git_clone(repo_url, clone_dir):
    """
    Clone a Git repository to a specified directory, removing the directory first if it exists.
    
    Args:
        repo_url (str): URL of the repository to clone.
        clone_dir (str): Directory to clone the repository into.
    """
    remove_file_or_dir(clone_dir)
    try:
        Repo.clone_from(repo_url, clone_dir)
        log(f"Cloned repository to: {clone_dir}")
    except Exception as e:
        log(f"Error cloning repository: {e}", level="ERROR")