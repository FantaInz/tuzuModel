import os
import shutil

import pandas as pd
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
            position_file_path = os.path.join(season_folder, position_file)

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
            output_file_path = os.path.join(training_data_folder, output_file_name)
            save_csv(combined_df, output_file_path)
            log(f"Combined data saved to {output_file_path}.")
        except Exception as e:
            log(f"Error combining data: {e}", level="ERROR")
    else:
        log("No data found for the specified positions in the given seasons.", level="WARNING")

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