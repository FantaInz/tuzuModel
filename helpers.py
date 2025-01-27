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

def load_csv(file_path, low_memory=False):
    """Load a CSV file and handle exceptions."""
    try:
        return pd.read_csv(file_path, low_memory=low_memory)
    except Exception as e:
        log(f"Error loading CSV {file_path}: {e}", level="DEBUG")
        return None

def save_csv(df, file_path):
    """Save a DataFrame to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
        log(f"Saved CSV: {file_path}", level="DEBUG")
    except Exception as e:
        log(f"Error saving CSV {file_path}: {e}", level="ERROR")


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