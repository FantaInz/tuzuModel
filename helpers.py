import os
import pandas as pd

def combine_position_data(data_directory, seasons, positions, output_file_name):
    """
    Combines data from specified positions for given seasons into a single dataset for xG prediction.

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
        if not os.path.exists(season_folder):
            print(f"Season folder not found for {season}. Skipping.")
            continue

        # Iterate through position files
        for position in positions:
            position_file = f"{position}_players.csv"
            position_file_path = os.path.join(season_folder, position_file)

            if os.path.exists(position_file_path):
                try:
                    position_data = pd.read_csv(position_file_path)
                    combined_data.append(position_data)
                    print(f"Loaded data from {position_file} for season {season}.")
                except Exception as e:
                    print(f"Error loading {position_file_path}: {e}")
            else:
                print(f"{position_file} not found for season {season}. Skipping.")

    # Combine all data into a single DataFrame
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        output_file_path = os.path.join(training_data_folder, output_file_name)
        combined_df.to_csv(output_file_path, index=False)
        print(f"Combined data saved to {output_file_path}.")
    else:
        print("No data found for the specified positions in the given seasons.")