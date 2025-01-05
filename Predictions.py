#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import xgboost as xgb
from helpers import load_csv, save_csv, log, check_file_exists, load_and_merge_prediction_data, preprocess_prediction_data, make_predictions


# In[2]:


def save_predictions(data, output_columns, prediction_column, output_file):
    """Format and save predictions to a CSV file."""
    output_data = data[output_columns + [prediction_column]].copy()
    output_data["gameweek"] = output_data["gameweek"].astype(int)
    
    # Check for duplicates
    duplicates = output_data.duplicated(subset=["unique_id", "first_name", "second_name", "gameweek"], keep=False)
    if duplicates.any():
        print("Duplicate rows detected:")
        print(output_data[duplicates])
    
    # Pivot the data to spread gameweek values into columns
    output_data = output_data.pivot(index=["unique_id", "first_name", "second_name"], columns="gameweek", values=prediction_column)
    output_data.reset_index(inplace=True)

    # Ensure all column names are strings
    output_data.columns = output_data.columns.map(str)

    # Rename gameweek columns to include "_Pts"
    output_data.rename(columns=lambda col: f"{col}_Pts" if col.isdigit() else col, inplace=True)

    # Rename unique_id column for clarity
    output_data.rename(columns={"unique_id": "ID"}, inplace=True)

    # Sort gameweek columns
    id_columns = ["ID", "first_name", "second_name"]
    gameweek_columns = sorted(
        [col for col in output_data.columns if col.endswith("_Pts")],
        key=lambda x: int(x.split('_')[0])  # Extract numerical gameweek for sorting
    )
    output_data = output_data[id_columns + gameweek_columns]

    # Sort rows by gameweek column values (optional)
    if gameweek_columns:
        output_data = output_data.sort_values(by=gameweek_columns, ascending=False)

    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


# In[3]:


data_directory = "Fantasy-Premier-League/data/2024-25"
positions = ["GK", "DEF", "MID", "FWD"]
models_path = "models"
output_directory = "predictions"

os.makedirs(output_directory, exist_ok=True)
all_predictions = []

for position in positions:
    print(f"Processing position: {position}")
    merged_data = load_and_merge_prediction_data(data_directory, [position])
    merged_data = preprocess_prediction_data(merged_data)
    
    model_path = os.path.join(models_path, f"{position.lower()}_prediction_model.json")
    output_file = os.path.join(output_directory, f"{position.lower()}_points.csv")
    
    # Make predictions
    predicted_data = make_predictions(merged_data, model_path, "Pts")
    if "pred_minutes" not in merged_data.columns:
        raise ValueError("Column 'pred_minutes' is missing from merged_data.")
    predicted_data["Pts"] = predicted_data["Pts"] * (merged_data["pred_minutes"] / 90)
    
    # Save individual position predictions
    save_predictions(predicted_data, ["unique_id", "first_name", "second_name", "gameweek"], "Pts", output_file)
    
    # Append the predictions dataframe for merging
    all_predictions.append(predicted_data)

# Combine all position dataframes into one
combined_predictions = pd.concat(all_predictions, ignore_index=True)

# Save the combined file
combined_output_file = os.path.join(output_directory, "all_positions_points.csv")
save_predictions(combined_predictions, ["unique_id", "first_name", "second_name", "gameweek"], "Pts", combined_output_file)

print(f"Combined predictions saved to {combined_output_file}")


# In[ ]:




