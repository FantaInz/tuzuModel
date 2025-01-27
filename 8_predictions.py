#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import joblib
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor


# In[2]:


data_directory = "Fantasy-Premier-League/data/prediction_files/"
models_directory = "models/"
output_directory = "predictions/"

input_files = {
    "gk": os.path.join(data_directory, "gk-ready.csv"),
    "def": os.path.join(data_directory, "def-ready.csv"),
    "mid": os.path.join(data_directory, "mid-ready.csv"),
    "fwd": os.path.join(data_directory, "fwd-ready.csv")
}

model_files = {
    "gk": os.path.join(models_directory, "gk_prediction_model.json"),
    "def": os.path.join(models_directory, "def_prediction_model.pkl"),
    "mid": os.path.join(models_directory, "mid_prediction_model.json"),
    "fwd": os.path.join(models_directory, "fwd_prediction_model.json")
}

output_files = {
    "gk": os.path.join(output_directory, "gk-predictions.csv"),
    "def": os.path.join(output_directory, "def-predictions.csv"),
    "mid": os.path.join(output_directory, "mid-predictions.csv"),
    "fwd": os.path.join(output_directory, "fwd-predictions.csv")
}

os.makedirs(output_directory, exist_ok=True)


# In[3]:


def load_model(model_path, model_type):
    if model_type == "pkl":
        with open(model_path, "rb") as f:
            return joblib.load(f)
    elif model_type == "json":
        model = CatBoostRegressor()
        model.load_model(model_path)
        return model
    else:
        raise ValueError(f"Unsupported model type for {model_path}")


# In[4]:


def make_predictions(input_file, model, output_file, position, selected_features=None):
    data = pd.read_csv(input_file)
    xMins = data["xMins"].fillna(0)

    if position == "def":
        categorical = ["own_team", "opponent_team", "was_home"]
        data[["own_team", "opponent_team"]] = data[["own_team", "opponent_team"]].astype(int)
        data[categorical] = data[categorical].astype("category")
        dummy = pd.get_dummies(data[categorical], drop_first=True)
        
        one_hot_teams = [col for col in data.columns if col.startswith("own_team_") or col.startswith("opponent_team_")]
        dummy = pd.get_dummies(data[categorical], drop_first=True)
        expected_order = (
            [f"own_team_{i}" for i in range(1, 25) if i != 1] +
            [f"opponent_team_{i}" for i in range(1, 25) if i != 1] +
            ["was_home_True"]
        )

        for col in expected_order:
            if col not in dummy.columns:
                dummy[col] = 0
        dummy = dummy[expected_order]

        features = pd.concat([data[selected_features], dummy], axis=1)
    else:
        try:
            selected_features = model.get_booster().feature_names
        except AttributeError:
            selected_features = model.feature_names_
        categorical = ["own_team", "opponent_team", "was_home"]
        data[["own_team", "opponent_team"]] = data[["own_team", "opponent_team"]].astype(int)
        data[categorical] = data[categorical].astype("category")
        features = data[selected_features]

    predictions = model.predict(features)
    adjusted_predictions = predictions * (xMins / 90)
    data["Pts"] = adjusted_predictions

    data = data.groupby(["Unique_ID", "Full_Name", "event"], as_index=False)["Pts"].sum()

    data["ID"] = data["Unique_ID"].astype(int)
    pivot_data = data.pivot(index=["ID", "Full_Name"], columns="event", values="Pts").reset_index()

    pivot_data.columns = [f"{int(col)}_Pts" if isinstance(col, (int, float)) else col for col in pivot_data.columns]

    first_event_col = [col for col in pivot_data.columns if col.endswith("_Pts")][0]
    pivot_data = pivot_data.sort_values(by=first_event_col, ascending=False)

    pivot_data.to_csv(output_file, index=False)
    print(f"Predictions saved to: {output_file}")


# In[5]:


for position, input_file in input_files.items():
    if not os.path.exists(input_file):
        print(f"Input file not found for {position}. Skipping.")
        continue

    model_file = model_files[position]
    if not os.path.exists(model_file):
        print(f"Model file not found for {position}. Skipping.")
        continue

    model_type = "pkl" if model_file.endswith(".pkl") else "json"
    model = load_model(model_file, model_type)

    if position == "def":
        selected_features = [
            'atk_def_diff', 'def_atk_diff', 'opponent_xG_rolling_4', 'opponent_deep_rolling_4',
            'opponent_xGA_rolling_4', 'opponent_deep_allowed_rolling_4', 'team_xG_rolling_4',
            'team_xGA_rolling_4', 'opponent_xGA_rolling_16', 'opponent_deep_allowed_rolling_16',
            'opponent_xG_rolling_16', 'opponent_deep_rolling_16', 'opponent_defense',
            'own_defense', 'team_xG_rolling_16', 'team_deep_rolling_16', 'team_xGA_rolling_16',
            'team_deep_allowed_rolling_16'
        ]
    else:
        selected_features = None

    output_file = output_files[position]
    make_predictions(input_file, model, output_file, position, selected_features)


# In[6]:


all_predictions = []
for position, output_file in output_files.items():
    if not os.path.exists(output_file):
        print(f"Output file not found for {position}. Skipping.")
        continue

    position_data = pd.read_csv(output_file)
    position_data["Position"] = position.upper()
    all_predictions.append(position_data)

all_positions = pd.concat(all_predictions, ignore_index=True)
point_columns = [col for col in all_positions.columns if col.endswith("_Pts")]
if point_columns:
    first_point_col = point_columns[0]
    all_positions = all_positions.sort_values(by=first_point_col, ascending=False)

all_positions_file = os.path.join(output_directory, "all_positions_points.csv")
all_positions.to_csv(all_positions_file, index=False)
print(f"All positions predictions saved to: {all_positions_file}")

