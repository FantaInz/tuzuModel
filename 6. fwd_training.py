#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from helpers import log, load_csv, save_csv
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.feature_selection import SelectKBest, mutual_info_regression
# import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.neural_network import MLPRegressor
# from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import seaborn as sns
# import optuna
from catboost import CatBoostRegressor


# In[2]:


# Load data and filter
data_directory = "Fantasy-Premier-League/data"
training_data_dir = os.path.join(data_directory, "training_data")
training_file = os.path.join(training_data_dir, "FWD_players.csv")
fwd_data = load_csv(training_file)
fwd_data = fwd_data[fwd_data["minutes"] >= 60]
fwd_data["atk_def_diff"] = fwd_data["own_attack"] - fwd_data["opponent_defense"]


# In[3]:


# Plot points
sns.set_style("whitegrid")

plt.figure(figsize=(14, 7))

min_points = int(fwd_data['total_points'].min())
max_points = int(fwd_data['total_points'].max())
plt.figure(figsize=(14, 7))
ax = sns.histplot(fwd_data['total_points'], bins=range(min_points, max_points + 2), discrete=True, color='skyblue', edgecolor='black')

for patch in ax.patches:
    height = patch.get_height()
    if height > 0: 
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 10,
            f"{int(height)}",
            ha="center",
            fontsize=14,
            color="black"
        )

plt.xlabel("Liczba punktów", fontsize=14)
plt.ylabel("Liczność", fontsize=14)

plt.xticks(ticks=range(min_points, max_points + 1), fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()


# In[4]:


stats_points = fwd_data["total_points"].describe()
variance = fwd_data["total_points"].var()
median_points = fwd_data["total_points"].median()

print("Statystyki punktów napastników:")
print(stats_points)
print(f"Wariancja: {variance:.2f}")
print(f"Mediana punktów zdobytych przez napastników: {median_points}")


# In[5]:


rolling_candidates = [
    "shots", "key_passes", "expected_goals", "expected_assists", "goals_scored",
    "bps", "ict_index", "influence", "creativity", "threat", "assists", "total_points"
]
rolling_periods = [4, 16]

for feature in rolling_candidates:
    for period in rolling_periods:
        sma_column = f"{feature}_rolling_{period}"
        group = fwd_data.groupby("Unique_ID")[feature]
        
        fwd_data[sma_column] = group.apply(
            lambda x: x.shift(1).rolling(window=period, min_periods=1).mean()
        ).reset_index(level=0, drop=True)

for feature in rolling_candidates:
    for period in rolling_periods:
        sma_column = f"{feature}_rolling_{period}"
        fwd_data[sma_column] = fwd_data[sma_column].bfill()

other_available = [
    "value", "atk_def_diff", "own_attack", "opponent_defense",
    'opponent_xGA_rolling_4', 'opponent_deep_allowed_rolling_4',
    'opponent_xGA_rolling_16', 'opponent_deep_allowed_rolling_16',
    'team_xG_rolling_4', 'team_deep_rolling_4', 'team_xG_rolling_16', 'team_deep_rolling_16'
]


# In[6]:


# features = fwd_data[
#     [f"{col}_rolling_{period}" for col in rolling_candidates for period in rolling_periods] +
#     other_available
# ]
# target = fwd_data["total_points"]  # Zmienna docelowa

# select_k_best = SelectKBest(score_func=lambda X, y: mutual_info_regression(X, y, random_state=42), k=20)
# pipeline = Pipeline([
#     ("select_k_best", select_k_best)
# ])

# pipeline.fit(features, target)

# all_feature_names = features.columns.tolist()

# selected_features = pipeline.named_steps["select_k_best"].get_support(indices=True)

# best_feature_names = [all_feature_names[i] for i in selected_features]
# selected_features = best_feature_names
# print(best_feature_names)


# In[7]:


categorical = ["own_team", "opponent_team", "was_home"]
fwd_data[categorical] = fwd_data[categorical].astype("category")
selected_features = ['shots_rolling_4', 'expected_goals_rolling_4', 'expected_assists_rolling_16', 'goals_scored_rolling_4', 'goals_scored_rolling_16', 
     'bps_rolling_16', 'ict_index_rolling_4', 'ict_index_rolling_16', 'influence_rolling_4', 'threat_rolling_16', 'assists_rolling_4', 
     'assists_rolling_16', 'total_points_rolling_16', 'own_attack', 'opponent_defense', 'opponent_xGA_rolling_4', 
     'opponent_deep_allowed_rolling_4', 'opponent_deep_allowed_rolling_16', 'team_xG_rolling_4', 'team_xG_rolling_16']

X = pd.concat([fwd_data[selected_features], fwd_data[categorical]], axis=1)
y = fwd_data["total_points"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = CatBoostRegressor(
        iterations=445, 
        learning_rate=0.015152378509149479, 
        depth=3, 
        l2_leaf_reg=0.541102028124743,
        bagging_temperature=6.405609751764768, 
        subsample=0.8630987194825457, 
        random_strength=5.299332390252302,
        random_seed=42,
        verbose=0,
        cat_features=categorical,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)
rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

importances = model.get_feature_importance(type='PredictionValuesChange')
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

top_8_features = feature_importances.head(8)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_8_features.values, y=top_8_features.index, palette="viridis")
plt.xlabel('Ważność')
plt.ylabel('Cechy')
plt.show()

models_folder = "models"
os.makedirs(models_folder, exist_ok=True)
model_path = os.path.join(models_folder, "fwd_prediction_model.json")
model.save_model(model_path)
print(f"Model saved at: {model_path}")


# In[ ]:




