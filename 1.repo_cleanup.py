#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from helpers import git_clone, remove_file_or_dir, filter_directory, log


# In[2]:


def clean_directory(directory):
    """
    Clean the Fantasy-Premier-League directory of unnecessary data.
    """
    if not os.path.exists(directory):
        log(f"Directory {directory} doesn't exist.", level="WARNING")
        return

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if item == "data":
            # Process "data" subdirectory
            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)

                # Remove too far seasons
                if sub_item not in ["2022-23", "2023-24", "2024-25"]:
                    remove_file_or_dir(sub_item_path)
                elif os.path.isdir(sub_item_path):
                    filter_directory(
                        sub_item_path,
                        keep_files=["fixtures.csv", "player_idlist.csv", "cleaned_players.csv", "players_raw.csv", "teams.csv", 
                                    "id_dict.csv"],
                        keep_dirs=["players", "understat"]
                    )
        else:
            remove_file_or_dir(item_path)

repo_url = "https://github.com/vaastav/Fantasy-Premier-League.git"
clone_dir = "Fantasy-Premier-League"

log("Cloning the repository...", level="INFO")
git_clone(repo_url, clone_dir)

log("Cleaning the directory...", level="INFO")
clean_directory(clone_dir)


# In[ ]:




