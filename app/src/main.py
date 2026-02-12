import os
import json
import matplotlib.pyplot as plt

with open("app/data/config.json", "r") as file:
    config = json.load(file)
    os.environ["SEASON_PATH"] = os.path.join(config["DATA_DOWNLOAD_PATH"], "seasons")
with open(config["KAGGLE_API_TOKEN_PATH"], "r") as file:
    os.environ["KAGGLE_API_TOKEN"] = file.read()

from download_and_sort_data import download_and_sort_data  # this import has to come last


# DEBUG IMPORTS
import pickle
import numpy as np
import pandas as pd

# TODO delete debug imports

if __name__ == "__main__":
    print("----- WELCOME TO THE BASKETBALL COMPUTER THINGY -----")
    download_and_sort_data(config)

    # debug
    test_year = "2025-2026"  # "1946-1947" "1995-1996" "2025-2026"
    with open(f"app\\data\\games\\seasons\\{test_year}\\{test_year}_season.pkl", "rb") as file:
        test_season = pickle.load(file)
    test_season.pretty_print()

    ref_data = pd.concat(
        [pd.read_csv(os.path.join(os.environ["SEASON_PATH"], dir, f"{dir}_full.csv")) for dir in os.listdir(os.environ["SEASON_PATH"])],
        ignore_index=True,
    )

    plt.figure(1)
    data = ref_data["GAME_total"].to_numpy()
    plt.hist(data, bins=(max(data) - min(data)))
    plt.title("Total Score of all NBA Games")

    plt.figure(2)
    data = ref_data["GAME_spread"].to_numpy()
    plt.hist(data, bins=(max(data) - min(data)))
    plt.title("Home Team Spread of all NBA Games")

    plt.figure(3)
    data = ref_data["GAME_homeWin"].to_numpy()
    plt.hist(data, bins=2)
    plt.title("Home vs Away all NBA games")

    plt.show()
