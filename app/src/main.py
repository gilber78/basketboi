import os
import json
import matplotlib.pyplot as plt

with open("app/data/config.json", "r") as file:
    config = json.load(file)
with open(config["KAGGLE_API_TOKEN_PATH"], "r") as file:
    os.environ["KAGGLE_API_TOKEN"] = file.read()

from download_and_sort_data import download_and_sort_data  # this import has to come last


# DEBUG IMPORTS
import pickle
import pandas as pd

# TODO delete debug imports

if __name__ == "__main__":
    print("----- WELCOME TO THE BASKETBALL COMPUTER THINGY -----")
    download_and_sort_data(config)

    # debug
    test_year = "2025-2026"  # "1946-1947" "1995-1996" "2025-2026"
    with open(f"app\\data\\games\\seasons\\{test_year}\\{test_year}_season.pkl", "rb") as file:
        test_season = pickle.load(file)
    test_season_df = pd.read_csv(f"app\\data\\games\\seasons\\{test_year}\\{test_year}_full.csv")
    print(test_season_df)
    test_season.pretty_print()
