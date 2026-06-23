import os
import json

with open("app/data/config.json", "r") as file:
    config = json.load(file)
    os.environ["SEASON_PATH"] = os.path.join(config["DATA_DOWNLOAD_PATH"], "seasons")
with open(config["KAGGLE_API_TOKEN_PATH"], "r") as file:
    os.environ["KAGGLE_API_TOKEN"] = file.read()

import pandas as pd
from Models import *
from functions import print_current_season
import statistics as stats
from download_and_sort_data import download_and_sort_data  # this import has to come last


def build_models_optim(config):
    # get reference data based on min year parameter
    ref_data = pd.concat(
        [
            pd.read_csv(os.path.join(os.environ["SEASON_PATH"], dir, f"{dir}_full.csv"))
            for dir in os.listdir(os.environ["SEASON_PATH"])
            if (int(dir.split("-")[0]) >= config["MIN_REFERENCE_DATA_YEAR"])
            and not ((int(dir.split("-")[0]) >= config["MIN_TEST_DATA_YEAR"]) and config["DEBUG"])
        ],
        ignore_index=True,
    )
    if config["DEBUG"]:
        test_data = pd.concat(
            [
                pd.read_csv(os.path.join(os.environ["SEASON_PATH"], dir, f"{dir}_full.csv"))
                for dir in os.listdir(os.environ["SEASON_PATH"])
                if (int(dir.split("-")[0]) >= config["MIN_TEST_DATA_YEAR"])
            ],
            ignore_index=True,
        )
        test_data = test_data[
            (test_data["HOME_games_played"] != 0)
            & (test_data["AWAY_games_played"] != 0)
            & (test_data["HOME_wins"] != 0)
            & (test_data["AWAY_wins"] != 0)
            & (test_data["HOME_home_wins"] != 0)
            & (test_data["AWAY_away_wins"] != 0)
            & (test_data["HOME_home_losses"] != 0)
            & (test_data["AWAY_away_losses"] != 0)
        ].reset_index()

    # calculate out the model
    # new weight function goes here
    MODEL_HOME_WIN_PR.calculate_model(ref_data)

    if config["DEBUG"]:
        pred_win = MODEL_HOME_WIN_PR.value(test_data, apply_mask=True)
        true_win = test_data["GAME_homeWin"].to_numpy()
        _, _, AUC = stats.calc_ROC_curve(pred_win, true_win)
        _, _, M, B = stats.calc_calibrated_slope_intercept(pred_win, true_win)
        print("AUC: ", AUC)
        print("Brier:", stats.calc_brier_score(pred_win, true_win))
        print("ECE:", stats.calc_ECE_score(pred_win, true_win))
        print("Slope:", M)
        print("Intercept:", B)


def main():
    print("----- WELCOME TO THE BASKETBALL COMPUTER THINGY -----")
    download_and_sort_data(config)  # donwload/sort raw data, if necessary
    build_models_optim(config)  # create plots that are passed as part of debug for model
    # TODO create optimization function that takes in the parameters we want to solve for (see model_stuff.txt) and uses something like scipy optimize


if __name__ == "__main__":
    main()
    print_current_season()
