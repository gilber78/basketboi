import os
import json

with open(os.path.join("app", "data", "config.json"), "r") as file:
    config = json.load(file)
    os.environ["SEASON_PATH"] = os.path.join(config["DATA_DOWNLOAD_PATH"], "seasons")
with open(config["KAGGLE_API_TOKEN_PATH"], "r") as file:
    os.environ["KAGGLE_USERNAME"] = file.readline()
    os.environ["KAGGLE_KEY"] = file.readline()

import pandas as pd
import statistics as stats
from functools import partial
from scipy.optimize import minimize

from Models import *
from functions import print_current_season
from download_and_sort_data import download_and_sort_data  # this import has to come last


def objective_function(x):
    # split the vector into compontnes
    min_reference_data_year = np.round(x[0])
    z = x[1]
    b = x[2]
    print(min_reference_data_year, z, b)

    # get reference data based on min year parameter
    ref_data = pd.concat(
        [
            pd.read_csv(os.path.join(os.environ["SEASON_PATH"], dir, f"{dir}_full.csv"))
            for dir in os.listdir(os.environ["SEASON_PATH"])
            if (int(dir.split("-")[0]) >= min_reference_data_year)
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
    MODEL_HOME_WIN_PR.weight_func = partial(recency_weight_function, z=z, b=b)
    MODEL_HOME_WIN_PR.calculate_model(ref_data)

    if config["DEBUG"]:
        pred_win = MODEL_HOME_WIN_PR.value(test_data, apply_mask=True)
        true_win = test_data["GAME_homeWin"].to_numpy()
        _, _, AUC = stats.calc_ROC_curve(pred_win, true_win)
        BRIER = stats.calc_brier_score(pred_win, true_win)
        ECE = stats.calc_ECE_score(pred_win, true_win)
        _, _, M, B = stats.calc_calibrated_slope_intercept(pred_win, true_win)
        """
        print("AUC: ", AUC)
        print("Brier:", BRIER)
        print("ECE:", ECE)
        print("Slope:", M)
        print("Intercept:", B)
        """
        return 100 * (M - 1) ** 2 + 100 * B**2 + 25 * ECE**2 + 9 * AUC**2 + 4 * BRIER**2


def build_models_optim():
    # setup optimization routine
    x0 = np.array([2014, -100, 5])  # YEAR, z, b

    # perform optimization routine
    result = minimize(
        objective_function,
        x0,
        method="Nelder-Mead",
        options={
            "maxiter": 500,
            "disp": True,
            "xatol": 1e-6,
            "fatol": 1e-6,
            "disp": True,
        },
        bounds=[(config["MIN_SEASON_YEAR"], config["MIN_TEST_DATA_YEAR"] - 1), (None, None), (None, None)],
    )
    print(result)

    # print the final result
    print("OPTIMUM IN  :::", result.x)
    print("OPTIMUM OUT :::", objective_function(x=result.x))


def main():
    print("----- WELCOME TO THE BASKETBALL COMPUTER THINGY -----")
    download_and_sort_data(config)  # donwload/sort raw data, if necessary
    build_models_optim()  # optimize the three remaining parameters


if __name__ == "__main__":
    main()
    print_current_season()
