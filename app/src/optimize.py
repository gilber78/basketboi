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
from scipy.optimize import minimize, differential_evolution

from Models import *
from functions import print_current_season
from download_and_sort_data import download_and_sort_data  # this import has to come last


def print_progress(*, intermediate_result):
    print(">>>", intermediate_result.x, intermediate_result.fun)


def objective_function_tuple(x):
    # split the vector into compontnes
    min_reference_data_year = np.round(x[0])
    z = x[1]
    b = x[2]

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

        return ECE, M, B, AUC, BRIER


def objective_function_scalar(x):
    ECE, M, B, AUC, BRIER = objective_function_tuple(x)
    return 4 * ECE**2
    # return 4 * (M - 1) ** 2 + 4 * B**2
    # return 4 * (AUC - 1) ** 2
    # return 4 * BRIER**2
    # return np.sqrt(55 * ECE**2 + 10 * (M - 1) ** 2 + 10 * B**2 + 5 * (AUC - 1) ** 2 + 20 * BRIER**2)
    # return np.sqrt(55 * ECE**2 + 5 * (M - 1) ** 2 + 15 * B**2 + 25 * BRIER**2)
    # return np.sqrt(25 * ECE**2 + 5 * (M - 1) ** 2 + 15 * B**2 + 5 * (AUC - 1) ** 2 + 50 * BRIER**2)


def optim_models_simplex(grid_search=False, x0=np.array([2014, -100, 5])):
    if grid_search:
        year_bound = (config["MIN_SEASON_YEAR"], config["MIN_TEST_DATA_YEAR"] - 1)
        z_bound = (-250, -5)
        b_bound = (1, 100)
        year_values = np.linspace(*year_bound, num=(year_bound[1] - year_bound[0] + 1) // 10)
        z_values = np.linspace(*z_bound, num=(z_bound[1] - z_bound[0] - 1) // 35)
        b_values = np.linspace(*b_bound, num=(b_bound[1] - b_bound[0] - 1) // 9)
        best_f = 100
        best_x = np.array([])
        for year in year_values:
            for z in z_values:
                for b in b_values:
                    x_test = np.array([year, z, b])
                    f_test = objective_function_scalar(x_test)
                    if f_test <= best_f:
                        best_x = x_test
                        best_f = f_test
                        print(">>>", best_x, best_f)
    else:
        year_bound = (config["MIN_SEASON_YEAR"], config["MIN_TEST_DATA_YEAR"] - 1)
        z_bound = (None, 0)
        b_bound = (0, None)
        best_x = x0

    # setup optimization routine - x0 is the winner of grid search
    print("RUNNING SIMPLEX METHOD...")
    result = minimize(
        objective_function_scalar,
        best_x,
        method="Nelder-Mead",
        options={
            "maxiter": 200,
            "disp": True,
            "xatol": 1e-6,
            "fatol": 1e-6,
            "disp": True,
        },
        bounds=[year_bound, z_bound, b_bound],
        callback=print_progress,
    )
    print(result)

    # print the final result
    print("initial point :::", x0)
    print("Score to beat :::", objective_function_scalar(x0))
    print("OPTIMUM IN    :::", result.x)
    print("OPTIMUM OUT   :::", objective_function_scalar(result.x))


def optim_models_genetic(x0=np.array([2014, -100, 5])):
    year_bound = (config["MIN_SEASON_YEAR"], config["MIN_TEST_DATA_YEAR"] - 1)
    z_bound = (-250, -5)
    b_bound = (1, 100)
    print("RUNNING GENETIC ALGORITHM...")
    result = differential_evolution(
        func=objective_function_scalar,
        bounds=[year_bound, z_bound, b_bound],
        x0=x0,
        strategy="best1bin",
        maxiter=500,
        callback=print_progress,
        disp=True,
        # polish=True,
        integrality=[True, False, False],
    )
    print(result)

    # print the final result
    # print("initial point :::", x0)
    # print("Score to beat :::", objective_function_scalar(x0))
    print("OPTIMUM IN    :::", result.x)
    print("OPTIMUM OUT   :::", objective_function_scalar(result.x))


def optim_models_daybyday():
    # TODO fill out this function with new logic, that can consider multiple models simultaneously without overwriting the same model each time
    # consider writing your own genetic optimization
    # as much parallelization you can write (calculate models at each iter) the better
    pass


def main():
    print("----- WELCOME TO THE BASKETBALL COMPUTER THINGY -----")
    download_and_sort_data(config)  # donwload/sort raw data, if necessary
    # optim_models_simplex(True)
    # optim_models_simplex(False, np.array([]))
    # optim_models_genetic(np.array([]))


if __name__ == "__main__":
    main()
    print_current_season()
