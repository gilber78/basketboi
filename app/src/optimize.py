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
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import ExpectedImprovement

from Models import *
from functions import print_current_season
from download_and_sort_data import download_and_sort_data  # this import has to come last


def objective_function_tuple(year, z, b):
    # get reference data based on min year parameter
    ref_data = pd.concat(
        [
            pd.read_csv(os.path.join(os.environ["SEASON_PATH"], dir, f"{dir}_full.csv"))
            for dir in os.listdir(os.environ["SEASON_PATH"])
            if (int(dir.split("-")[0]) >= year) and not (int(dir.split("-")[0]) >= config["MIN_TEST_DATA_YEAR"])
        ],
        ignore_index=True,
    )
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

    # evaluate the original model with substituted parameters
    MODEL_HOME_WIN_PR.weight_func = partial(recency_weight_function, z=z, b=b)
    MODEL_HOME_WIN_PR.calculate_model(ref_data)

    # looping mechanism, to check how we do live with updated models after each day of the season
    pred_win = []
    test_dates = test_data["GAME_gameDate"].unique()
    for game_date in test_dates:
        print(">>>", game_date, year, z, b)
        for _, row in test_data[test_data["GAME_gameDate"] == game_date].iterrows():
            pred_win.append(MODEL_HOME_WIN_PR.value(row, apply_mask=True)[0])
            ref_data.loc[len(ref_data)] = row
        MODEL_HOME_WIN_PR.calculate_model(ref_data)
    pred_win = np.array(pred_win)
    true_win = test_data["GAME_homeWin"].to_numpy()
    _, _, AUC = stats.calc_ROC_curve(pred_win, true_win)
    BRIER = stats.calc_brier_score(pred_win, true_win)
    ECE = stats.calc_ECE_score(pred_win, true_win)
    _, _, M, B = stats.calc_calibrated_slope_intercept(pred_win, true_win)

    return ECE, M, B, AUC, BRIER


def objective_function_scalar(year, z, b):
    ECE, M, B, AUC, BRIER = objective_function_tuple(year, z, b)
    return -4 * ECE**2
    # return -4 * (M - 1) ** 2 + 4 * B**2
    # return -4 * (AUC - 1) ** 2
    # return -4 * BRIER**2
    # return -np.sqrt(55 * ECE**2 + 10 * (M - 1) ** 2 + 10 * B**2 + 5 * (AUC - 1) ** 2 + 20 * BRIER**2)
    # return -np.sqrt(55 * ECE**2 + 5 * (M - 1) ** 2 + 15 * B**2 + 25 * BRIER**2)
    # return -np.sqrt(25 * ECE**2 + 5 * (M - 1) ** 2 + 15 * B**2 + 5 * (AUC - 1) ** 2 + 50 * BRIER**2)


def optim_models_daybyday(
    x0=None,
    year_bounds=(config["MIN_SEASON_YEAR"], config["MIN_TEST_DATA_YEAR"] - 1),
    z_bounds=(-100, 0),
    b_bounds=(0, 500),
    init_points=5,
    n_iter=5,
    verbose=2,
    from_file=None,
    to_file="optim_data/bo-optimizer.json",
):
    # initialize BO object (if from file or from scratch)
    optimizer = BayesianOptimization(
        f=objective_function_scalar,
        acquisition_function=ExpectedImprovement(xi=0.0),  # likely will prever something small and less than 0.1
        pbounds={
            "year": (year_bounds[0], year_bounds[1], int),
            "z": (z_bounds[0], z_bounds[1]),
            "b": (b_bounds[0], b_bounds[1]),
        },
        verbose=verbose,
        # random_state=1,
    )
    if from_file is not None:
        optimizer.load_state(from_file)

    # probe x0s, if supplied
    if x0 is not None:
        for x in x0:
            optimizer.probe(params={"year": x[0], "z": x[1], "b": x[2]}, lazy=True)  # set lazy to true

    # run minimize/maximize
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    # save optimizer to json
    optimizer.save_state(to_file)
    # print(optimizer.res)

    # return the suggested next point(s)
    return optimizer.suggest()


def main():
    print("----- WELCOME TO THE BASKETBALL COMPUTER THINGY -----")
    download_and_sort_data(config)  # donwload/sort raw data, if necessary

    # calls of optim_models_daybyday
    next_point = optim_models_daybyday(
        [(2020, -42, 51)],
        [(2014, -100, 5)],
        [(2015, -50, 50)],
        [(2018, -40, 55)],
        # from_file="optim_data/bo-optimizer.json",
        to_file="optim_data/bo-optimizer.json",
    )
    print(next_point)

    # final call of suggested point to the tuple function, print out results of all states
    print(objective_function_tuple(next_point["year"], next_point["z"], next_point["b"]))


if __name__ == "__main__":
    main()
    print_current_season()

# TODO ? eventually put the debug plots here instead of being inside build_models
