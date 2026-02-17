import os
import pickle
import pandas as pd

import plotting
from Models import MODEL_HOME_WIN_PR, MODEL_HOME_SPREAD, MODEL_TOTAL_SCORE


def build_models(config):
    # get reference data
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

    # calculate out all the models
    MODEL_HOME_WIN_PR.calculate_model(ref_data)
    MODEL_HOME_SPREAD.calculate_model(ref_data)
    MODEL_TOTAL_SCORE.calculate_model(ref_data)

    if config["DEBUG"]:
        pred_win = MODEL_HOME_WIN_PR.value(test_data)
        pred_spread = MODEL_HOME_SPREAD.value(test_data)
        pred_total = MODEL_TOTAL_SCORE.value(test_data)
        true_spread = test_data["GAME_spread"].to_numpy()
        true_win = test_data["GAME_homeWin"].to_numpy()
        true_total = test_data["GAME_total"].to_numpy()
        print("HOME WIN % MODEL  -", MODEL_HOME_WIN_PR.coeffs.T[0], MODEL_HOME_WIN_PR.ref_Rsquared)
        print("SPREAD MODEL      -", MODEL_HOME_SPREAD.coeffs.T[0], MODEL_HOME_SPREAD.ref_Rsquared)
        print("TOTAL SCORE MODEL -", MODEL_TOTAL_SCORE.coeffs.T[0], MODEL_TOTAL_SCORE.ref_Rsquared)
        print("data sizes:", len(ref_data), len(test_data), len(ref_data) + len(test_data), 61043)

        # plotting
        plotting.plot_2d_histogram(pred_total, true_total, "Predicted vs Actual Total Score of NBA Games")
        plotting.plot_2d_histogram(pred_spread, true_spread, "Predicted vs Actual Home Team Spread of NBA Games")
        plotting.plot_pdf_function(pred_win, true_win, "Predicted vs Actual Home Team Win % of NBA games")

        # extra plots, for refining the model performance

    # DEBUG
    # TODO delete when this is a stable part of the program, getting current year standings
    test_year = "2025-2026"  # "1946-1947" "1995-1996" "2025-2026"
    with open(f"app\\data\\games\\seasons\\{test_year}\\{test_year}_season.pkl", "rb") as file:
        test_season = pickle.load(file)
    print()
    test_season.pretty_print()
