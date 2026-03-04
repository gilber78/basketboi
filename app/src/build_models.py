import os
import pandas as pd

import plotting

# from Models import MODEL_HOME_WIN_PR, MODEL_HOME_SPREAD, MODEL_TOTAL_SCORE
from Models import *


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
        sample_data = pd.concat(
            [
                pd.read_csv(os.path.join(os.environ["SEASON_PATH"], dir, f"{dir}_full.csv"))
                for dir in os.listdir(os.environ["SEASON_PATH"])
                if (int(dir.split("-")[0]) >= config["MIN_REFERENCE_DATA_YEAR"])
                and not ((int(dir.split("-")[0]) >= config["MIN_TEST_DATA_YEAR"]) and config["DEBUG"])
            ],
            ignore_index=True,
        )
        sample_data = sample_data[
            (sample_data["HOME_games_played"] != 0)
            & (sample_data["AWAY_games_played"] != 0)
            & (sample_data["HOME_wins"] != 0)
            & (sample_data["AWAY_wins"] != 0)
            & (sample_data["HOME_home_wins"] != 0)
            & (sample_data["AWAY_away_wins"] != 0)
            & (sample_data["HOME_home_losses"] != 0)
            & (sample_data["AWAY_away_losses"] != 0)
        ].reset_index()

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

        # plotting full models
        # TODO put back these plots
        # plotting.plot_2d_histogram(pred_total, true_total, "Predicted vs Actual Total Score of NBA Games")
        # plotting.plot_2d_histogram(pred_spread, true_spread, "Predicted vs Actual Home Team Spread of NBA Games")
        plotting.plot_pdf_function(pred_win, true_win, "Predicted vs Actual Home Team Win % of NBA games")

        # intermediate plotting and term valuation
        # TODO find proper bounds for all these items, somehow
        # HOME
        plotting.plot_pdf_function(HOME_WIN_PERCENTAGE.value(sample_data), sample_data["GAME_homeWin"].to_numpy(), "DEBUG % v HOME_WIN_PERCENTAGE")
        plotting.plot_pdf_function(
            HOME_POINTS_FOR_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v HOME_POINTS_FOR_PER_GAME",
            binwidth=1,
            bounds=(76, 130),
        )
        plotting.plot_pdf_function(
            HOME_POINTS_AGAINST_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v HOME_POINTS_AGAINST_PER_GAME",
            binwidth=1,
            bounds=(78, 139),
        )
        plotting.plot_pdf_function(
            HOME_STREAK.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v HOME_STREAK",
            binwidth=1,
            bounds=(-26, 30),
        )
        plotting.plot_pdf_function(
            HOME_LAST10_W.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v HOME_LAST10_W",
            binwidth=1,
            bounds=(0, 10),
        )
        plotting.plot_pdf_function(
            HOME_LAST10_L.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v HOME_LAST10_L",
            binwidth=1,
            bounds=(0, 10),
        )
        plotting.plot_pdf_function(
            HOME_HOME_WIN_PERCENTAGE.value(sample_data), sample_data["GAME_homeWin"].to_numpy(), "DEBUG % v HOME_HOME_WIN_PERCENTAGE"
        )
        plotting.plot_pdf_function(
            HOME_HOME_POINTS_FOR_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v HOME_HOME_POINTS_FOR_PER_GAME",
            binwidth=1,
            bounds=(76, 130),
        )
        plotting.plot_pdf_function(
            HOME_HOME_POINTS_AGAINST_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v HOME_HOME_POINTS_AGAINST_PER_GAME",
            binwidth=1,
            bounds=(80, 130),
        )
        plotting.plot_pdf_function(
            HOME_HOME_STREAK.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v HOME_HOME_STREAK",
            binwidth=1,
            bounds=(-18, 32),
        )
        plotting.plot_pdf_function(
            HOME_HOME_LAST10_W.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v HOME_HOME_LAST10_W",
            binwidth=1,
            bounds=(0, 10),
        )
        plotting.plot_pdf_function(
            HOME_HOME_LAST10_L.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v HOME_HOME_LAST10_L",
            binwidth=1,
            bounds=(0, 10),
        )
        plotting.plot_pdf_function(
            HOME_WIN_POINTS_FOR_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v HOME_WIN_POINTS_FOR_PER_GAME",
            binwidth=1,
            bounds=(80, 130),
        )
        plotting.plot_pdf_function(
            HOME_WIN_POINTS_AGAINST_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v HOME_WIN_POINTS_AGAINST_PER_GAME",
            binwidth=1,
            bounds=(80, 125),
        )
        plotting.plot_pdf_function(
            HOME_LOSS_POINTS_FOR_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v HOME_LOSS_POINTS_FOR_PER_GAME",
            binwidth=1,
            bounds=(80, 130),
        )
        plotting.plot_pdf_function(
            HOME_LOSS_POINTS_AGAINST_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v HOME_LOSS_POINTS_AGAINST_PER_GAME",
            binwidth=1,
            bounds=(80, 130),
        )
        plotting.plot_pdf_function(
            HOME_HOMEWIN_POINTS_FOR_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v HOME_HOMEWIN_POINTS_FOR_PER_GAME",
            binwidth=1,
            bounds=(80, 130),
        )
        plotting.plot_pdf_function(
            HOME_HOMEWIN_POINTS_AGAINST_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v HOME_HOMEWIN_POINTS_AGAINST_PER_GAME",
            binwidth=1,
            bounds=(80, 130),
        )
        plotting.plot_pdf_function(
            HOME_HOMELOSS_POINTS_FOR_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v HOME_HOMELOSS_POINTS_FOR_PER_GAME",
            binwidth=1,
            bounds=(80, 130),
        )
        plotting.plot_pdf_function(
            HOME_HOMELOSS_POINTS_AGAINST_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v HOME_HOMELOSS_POINTS_AGAINST_PER_GAME",
            binwidth=1,
            bounds=(80, 130),
        )

        # AWAY
        plotting.plot_pdf_function(AWAY_WIN_PERCENTAGE.value(sample_data), sample_data["GAME_homeWin"].to_numpy(), "DEBUG % v AWAY_WIN_PERCENTAGE")
        plotting.plot_pdf_function(
            AWAY_POINTS_FOR_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v AWAY_POINTS_FOR_PER_GAME",
            binwidth=1,
            bounds=(80, 130),
        )
        plotting.plot_pdf_function(
            AWAY_POINTS_AGAINST_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v AWAY_POINTS_AGAINST_PER_GAME",
            binwidth=1,
            bounds=(80, 130),
        )
        plotting.plot_pdf_function(
            AWAY_STREAK.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v AWAY_STREAK",
            binwidth=1,
            bounds=(-26, 30),
        )
        plotting.plot_pdf_function(
            AWAY_LAST10_W.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v AWAY_LAST10_W",
            binwidth=1,
            bounds=(0, 10),
        )
        plotting.plot_pdf_function(
            AWAY_LAST10_L.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v AWAY_LAST10_L",
            binwidth=1,
            bounds=(0, 10),
        )
        plotting.plot_pdf_function(
            AWAY_AWAY_WIN_PERCENTAGE.value(sample_data), sample_data["GAME_homeWin"].to_numpy(), "DEBUG % v AWAY_AWAY_WIN_PERCENTAGE"
        )
        plotting.plot_pdf_function(
            AWAY_AWAY_POINTS_FOR_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v AWAY_AWAY_POINTS_FOR_PER_GAME",
            binwidth=1,
            bounds=(80, 130),
        )
        plotting.plot_pdf_function(
            AWAY_AWAY_POINTS_AGAINST_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v AWAY_AWAY_POINTS_AGAINST_PER_GAME",
            binwidth=1,
            bounds=(80, 130),
        )
        plotting.plot_pdf_function(
            AWAY_AWAY_STREAK.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v AWAY_AWAY_STREAK",
            binwidth=1,
            bounds=(-36, 15),
        )
        plotting.plot_pdf_function(
            AWAY_AWAY_LAST10_W.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v AWAY_AWAY_LAST10_W",
            binwidth=1,
            bounds=(0, 10),
        )
        plotting.plot_pdf_function(
            AWAY_AWAY_LAST10_L.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v AWAY_AWAY_LAST10_L",
            binwidth=1,
            bounds=(0, 10),
        )
        plotting.plot_pdf_function(
            AWAY_WIN_POINTS_FOR_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v AWAY_WIN_POINTS_FOR_PER_GAME",
            binwidth=1,
            bounds=(83, 136),
        )
        plotting.plot_pdf_function(
            AWAY_WIN_POINTS_AGAINST_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v AWAY_WIN_POINTS_AGAINST_PER_GAME",
            binwidth=1,
            bounds=(80, 125),
        )
        plotting.plot_pdf_function(
            AWAY_LOSS_POINTS_FOR_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v AWAY_LOSS_POINTS_FOR_PER_GAME",
            binwidth=1,
            bounds=(80, 130),
        )
        plotting.plot_pdf_function(
            AWAY_LOSS_POINTS_AGAINST_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v AWAY_LOSS_POINTS_AGAINST_PER_GAME",
            binwidth=1,
            bounds=(82, 130),
        )
        plotting.plot_pdf_function(
            AWAY_AWAYWIN_POINTS_FOR_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v AWAY_AWAYWIN_POINTS_FOR_PER_GAME",
            binwidth=1,
            bounds=(80, 130),
        )
        plotting.plot_pdf_function(
            AWAY_AWAYWIN_POINTS_AGAINST_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v AWAY_AWAYWIN_POINTS_AGAINST_PER_GAME",
            binwidth=1,
            bounds=(80, 130),
        )
        plotting.plot_pdf_function(
            AWAY_AWAYLOSS_POINTS_FOR_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v AWAY_AWAYLOSS_POINTS_FOR_PER_GAME",
            binwidth=1,
            bounds=(80, 130),
        )
        plotting.plot_pdf_function(
            AWAY_AWAYLOSS_POINTS_AGAINST_PER_GAME.value(sample_data),
            sample_data["GAME_homeWin"].to_numpy(),
            "DEBUG % v AWAY_AWAYLOSS_POINTS_AGAINST_PER_GAME",
            binwidth=1,
            bounds=(80, 130),
        )
