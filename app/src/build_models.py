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
        """
        # looping mechanism, to check how we do live with updated models after each game
        pred_win = []
        count = 0
        for _, row in test_data.iterrows():
            pred_win.append(MODEL_HOME_WIN_PR.value(row, apply_mask=True)[0])
            pd.concat([ref_data, row], ignore_index=True)
            MODEL_HOME_WIN_PR.calculate_model(ref_data)
            count += 1
            print("COUNT -", count)
        pred_win = np.array(pred_win)
        """

        pred_win = MODEL_HOME_WIN_PR.value(test_data, apply_mask=True)
        pred_spread = MODEL_HOME_SPREAD.value(test_data)
        pred_total = MODEL_TOTAL_SCORE.value(test_data)
        true_win = test_data["GAME_homeWin"].to_numpy()
        true_spread = test_data["GAME_spread"].to_numpy()
        true_total = test_data["GAME_total"].to_numpy()
        print("HOME WIN % MODEL  -", MODEL_HOME_WIN_PR.coeffs.T[0], MODEL_HOME_WIN_PR.ref_Rsquared)
        print("SPREAD MODEL      -", MODEL_HOME_SPREAD.coeffs.T[0], MODEL_HOME_SPREAD.ref_Rsquared)
        print("TOTAL SCORE MODEL -", MODEL_TOTAL_SCORE.coeffs.T[0], MODEL_TOTAL_SCORE.ref_Rsquared)
        print("HOME WIN % MASK -", MODEL_HOME_WIN_PR.m, MODEL_HOME_WIN_PR.b)

        # plotting full models
        plotting.plot_pdf_function(pred_win, true_win, "Predicted vs Actual Home Team Win % of NBA games")
        # plotting.plot_2d_histogram(pred_total, true_total, "Predicted vs Actual Total Score of NBA Games")
        # plotting.plot_2d_histogram(pred_spread, true_spread, "Predicted vs Actual Home Team Spread of NBA Games")

        # intermediate plotting and term valuation
        test_terms = [
            # home team params
            HOME_WIN_PERCENTAGE,
            HOME_POINTS_FOR_PER_GAME,
            HOME_POINTS_AGAINST_PER_GAME,
            HOME_STREAK,
            HOME_LAST10_W,
            HOME_LAST10_L,
            HOME_HOME_WIN_PERCENTAGE,
            HOME_HOME_POINTS_FOR_PER_GAME,
            HOME_HOME_POINTS_AGAINST_PER_GAME,
            HOME_HOME_STREAK,
            HOME_HOME_LAST10_W,
            HOME_HOME_LAST10_L,
            HOME_WIN_POINTS_FOR_PER_GAME,
            HOME_WIN_POINTS_AGAINST_PER_GAME,
            HOME_LOSS_POINTS_FOR_PER_GAME,
            HOME_LOSS_POINTS_AGAINST_PER_GAME,
            HOME_HOMEWIN_POINTS_FOR_PER_GAME,
            HOME_HOMEWIN_POINTS_AGAINST_PER_GAME,
            HOME_HOMELOSS_POINTS_FOR_PER_GAME,
            HOME_HOMELOSS_POINTS_AGAINST_PER_GAME,
            # away team params
            AWAY_WIN_PERCENTAGE,
            AWAY_POINTS_FOR_PER_GAME,
            AWAY_POINTS_AGAINST_PER_GAME,
            AWAY_STREAK,
            AWAY_LAST10_W,
            AWAY_LAST10_L,
            AWAY_AWAY_WIN_PERCENTAGE,
            AWAY_AWAY_POINTS_FOR_PER_GAME,
            AWAY_AWAY_POINTS_AGAINST_PER_GAME,
            AWAY_AWAY_STREAK,
            AWAY_AWAY_LAST10_W,
            AWAY_AWAY_LAST10_L,
            AWAY_WIN_POINTS_FOR_PER_GAME,
            AWAY_WIN_POINTS_AGAINST_PER_GAME,
            AWAY_LOSS_POINTS_FOR_PER_GAME,
            AWAY_LOSS_POINTS_AGAINST_PER_GAME,
            AWAY_AWAYWIN_POINTS_FOR_PER_GAME,
            AWAY_AWAYWIN_POINTS_AGAINST_PER_GAME,
            AWAY_AWAYLOSS_POINTS_FOR_PER_GAME,
            AWAY_AWAYLOSS_POINTS_AGAINST_PER_GAME,
        ]

        def find_bounds(terms: list, data: pd.DataFrame):
            num_bins = 301
            bins = np.linspace(-100, 200, num_bins)
            for term in terms:
                value = term.value(data)
                sizes = [len(value[(bins[i - 1] <= value) & (value <= bins[i])]) for i in range(1, num_bins)]
                best = 0
                current = 0
                best_i = None
                best_j = None
                i = 0
                j = 1
                while True:
                    j += 1
                    if j >= len(sizes):
                        break
                    if sizes[j] == 0:
                        current = j - i
                        if current > best:
                            best = current
                            best_i = i
                            best_j = j
                        i = j
                if np.sign(bins[best_i]) == -1 and np.sign(bins[best_j]) == -1:
                    print(0, 1)
                else:
                    print(bins[best_i] + 1, bins[best_j])

        # find_bounds(test_terms, sample_data)
        # TODO delete these 40 plots
        if not True:
            # HOME
            plotting.plot_pdf_function_DEBUG(
                HOME_WIN_PERCENTAGE.value(sample_data), sample_data["GAME_homeWin"].to_numpy(), "DEBUG % v HOME_WIN_PERCENTAGE"
            )
            plotting.plot_pdf_function_DEBUG(
                HOME_POINTS_FOR_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v HOME_POINTS_FOR_PER_GAME",
                binwidth=1,
                bounds=(76, 130),
            )
            plotting.plot_pdf_function_DEBUG(
                HOME_POINTS_AGAINST_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v HOME_POINTS_AGAINST_PER_GAME",
                binwidth=1,
                bounds=(78, 139),
            )
            plotting.plot_pdf_function_DEBUG(
                HOME_STREAK.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v HOME_STREAK",
                binwidth=1,
                bounds=(-27, 31),
            )
            plotting.plot_pdf_function_DEBUG(
                HOME_LAST10_W.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v HOME_LAST10_W",
                binwidth=1,
                bounds=(-1, 11),
            )
            plotting.plot_pdf_function_DEBUG(
                HOME_LAST10_L.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v HOME_LAST10_L",
                binwidth=1,
                bounds=(-1, 11),
            )
            plotting.plot_pdf_function_DEBUG(
                HOME_HOME_WIN_PERCENTAGE.value(sample_data), sample_data["GAME_homeWin"].to_numpy(), "DEBUG % v HOME_HOME_WIN_PERCENTAGE"
            )
            plotting.plot_pdf_function_DEBUG(
                HOME_HOME_POINTS_FOR_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v HOME_HOME_POINTS_FOR_PER_GAME",
                binwidth=1,
                bounds=(72, 133),
            )
            plotting.plot_pdf_function_DEBUG(
                HOME_HOME_POINTS_AGAINST_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v HOME_HOME_POINTS_AGAINST_PER_GAME",
                binwidth=1,
                bounds=(74, 135),
            )
            plotting.plot_pdf_function_DEBUG(
                HOME_HOME_STREAK.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v HOME_HOME_STREAK",
                binwidth=1,
                bounds=(-19, 33),
            )
            plotting.plot_pdf_function_DEBUG(
                HOME_HOME_LAST10_W.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v HOME_HOME_LAST10_W",
                binwidth=1,
                bounds=(-1, 11),
            )
            plotting.plot_pdf_function_DEBUG(
                HOME_HOME_LAST10_L.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v HOME_HOME_LAST10_L",
                binwidth=1,
                bounds=(-1, 11),
            )
            plotting.plot_pdf_function_DEBUG(
                HOME_WIN_POINTS_FOR_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v HOME_WIN_POINTS_FOR_PER_GAME",
                binwidth=1,
                bounds=(77, 138),
            )
            plotting.plot_pdf_function_DEBUG(
                HOME_WIN_POINTS_AGAINST_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v HOME_WIN_POINTS_AGAINST_PER_GAME",
                binwidth=1,
                bounds=(73, 126),
            )
            plotting.plot_pdf_function_DEBUG(
                HOME_LOSS_POINTS_FOR_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v HOME_LOSS_POINTS_FOR_PER_GAME",
                binwidth=1,
                bounds=(72, 131),
            )
            plotting.plot_pdf_function_DEBUG(
                HOME_LOSS_POINTS_AGAINST_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v HOME_LOSS_POINTS_AGAINST_PER_GAME",
                binwidth=1,
                bounds=(80, 143),
            )
            plotting.plot_pdf_function_DEBUG(
                HOME_HOMEWIN_POINTS_FOR_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v HOME_HOMEWIN_POINTS_FOR_PER_GAME",
                binwidth=1,
                bounds=(76, 146),
            )
            plotting.plot_pdf_function_DEBUG(
                HOME_HOMEWIN_POINTS_AGAINST_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v HOME_HOMEWIN_POINTS_AGAINST_PER_GAME",
                binwidth=1,
                bounds=(66, 134),
            )
            plotting.plot_pdf_function_DEBUG(
                HOME_HOMELOSS_POINTS_FOR_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v HOME_HOMELOSS_POINTS_FOR_PER_GAME",
                binwidth=1,
                bounds=(64, 136),
            )
            plotting.plot_pdf_function_DEBUG(
                HOME_HOMELOSS_POINTS_AGAINST_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v HOME_HOMELOSS_POINTS_AGAINST_PER_GAME",
                binwidth=1,
                bounds=(78, 149),
            )
            # AWAY
            plotting.plot_pdf_function_DEBUG(
                AWAY_WIN_PERCENTAGE.value(sample_data), sample_data["GAME_homeWin"].to_numpy(), "DEBUG % v AWAY_WIN_PERCENTAGE"
            )
            plotting.plot_pdf_function_DEBUG(
                AWAY_POINTS_FOR_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v AWAY_POINTS_FOR_PER_GAME",
                binwidth=1,
                bounds=(76, 131),
            )
            plotting.plot_pdf_function_DEBUG(
                AWAY_POINTS_AGAINST_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v AWAY_POINTS_AGAINST_PER_GAME",
                binwidth=1,
                bounds=(78, 135),
            )
            plotting.plot_pdf_function_DEBUG(
                AWAY_STREAK.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v AWAY_STREAK",
                binwidth=1,
                bounds=(-26, 34),
            )
            plotting.plot_pdf_function_DEBUG(
                AWAY_LAST10_W.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v AWAY_LAST10_W",
                binwidth=1,
                bounds=(-1, 11),
            )
            plotting.plot_pdf_function_DEBUG(
                AWAY_LAST10_L.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v AWAY_LAST10_L",
                binwidth=1,
                bounds=(-1, 11),
            )
            plotting.plot_pdf_function_DEBUG(
                AWAY_AWAY_WIN_PERCENTAGE.value(sample_data), sample_data["GAME_homeWin"].to_numpy(), "DEBUG % v AWAY_AWAY_WIN_PERCENTAGE"
            )
            plotting.plot_pdf_function_DEBUG(
                AWAY_AWAY_POINTS_FOR_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v AWAY_AWAY_POINTS_FOR_PER_GAME",
                binwidth=1,
                bounds=(74, 132),
            )
            plotting.plot_pdf_function_DEBUG(
                AWAY_AWAY_POINTS_AGAINST_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v AWAY_AWAY_POINTS_AGAINST_PER_GAME",
                binwidth=1,
                bounds=(73, 135),
            )
            plotting.plot_pdf_function_DEBUG(
                AWAY_AWAY_STREAK.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v AWAY_AWAY_STREAK",
                binwidth=1,
                bounds=(-37, 16),
            )
            plotting.plot_pdf_function_DEBUG(
                AWAY_AWAY_LAST10_W.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v AWAY_AWAY_LAST10_W",
                binwidth=1,
                bounds=(-1, 11),
            )
            plotting.plot_pdf_function_DEBUG(
                AWAY_AWAY_LAST10_L.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v AWAY_AWAY_LAST10_L",
                binwidth=1,
                bounds=(-1, 11),
            )
            plotting.plot_pdf_function_DEBUG(
                AWAY_WIN_POINTS_FOR_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v AWAY_WIN_POINTS_FOR_PER_GAME",
                binwidth=1,
                bounds=(83, 137),
            )
            plotting.plot_pdf_function_DEBUG(
                AWAY_WIN_POINTS_AGAINST_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v AWAY_WIN_POINTS_AGAINST_PER_GAME",
                binwidth=1,
                bounds=(71, 125),
            )
            plotting.plot_pdf_function_DEBUG(
                AWAY_LOSS_POINTS_FOR_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v AWAY_LOSS_POINTS_FOR_PER_GAME",
                binwidth=1,
                bounds=(67, 130),
            )
            plotting.plot_pdf_function_DEBUG(
                AWAY_LOSS_POINTS_AGAINST_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v AWAY_LOSS_POINTS_AGAINST_PER_GAME",
                binwidth=1,
                bounds=(82, 146),
            )
            plotting.plot_pdf_function_DEBUG(
                AWAY_AWAYWIN_POINTS_FOR_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v AWAY_AWAYWIN_POINTS_FOR_PER_GAME",
                binwidth=1,
                bounds=(72, 144),
            )
            plotting.plot_pdf_function_DEBUG(
                AWAY_AWAYWIN_POINTS_AGAINST_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v AWAY_AWAYWIN_POINTS_AGAINST_PER_GAME",
                binwidth=1,
                bounds=(64, 137),
            )
            plotting.plot_pdf_function_DEBUG(
                AWAY_AWAYLOSS_POINTS_FOR_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v AWAY_AWAYLOSS_POINTS_FOR_PER_GAME",
                binwidth=1,
                bounds=(62, 131),
            )
            plotting.plot_pdf_function_DEBUG(
                AWAY_AWAYLOSS_POINTS_AGAINST_PER_GAME.value(sample_data),
                sample_data["GAME_homeWin"].to_numpy(),
                "DEBUG % v AWAY_AWAYLOSS_POINTS_AGAINST_PER_GAME",
                binwidth=1,
                bounds=(76, 147),
            )
