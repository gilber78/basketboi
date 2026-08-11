import os
import json

with open(os.path.join("app", "data", "config.json"), "r") as file:
    config = json.load(file)
    os.environ["SEASON_PATH"] = os.path.join(config["DATA_DOWNLOAD_PATH"], "seasons")
with open(config["KAGGLE_API_TOKEN_PATH"], "r") as file:
    os.environ["KAGGLE_USERNAME"] = file.readline()
    os.environ["KAGGLE_KEY"] = file.readline()

import plotting
import pandas as pd
import statistics as stats
import matplotlib.pyplot as plt
from functools import partial
from bayes_opt import BayesianOptimization

from Models import *
from functions import print_current_season
from download_and_sort_data import download_and_sort_data  # this import has to come last


def objective_function_tuple(year, z, b, debug_prints=False, debug_plots=False, debug_debug_plots=False):
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
        # print(">>>", game_date)
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

    # debug ouputs, based on the optional parameters
    if debug_prints:
        print("=========", year, z, b, "=========")
        print("ECE:", ECE)
        print("Slope:", M)
        print("Intercept:", B)
        print("AUC: ", AUC)
        print("Brier:", BRIER)

    if debug_plots:
        plotting.plot_pdf_function(pred_win, true_win, "Predicted vs Actual Home Team Win % of NBA games")
        plotting.plot_ROC_curve(pred_win, true_win, "ROC curve for Home Team Win % of NBA games%")

    if debug_debug_plots:
        # get debug data needed for these below plots
        sample_data = pd.concat(
            [
                pd.read_csv(os.path.join(os.environ["SEASON_PATH"], dir, f"{dir}_full.csv"))
                for dir in os.listdir(os.environ["SEASON_PATH"])
                if (int(dir.split("-")[0]) >= config["MIN_REFERENCE_DATA_YEAR"])
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

        find_bounds(test_terms, sample_data)

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

    return ECE, M, B, AUC, BRIER


def objective_function_scalar(year, z, b):
    ECE, M, B, AUC, BRIER = objective_function_tuple(year, z, b)
    # return -4 * ECE**2
    return (-4 * ECE**2) + (-4 * (M - 1) ** 2) + (-4 * B**2)
    # return -4 * (M - 1) ** 2 + 4 * B**2
    # return -4 * (AUC - 1) ** 2
    # return -4 * BRIER**2
    # return -np.sqrt(55 * ECE**2 + 10 * (M - 1) ** 2 + 10 * B**2 + 5 * (AUC - 1) ** 2 + 20 * BRIER**2)
    # return -np.sqrt(55 * ECE**2 + 5 * (M - 1) ** 2 + 15 * B**2 + 25 * BRIER**2)
    # return -np.sqrt(25 * ECE**2 + 5 * (M - 1) ** 2 + 15 * B**2 + 5 * (AUC - 1) ** 2 + 50 * BRIER**2)


def optim_models_daybyday(
    x0=None,
    year_bounds=(config["MIN_SEASON_YEAR"], config["MIN_TEST_DATA_YEAR"] - 1),
    z_bounds=(-100, -2),
    b_bounds=(1, 100),
    init_points=5,
    n_iter=5,
    # xi=0.05,  # likely will prever something small and less than 0.1
    verbose=2,
    from_register: list | None = None,
    from_file: str | None = None,
    to_file="optim_data/bo-optimizer.json",
):
    # initialize BO object (if from file or from scratch)
    optimizer = BayesianOptimization(
        f=objective_function_scalar,
        pbounds={
            "year": (year_bounds[0], year_bounds[1], int),
            "z": (z_bounds[0], z_bounds[1]),
            "b": (b_bounds[0], b_bounds[1]),
        },
        # acquisition_function=ExpectedImprovement(xi=xi),
        verbose=verbose,
    )

    # initial values, if supplied to the function
    if (from_register is not None) and (from_file is not None):
        raise Exception("Cannot optimize from both register and file -- pick one arg to pass")
    elif from_register:
        param_list = from_register[0]
        target_list = from_register[1]
        for i in range(len(param_list)):
            optimizer.register(params=param_list[i], target=target_list[i])
    elif from_file is not None:
        optimizer.load_state(from_file)

    try:
        # probe x0s, if supplied
        if x0 is not None:
            for x in x0:
                optimizer.probe(params={"year": x[0], "z": x[1], "b": x[2]}, lazy=True)  # set lazy to true

        # run minimize/maximize
        optimizer.maximize(init_points=init_points, n_iter=n_iter)

    except Exception as e:
        print(e)

    except KeyboardInterrupt:
        print("<<< STOPPING PREMATURELY, DUMPING TO FILE >>>")

    finally:
        # save optimizer to json
        optimizer.save_state(to_file)

        # return the suggested next point(s)
        return optimizer.max, optimizer.suggest()


def main():
    print("----- WELCOME TO THE BASKETBALL MODEL OPTIMIZER -----")
    download_and_sort_data(config)  # donwload/sort raw data, if necessary

    '''
    # calls of optim_models_daybyday
    """
        [
            (2020, -42, 51),
            (2020, -39.62048, 79.193165),
            (2020, -39.10584517831557, 78.45754303601926),
            (2020, -34.96927644039598, 64.49692881215984),
            (2020, -31.73873, 41.673835),
            (2020, -33.32630956728241, 61.40157044130502),
            (2020, -22.949122292006116, 28.682112542279647),
            (2020, -40.49972745901825, 68.89789705377231), **
        ],
    """
    best_value, next_point = optim_models_daybyday(
        year_bounds=(2020, 2020),
        from_file="optim_data/bo-optimizer5.json",
        to_file="optim_data/bo-optimizer5.json",
        init_points=1,
        n_iter=1,
    )
    # if we ever resume just ECE, use bo-optimizer3.json instead. optimizer5 is due to an error in the objective scalar function. 4 is broken.
    # 6 should be made in order to better balance ECE with slope/intercept errors (maybe go 8 and 4 instead of 4 and 4 for weights?)

    # note that the changes need to occur in the json file as well as the code
    print("BEST VALUE:", best_value)
    print("NEXT TRIAL:", next_point)

    # final call of suggested point to the tuple function, print out results of all states
    objective_function_tuple(best_value["params"]["year"], best_value["params"]["z"], best_value["params"]["b"], debug_plots=True)
    '''

    objective_function_tuple(2020, -40.49972745901825, 68.89789705377231, debug_prints=True, debug_plots=True)  # test no debug out


if __name__ == "__main__":
    main()
    print_current_season()
    plt.show()
