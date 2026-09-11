import os
import sys
import json
import argparse
from pathlib import Path

with open(os.path.join("app", "data", "config.json"), "r") as file:
    config = json.load(file)
    os.environ["SEASON_PATH"] = os.path.join(config["DATA_DOWNLOAD_PATH"], "seasons")
with open(config["KAGGLE_API_TOKEN_PATH"], "r") as file:
    os.environ["KAGGLE_USERNAME"] = file.readline()
    os.environ["KAGGLE_KEY"] = file.readline()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from bayes_opt import BayesianOptimization

import statistics as stats
import plotting as plotting
from server.Models import *
from server.functions import print_current_season
from server.download_and_sort_data import download_and_sort_data  # this import has to come last

TEST_TERMS = [
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

TITLES_LAMBDA = lambda title_string: [
    # home team params
    f"DEBUG {title_string} v HOME_WIN_PERCENTAGE",
    f"DEBUG {title_string} v HOME_POINTS_FOR_PER_GAME",
    f"DEBUG {title_string} v HOME_POINTS_AGAINST_PER_GAME",
    f"DEBUG {title_string} v HOME_STREAK",
    f"DEBUG {title_string} v HOME_LAST10_W",
    f"DEBUG {title_string} v HOME_LAST10_L",
    f"DEBUG {title_string} v HOME_HOME_WIN_PERCENTAGE",
    f"DEBUG {title_string} v HOME_HOME_POINTS_FOR_PER_GAME",
    f"DEBUG {title_string} v HOME_HOME_POINTS_AGAINST_PER_GAME",
    f"DEBUG {title_string} v HOME_HOME_STREAK",
    f"DEBUG {title_string} v HOME_HOME_LAST10_W",
    f"DEBUG {title_string} v HOME_HOME_LAST10_L",
    f"DEBUG {title_string} v HOME_WIN_POINTS_FOR_PER_GAME",
    f"DEBUG {title_string} v HOME_WIN_POINTS_AGAINST_PER_GAME",
    f"DEBUG {title_string} v HOME_LOSS_POINTS_FOR_PER_GAME",
    f"DEBUG {title_string} v HOME_LOSS_POINTS_AGAINST_PER_GAME",
    f"DEBUG {title_string} v HOME_HOMEWIN_POINTS_FOR_PER_GAME",
    f"DEBUG {title_string} v HOME_HOMEWIN_POINTS_AGAINST_PER_GAME",
    f"DEBUG {title_string} v HOME_HOMELOSS_POINTS_FOR_PER_GAME",
    f"DEBUG {title_string} v HOME_HOMELOSS_POINTS_AGAINST_PER_GAME",
    # away team params
    f"DEBUG {title_string} v AWAY_WIN_PERCENTAGE",
    f"DEBUG {title_string} v AWAY_POINTS_FOR_PER_GAME",
    f"DEBUG {title_string} v AWAY_POINTS_AGAINST_PER_GAME",
    f"DEBUG {title_string} v AWAY_STREAK",
    f"DEBUG {title_string} v AWAY_LAST10_W",
    f"DEBUG {title_string} v AWAY_LAST10_L",
    f"DEBUG {title_string} v AWAY_AWAY_WIN_PERCENTAGE",
    f"DEBUG {title_string} v AWAY_AWAY_POINTS_FOR_PER_GAME",
    f"DEBUG {title_string} v AWAY_AWAY_POINTS_AGAINST_PER_GAME",
    f"DEBUG {title_string} v AWAY_AWAY_STREAK",
    f"DEBUG {title_string} v AWAY_AWAY_LAST10_W",
    f"DEBUG {title_string} v AWAY_AWAY_LAST10_L",
    f"DEBUG {title_string} v AWAY_WIN_POINTS_FOR_PER_GAME",
    f"DEBUG {title_string} v AWAY_WIN_POINTS_AGAINST_PER_GAME",
    f"DEBUG {title_string} v AWAY_LOSS_POINTS_FOR_PER_GAME",
    f"DEBUG {title_string} v AWAY_LOSS_POINTS_AGAINST_PER_GAME",
    f"DEBUG {title_string} v AWAY_AWAYWIN_POINTS_FOR_PER_GAME",
    f"DEBUG {title_string} v AWAY_AWAYWIN_POINTS_AGAINST_PER_GAME",
    f"DEBUG {title_string} v AWAY_AWAYLOSS_POINTS_FOR_PER_GAME",
    f"DEBUG {title_string} v AWAY_AWAYLOSS_POINTS_AGAINST_PER_GAME",
]


def get_args():
    parser = argparse.ArgumentParser(
        description="""optimize.py ::: contains the backend utilities necessary to improve and fine-tune all the NBA models that BASKETBOI runs"""
    )
    parser.add_argument(
        "--model",
        "-m",
        dest="model",
        type=str,
        required=True,
        choices=["homeSpread", "homeWin", "totalScore"],
        help="Only required arg, specifies which of the three models to run the optimization against. Choices are [homeSpread, homeWin, totalScore].",
    )
    parser.add_argument(
        "--optimize",
        "-o",
        dest="optimize",
        default=False,
        action="store_true",
        help="Calls the full bayesian optimization protocol and utilizes --init-points and --num-iters args. Default is False.",
    )
    parser.add_argument(
        "--seeded",
        "-s",
        dest="seeded",
        default=False,
        action="store_true",
        help="Seeds the bayesian optimization with whatever is set as the current configured poitn for the specified model. Default is False.",
    )
    parser.add_argument(
        "--init-points",
        "-ip",
        dest="init_points",
        type=int,
        default=0,
        help="Sets number of random points for the bayesian optimizer to sample before searching. Default is 0.",
    )
    parser.add_argument(
        "--num-iters",
        "-ni",
        dest="num_iters",
        type=int,
        default=1,
        help="Sets number of iterations for the bayesian optimizer to perform. Default is 1.",
    )
    parser.add_argument(
        "--from-file",
        "-ff",
        dest="from_file",
        default=False,
        action="store_true",
        help="Tells the bayesian optimizer whether to read a previous state from a file. Default is False.",
    )
    parser.add_argument(
        "--file-name",
        "-fn",
        dest="file_name",
        type=str,
        default="app/optim/bo-optimizer.json",
        help="Tells the bayesian optimizer what file to output the state to. Default is app/optim/bo-optimizer.json.",
    )
    parser.add_argument(
        "--daybyday-prints",
        "-dbdpr",
        dest="daybyday_prints",
        default=False,
        action="store_true",
        help="Calls line-by-line print statements for the final function evaluations. Default is False.",
    )
    parser.add_argument(
        "--debug-prints",
        "-dpr",
        dest="debug_prints",
        default=False,
        action="store_true",
        help="Calls debug prints to the final function evaluation. Default is False.",
    )
    parser.add_argument(
        "--debug-plots",
        "-dpl",
        dest="debug_plots",
        default=False,
        action="store_true",
        help="Calls debug plots to the final function evaluation. Default is False.",
    )
    parser.add_argument(
        "--debug-debug-plots",
        "-ddpl",
        dest="debug_debug_plots",
        default=False,
        action="store_true",
        help="Calls the debug-debug plots to the final function evaluation. Default is False.",
    )
    return parser.parse_args()


ARGS = get_args()


class BaseOptimizer:
    def __init__(self, MODEL: Model, config_data_column: str, sample_data_column: str, debug_debug_fig_title: str, debug_debug_fig_path: str):
        self.MODEL = MODEL
        self.config_data_column = config_data_column
        self.sample_data_column = sample_data_column
        self.titles = TITLES_LAMBDA(debug_debug_fig_title)
        self.debug_debug_fig_path = debug_debug_fig_path

    def _get_pred_and_true_array(self, year, z, b, daybyday_prints=False):
        # get reference data based on min year parameter
        ref_data = pd.concat(
            [
                pd.read_csv(os.path.join(os.environ["SEASON_PATH"], dir, f"{dir}_full.csv"))
                for dir in os.listdir(os.environ["SEASON_PATH"])
                if (int(dir.split("-")[0]) >= year) and not (int(dir.split("-")[0]) >= config["TEST_DATA_YEAR"])
            ],
            ignore_index=True,
        )
        test_data = pd.concat(
            [
                pd.read_csv(os.path.join(os.environ["SEASON_PATH"], dir, f"{dir}_full.csv"))
                for dir in os.listdir(os.environ["SEASON_PATH"])
                if (int(dir.split("-")[0]) >= config["TEST_DATA_YEAR"])
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
        self.MODEL.weight_func = partial(recency_weight_function, z=z, b=b)
        self.MODEL.calculate_model(ref_data)

        # looping mechanism, to check how we do live with updated models after each day of the season
        pred = []
        test_dates = test_data["GAME_gameDate"].unique()
        for game_date in test_dates:
            if daybyday_prints:
                print(">>>", game_date)
            for _, row in test_data[test_data["GAME_gameDate"] == game_date].iterrows():
                pred.append(self.MODEL.value(row, apply_mask=True)[0])
                ref_data.loc[len(ref_data)] = row
            MODEL_HOME_WIN_PR.calculate_model(ref_data)
        pred = np.array(pred)
        true = test_data["GAME_homeWin"].to_numpy()

        return pred, true

    def objective_function_tuple(self):
        raise NotImplementedError("Subclass must modify this function.")

    def objective_function_scalar(self):
        raise NotImplementedError("Subclass must modify this function.")

    def optim_models_daybyday(
        self,
        x0=None,
        year_bounds=(config["MIN_SEASON_YEAR"], config["TEST_DATA_YEAR"] - 1),
        z_bounds=(-100, -2),
        b_bounds=(1, 100),
        init_points=5,
        n_iter=5,
        verbose=2,
        from_file: str | None = None,
        to_file="app/optim/bo-optimizer.json",
    ):
        # initialize BO object (if from file or from scratch)
        optimizer = BayesianOptimization(
            f=self.objective_function_scalar,
            pbounds={
                "year": (year_bounds[0], year_bounds[1], int),
                "z": (z_bounds[0], z_bounds[1]),
                "b": (b_bounds[0], b_bounds[1]),
            },
            # acquisition_function=ExpectedImprovement(xi=xi),
            verbose=verbose,
        )

        # initial values, if supplied to the function
        if from_file is not None:
            optimizer.load_state(from_file)

        try:
            # probe x0s, if supplied
            if x0 is not None:
                optimizer.probe(params={"year": x0[0], "z": x0[1], "b": x0[2]}, lazy=True)

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

    def _gen_debug_debug_plots(self):
        # read sample data
        sample_data = pd.concat(
            [
                pd.read_csv(os.path.join(os.environ["SEASON_PATH"], dir, f"{dir}_full.csv"))
                for dir in os.listdir(os.environ["SEASON_PATH"])
                if (int(dir.split("-")[0]) >= config["REFERENCE_DATA_YEAR"])
                # not excluding the test data makes the specific linear v cubic relationship more clear
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
        sample_output = sample_data[self.sample_data_column].to_numpy()

        # loop through the test terms and generate a tuple of plot metadata
        os.makedirs(self.debug_debug_fig_path, exist_ok=True)
        print(":::: signal strengths")
        for i in range(len(TEST_TERMS)):
            # generate plot metadata
            term = TEST_TERMS[i]
            title = self.titles[i]
            binwidth = 1 if "PERCENTAGE" not in title else 0.05
            term_values = term.value(sample_data)
            bounds = (term_values.min(), term_values.max())

            # create and save plot
            print(f"FIGURE {i+1}: ", end="")
            plotting.plot_pdf_function_DEBUG(term_values, sample_output, title, binwidth, bounds)
            fig = plt.gcf()
            fig.savefig(f"{self.debug_debug_fig_path}/Figure{i+1:02d}.png")
            plt.close(fig)


class HomeSpreadOptimizer(BaseOptimizer):
    # reference the other completed optimizer(s) to implement this class
    pass


class HomeWinOptimizer(BaseOptimizer):
    def __init__(
        self,
        MODEL=MODEL_HOME_WIN_PR,
        config_data_column="HOME_WIN_PR_PARAMETERS",
        sample_data_column="GAME_homeWin",
        debug_debug_fig_title="Home Win %",
        debug_debug_fig_path=os.path.join(config["OPTIM_SAVE_PATH"], "home_win_pr_figs"),
    ):
        super().__init__(MODEL, config_data_column, sample_data_column, debug_debug_fig_title, debug_debug_fig_path)

    def objective_function_tuple(self, year, z, b, daybyday_prints=False, debug_prints=False, debug_plots=False, debug_debug_plots=False):
        pred_win, true_win = self._get_pred_and_true_array(year, z, b, daybyday_prints)
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
            print("Model variance:", self.MODEL.var)
            print("Model stdev:", self.MODEL.std)

        if debug_debug_plots:
            self._gen_debug_debug_plots()

        if debug_plots:
            plotting.plot_pdf_function(pred_win, true_win, "Predicted vs Actual Home Team Win % of NBA games", std=self.MODEL.std)
            plotting.plot_ROC_curve(pred_win, true_win, "ROC curve for Home Team Win % of NBA games%")

        return ECE, M, B, AUC, BRIER

    def objective_function_scalar(self, year, z, b):
        ECE, M, B, AUC, BRIER = self.objective_function_tuple(year, z, b)
        return (-4 * ECE**2) + (-4 * (M - 1) ** 2) + (-4 * B**2)


class TotalScoreOptimizer(BaseOptimizer):
    # reference the other completed optimizer(s) to implement this class
    pass


def optimize(optimizer):
    print("----- WELCOME TO THE OPTIMIZER -----")

    if ARGS.optimize:
        # calls of optim_models_daybyday
        best_value, next_point = optimizer.optim_models_daybyday(
            x0=(
                (config["REFERENCE_DATA_YEAR"], config[optimizer.config_data_column]["z"], config[optimizer.config_data_column]["b"])
                if ARGS.seeded
                else None
            ),
            year_bounds=(config["REFERENCE_DATA_YEAR"], config["REFERENCE_DATA_YEAR"]),
            from_file=ARGS.file_name if ARGS.from_file else None,
            to_file=ARGS.file_name,
            init_points=ARGS.init_points,
            n_iter=ARGS.num_iters,
        )

        # note that the changes need to occur in the json file as well as the code
        print("BEST VALUE:", best_value)
        print("NEXT TRIAL:", next_point)

        # final call of suggested point to the tuple function, print out results of all states
        optimizer.objective_function_tuple(
            best_value["params"]["year"],
            best_value["params"]["z"],
            best_value["params"]["b"],
            daybyday_prints=ARGS.daybyday_prints,
            debug_prints=ARGS.debug_prints,
            debug_plots=ARGS.debug_plots,
            debug_debug_plots=ARGS.debug_debug_plots,
        )

    # this function *AS IT STANDS* should be the default when no args are passed to argparse (except for which model to call, obviously)
    optimizer.objective_function_tuple(
        config["REFERENCE_DATA_YEAR"],
        config[optimizer.config_data_column]["z"],
        config[optimizer.config_data_column]["b"],
        daybyday_prints=ARGS.daybyday_prints,
        debug_prints=ARGS.debug_prints,
        debug_plots=ARGS.debug_plots,
        debug_debug_plots=ARGS.debug_debug_plots,
    )


if __name__ == "__main__":
    # update data pull, if necessary
    if config["ALLOW_DATA_DOWNLOAD"]:
        download_and_sort_data(config)
    else:
        print("!! Downloads halted by supplied config !!")

    # if uncommented, only calls the debug debug plots then exits
    # HomeWinOptimizer()._gen_debug_debug_plots()
    # exit()

    # call optimize with the correct object based on specified args
    if ARGS.model == "homeWin":
        optimize(HomeWinOptimizer())
    else:
        raise NotImplementedError

    # cleanup
    print_current_season()
    plt.show()
