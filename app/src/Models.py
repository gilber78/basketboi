import numpy as np
import pandas as pd

np.set_printoptions(linewidth=np.inf)


# TODO allow Term class to accept model as constant value
class Term:
    def __init__(self, constant_names, num_names, den_names):
        # class that holds how to calculate a model term from either the reference dataframe or a team object
        # uses a list of lists approach. Terms are added inside each list, multiplied across sublists. Can only nest 2 deep
        self.constant_names = constant_names
        self.num_names = num_names
        self.den_names = den_names

    def value(self, ref_series: pd.Series):
        constant = self._sub_value(self.constant_names, ref_series)
        num = self._sub_value(self.num_names, ref_series)
        den = self._sub_value(self.den_names, ref_series)
        return constant * num / den

    def _sub_value(self, item_list, ref_series=pd.DataFrame):
        if item_list == []:
            return pd.Series([1] * len(ref_series))
        else:
            subval = None
            for item in item_list:
                if type(item) == list:
                    if not subval:
                        subval = 1
                    subval *= self._sub_value(item, ref_series)
                else:
                    if not subval:
                        subval = 0
                    subval += ref_series[item]
            return subval


# term bank, mostly for use inside this file
CONSTANT = Term([], [], [])

HOME_WIN_PERCENTAGE = Term([], ["HOME_wins"], ["HOME_games_played"])
HOME_POINTS_FOR_PER_GAME = Term([], ["HOME_points_for"], ["HOME_games_played"])
HOME_POINTS_AGAINST_PER_GAME = Term([], ["HOME_points_against"], ["HOME_games_played"])
HOME_STREAK = Term([], ["HOME_streak"], [])
HOME_LAST10_W = Term([], ["HOME_last10_w"], [])
HOME_LAST10_L = Term([], ["HOME_last10_l"], [])

AWAY_WIN_PERCENTAGE = Term([], ["AWAY_wins"], ["AWAY_games_played"])
AWAY_POINTS_FOR_PER_GAME = Term([], ["AWAY_points_for"], ["AWAY_games_played"])
AWAY_POINTS_AGAINST_PER_GAME = Term([], ["AWAY_points_against"], ["AWAY_games_played"])
AWAY_STREAK = Term([], ["AWAY_streak"], [])
AWAY_LAST10_W = Term([], ["AWAY_last10_w"], [])
AWAY_LAST10_L = Term([], ["AWAY_last10_l"], [])


class Model:
    def __init__(self, terms: list, target: str, bounds=[None, None]):
        self.terms = terms
        self.target = target
        self.bounds = bounds

    def calculate_model(self, ref_data: pd.DataFrame):
        vandermode = np.vstack([term.value(ref_data).to_numpy() for term in self.terms]).T
        # TODO configure how to handle zeros/NaN in the model data (ie first games of the season)
        # How should these values these values be set to and/or removed?
        # make sure yvals matches in size if applicable
        vandermode = np.nan_to_num(vandermode)
        yvals = ref_data[self.target].to_numpy().reshape(-1, 1)
        self.coeffs = np.linalg.pinv(vandermode) @ yvals
        SS_res = sum((yvals - vandermode @ self.coeffs) ** 2)[0]
        SS_tot = sum((yvals - np.mean(yvals)) ** 2)[0]
        self.ref_Rsquared = 1 - SS_res / SS_tot

    def value(self, input_data: pd.Series):
        vals = (np.nan_to_num(np.vstack([term.value(input_data).to_numpy() for term in self.terms]).T) @ self.coeffs).T[0]
        if self.bounds[0] or self.bounds[1]:
            return np.clip(vals, a_min=self.bounds[0], a_max=self.bounds[1])
        return vals


# model bank, these get exported to the other file(s) that want them
# TODO expand the available terms to home/away/win/loss combos, since those may make a difference too
# TODO add line of best fit mask
MODEL_HOME_WIN_PR = Model(
    [
        # constant
        CONSTANT,
        # home team params
        HOME_WIN_PERCENTAGE,
        HOME_POINTS_FOR_PER_GAME,
        HOME_POINTS_AGAINST_PER_GAME,
        HOME_STREAK,
        HOME_LAST10_W,
        HOME_LAST10_L,
        # away team params
        HOME_WIN_PERCENTAGE,
        AWAY_POINTS_FOR_PER_GAME,
        AWAY_POINTS_AGAINST_PER_GAME,
        AWAY_STREAK,
        AWAY_LAST10_W,
        AWAY_LAST10_L,
    ],
    "GAME_homeWin",
    [0, 1],
)
MODEL_HOME_SPREAD = Model(
    [
        # constant
        CONSTANT,
        # home team params
        HOME_POINTS_FOR_PER_GAME,
        HOME_POINTS_AGAINST_PER_GAME,
        # away team params
        AWAY_POINTS_FOR_PER_GAME,
        AWAY_POINTS_AGAINST_PER_GAME,
    ],
    "GAME_spread",
)
MODEL_TOTAL_SCORE = Model(
    [
        # constant
        CONSTANT,
        # home team params
        HOME_POINTS_FOR_PER_GAME,
        HOME_POINTS_AGAINST_PER_GAME,
        # away team params
        AWAY_POINTS_FOR_PER_GAME,
        AWAY_POINTS_AGAINST_PER_GAME,
    ],
    "GAME_total",
    [0, None],
)
