import copy
import numpy as np
import pandas as pd
from functools import partial

from server.functions import fractional_year_since

np.set_printoptions(linewidth=np.inf)


def recency_weight_function(x, z, b):
    if x < z:
        val = 0
    elif x < z / 2:
        val = 2 * ((z - x) / z) ** 2
    elif x < 0:
        val = 1 - 2 * (x / z) ** 2
    else:
        val = 1
    return val**b


# TODO place the z/bs for each weight function in the config file if necessary
HOME_WIN_WEIGHT_FUNCTION = partial(recency_weight_function, z=-40.49972745901825, b=68.89789705377231)
EVEN_WEIGHT_FUNCTION = lambda x: 1


class Term:
    def __init__(self, constant_names, num_names, den_names):
        # class that holds how to calculate a model term from either the reference dataframe or a team object
        # uses a list of lists approach. Terms are added inside each list, multiplied across sublists. Can only nest 2 deep
        self.constant_names = constant_names
        self.num_names = num_names
        self.den_names = den_names
        self.degree = int(1)

    def value(self, ref_series: pd.Series):
        constant = self._sub_value(self.constant_names, ref_series)
        num = self._sub_value(self.num_names, ref_series)
        den = self._sub_value(self.den_names, ref_series)
        return (constant * num / den) ** self.degree

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

    def set_degree(self, val):
        self.degree = int(val)


class Model:
    def __init__(self, terms: list, target: str, bounds=[None, None], weight_func=EVEN_WEIGHT_FUNCTION):
        self.terms = terms
        self.target = target
        self.bounds = bounds
        self.ref_date = None
        self.coeffs = None
        self.ref_Rsquared = None
        self.m = None
        self.b = None
        self.weight_func = weight_func

    def calculate_model(self, ref_data: pd.DataFrame):
        # create values
        self.ref_date = np.max(ref_data["GAME_gameDate"])
        weights = np.array([self.weight_func(fractional_year_since(x["GAME_gameDate"], self.ref_date)) for _, x in ref_data.iterrows()])
        vandermonde = np.vstack([term.value(ref_data).to_numpy() for term in self.terms]).T
        yvals = ref_data[self.target].to_numpy().reshape(-1, 1)

        # straight up remove rows with any nan or inf
        mask = np.logical_and(~np.isnan(vandermonde).any(axis=1), ~np.isinf(vandermonde).any(axis=1))
        weights = weights[mask]
        vandermonde = vandermonde[mask]
        yvals = yvals[mask]

        # calculate, SVD on vandermonde
        Wmatrix = np.diag(weights)
        Amatrix = vandermonde.T @ Wmatrix @ vandermonde
        Bvector = vandermonde.T @ Wmatrix @ yvals
        self.coeffs = np.linalg.inv(Amatrix) @ Bvector
        # self.coeffs = np.linalg.pinv(vandermonde) @ yvals
        SS_res = sum((yvals - vandermonde @ self.coeffs) ** 2)[0]
        SS_tot = sum((yvals - np.mean(yvals)) ** 2)[0]
        self.ref_Rsquared = 1 - SS_res / SS_tot

        # calculate linear mask
        xvals = self.value(ref_data)
        xvals = xvals[mask]
        m, b = np.polyfit(xvals, yvals, 1)  # TODO experiment with mask using w keyword here
        self.m, self.b = m[0], b[0]
        # TODO estimate variance/standard deviation of the whole model, so we can z-score the probability (for spread/total predictions)

    def value(self, input_data: pd.Series, apply_mask=False):
        vals = (np.nan_to_num(np.vstack([term.value(input_data).to_numpy() for term in self.terms]).T) @ self.coeffs).T[0]
        if apply_mask:
            vals = vals * self.m + self.b
        if self.bounds[0] or self.bounds[1]:
            vals = np.clip(vals, a_min=self.bounds[0], a_max=self.bounds[1])
        return vals


# term bank, mostly for use inside this file
CONSTANT = Term([], [], [])

# home terms
HOME_WIN_PERCENTAGE = Term([], ["HOME_wins"], ["HOME_games_played"])
HOME_POINTS_FOR_PER_GAME = Term([], ["HOME_points_for"], ["HOME_games_played"])
HOME_POINTS_FOR_PER_GAME_2 = copy.deepcopy(HOME_POINTS_FOR_PER_GAME)
HOME_POINTS_FOR_PER_GAME_2.set_degree(2)
HOME_POINTS_FOR_PER_GAME_3 = copy.deepcopy(HOME_POINTS_FOR_PER_GAME)
HOME_POINTS_FOR_PER_GAME_3.set_degree(3)
HOME_POINTS_AGAINST_PER_GAME = Term([], ["HOME_points_against"], ["HOME_games_played"])
HOME_STREAK = Term([], ["HOME_streak"], [])
HOME_STREAK_2 = copy.deepcopy(HOME_STREAK)
HOME_STREAK_2.set_degree(2)
HOME_STREAK_3 = copy.deepcopy(HOME_STREAK)
HOME_STREAK_3.set_degree(3)
HOME_LAST10_W = Term([], ["HOME_last10_w"], [])
HOME_LAST10_L = Term([], ["HOME_last10_l"], [])
HOME_HOME_WIN_PERCENTAGE = Term([], ["HOME_home_wins"], ["HOME_home_games_played"])
HOME_HOME_POINTS_FOR_PER_GAME = Term([], ["HOME_home_points_for"], ["HOME_home_games_played"])
HOME_HOME_POINTS_FOR_PER_GAME_2 = copy.deepcopy(HOME_HOME_POINTS_FOR_PER_GAME)
HOME_HOME_POINTS_FOR_PER_GAME_2.set_degree(2)
HOME_HOME_POINTS_AGAINST_PER_GAME = Term([], ["HOME_home_points_against"], ["HOME_home_games_played"])
HOME_HOME_STREAK = Term([], ["HOME_home_streak"], [])
HOME_HOME_STREAK_2 = copy.deepcopy(HOME_HOME_STREAK)
HOME_HOME_STREAK_2.set_degree(2)
HOME_HOME_STREAK_3 = copy.deepcopy(HOME_HOME_STREAK)
HOME_HOME_STREAK_3.set_degree(3)
HOME_HOME_LAST10_W = Term([], ["HOME_home_last10_w"], [])
HOME_HOME_LAST10_L = Term([], ["HOME_home_last10_l"], [])

# dunno how these fit in (yet)
HOME_WIN_POINTS_FOR_PER_GAME = Term([], ["HOME_win_points_for"], ["HOME_wins"])
HOME_WIN_POINTS_AGAINST_PER_GAME = Term([], ["HOME_win_points_against"], ["HOME_wins"])
HOME_LOSS_POINTS_FOR_PER_GAME = Term([], ["HOME_loss_points_for"], ["HOME_losses"])
HOME_LOSS_POINTS_AGAINST_PER_GAME = Term([], ["HOME_loss_points_against"], ["HOME_losses"])
HOME_HOMEWIN_POINTS_FOR_PER_GAME = Term([], ["HOME_homewin_points_for"], ["HOME_home_wins"])
HOME_HOMEWIN_POINTS_AGAINST_PER_GAME = Term([], ["HOME_homewin_points_against"], ["HOME_home_wins"])
HOME_HOMELOSS_POINTS_FOR_PER_GAME = Term([], ["HOME_homeloss_points_for"], ["HOME_home_losses"])
HOME_HOMELOSS_POINTS_AGAINST_PER_GAME = Term([], ["HOME_homeloss_points_against"], ["HOME_home_losses"])

# away terms
AWAY_WIN_PERCENTAGE = Term([], ["AWAY_wins"], ["AWAY_games_played"])
AWAY_POINTS_FOR_PER_GAME = Term([], ["AWAY_points_for"], ["AWAY_games_played"])
AWAY_POINTS_FOR_PER_GAME_2 = copy.deepcopy(AWAY_POINTS_FOR_PER_GAME)
AWAY_POINTS_FOR_PER_GAME_2.set_degree(2)
AWAY_POINTS_FOR_PER_GAME_3 = copy.deepcopy(AWAY_POINTS_FOR_PER_GAME)
AWAY_POINTS_FOR_PER_GAME_3.set_degree(3)
AWAY_POINTS_AGAINST_PER_GAME = Term([], ["AWAY_points_against"], ["AWAY_games_played"])
AWAY_STREAK = Term([], ["AWAY_streak"], [])
AWAY_STREAK_2 = copy.deepcopy(AWAY_STREAK)
AWAY_STREAK_2.set_degree(2)
AWAY_STREAK_3 = copy.deepcopy(AWAY_STREAK)
AWAY_STREAK_3.set_degree(3)
AWAY_LAST10_W = Term([], ["AWAY_last10_w"], [])
AWAY_LAST10_L = Term([], ["AWAY_last10_l"], [])
AWAY_AWAY_WIN_PERCENTAGE = Term([], ["AWAY_away_wins"], ["AWAY_away_games_played"])
AWAY_AWAY_POINTS_FOR_PER_GAME = Term([], ["AWAY_away_points_for"], ["AWAY_away_games_played"])
AWAY_AWAY_POINTS_FOR_PER_GAME_2 = copy.deepcopy(AWAY_AWAY_POINTS_FOR_PER_GAME)
AWAY_AWAY_POINTS_FOR_PER_GAME_2.set_degree(2)
AWAY_AWAY_POINTS_AGAINST_PER_GAME = Term([], ["AWAY_away_points_against"], ["AWAY_away_games_played"])
AWAY_AWAY_STREAK = Term([], ["AWAY_away_streak"], [])
AWAY_AWAY_STREAK_2 = copy.deepcopy(AWAY_AWAY_STREAK)
AWAY_AWAY_STREAK_2.set_degree(2)
AWAY_AWAY_STREAK_3 = copy.deepcopy(AWAY_AWAY_STREAK)
AWAY_AWAY_STREAK_3.set_degree(3)
AWAY_AWAY_LAST10_W = Term([], ["AWAY_away_last10_w"], [])
AWAY_AWAY_LAST10_L = Term([], ["AWAY_away_last10_l"], [])

# dunno how these fit in (yet)
AWAY_WIN_POINTS_FOR_PER_GAME = Term([], ["AWAY_win_points_for"], ["AWAY_wins"])
AWAY_WIN_POINTS_AGAINST_PER_GAME = Term([], ["AWAY_win_points_against"], ["AWAY_wins"])
AWAY_LOSS_POINTS_FOR_PER_GAME = Term([], ["AWAY_loss_points_for"], ["AWAY_losses"])
AWAY_LOSS_POINTS_AGAINST_PER_GAME = Term([], ["AWAY_loss_points_against"], ["AWAY_losses"])
AWAY_AWAYWIN_POINTS_FOR_PER_GAME = Term([], ["AWAY_awaywin_points_for"], ["AWAY_away_wins"])
AWAY_AWAYWIN_POINTS_AGAINST_PER_GAME = Term([], ["AWAY_awaywin_points_against"], ["AWAY_away_wins"])
AWAY_AWAYLOSS_POINTS_FOR_PER_GAME = Term([], ["AWAY_awayloss_points_for"], ["AWAY_away_losses"])
AWAY_AWAYLOSS_POINTS_AGAINST_PER_GAME = Term([], ["AWAY_awayloss_points_against"], ["AWAY_away_losses"])

# model bank, these get exported to the other file(s) that want them
MODEL_HOME_WIN_PR = Model(
    [
        # constant
        CONSTANT,
        # home team params
        HOME_WIN_PERCENTAGE,
        HOME_POINTS_FOR_PER_GAME,
        HOME_POINTS_FOR_PER_GAME_2,
        HOME_POINTS_FOR_PER_GAME_3,
        HOME_POINTS_AGAINST_PER_GAME,
        HOME_STREAK,
        HOME_STREAK_2,
        HOME_STREAK_3,
        HOME_LAST10_W,
        HOME_LAST10_L,
        HOME_HOME_WIN_PERCENTAGE,
        HOME_HOME_POINTS_FOR_PER_GAME,
        HOME_HOME_POINTS_FOR_PER_GAME_2,
        HOME_HOME_POINTS_AGAINST_PER_GAME,
        HOME_HOME_STREAK,
        HOME_HOME_STREAK_2,
        HOME_HOME_STREAK_3,
        HOME_HOME_LAST10_W,
        HOME_HOME_LAST10_L,
        # HOME_WIN_POINTS_FOR_PER_GAME,
        # HOME_WIN_POINTS_AGAINST_PER_GAME,
        # HOME_LOSS_POINTS_FOR_PER_GAME,
        # HOME_LOSS_POINTS_AGAINST_PER_GAME,
        # HOME_HOMEWIN_POINTS_FOR_PER_GAME,
        # HOME_HOMEWIN_POINTS_AGAINST_PER_GAME,
        # HOME_HOMELOSS_POINTS_FOR_PER_GAME,
        # HOME_HOMELOSS_POINTS_AGAINST_PER_GAME,
        # away team params
        AWAY_WIN_PERCENTAGE,
        AWAY_POINTS_FOR_PER_GAME,
        AWAY_POINTS_FOR_PER_GAME_2,
        AWAY_POINTS_FOR_PER_GAME_3,
        AWAY_POINTS_AGAINST_PER_GAME,
        AWAY_STREAK,
        AWAY_STREAK_2,
        AWAY_STREAK_3,
        AWAY_LAST10_W,
        AWAY_LAST10_L,
        AWAY_AWAY_WIN_PERCENTAGE,
        AWAY_AWAY_POINTS_FOR_PER_GAME,
        AWAY_AWAY_POINTS_FOR_PER_GAME_2,
        AWAY_AWAY_POINTS_AGAINST_PER_GAME,
        AWAY_AWAY_STREAK,
        AWAY_AWAY_STREAK_2,
        AWAY_AWAY_STREAK_3,
        AWAY_AWAY_LAST10_W,
        AWAY_AWAY_LAST10_L,
        # AWAY_WIN_POINTS_FOR_PER_GAME,
        # AWAY_WIN_POINTS_AGAINST_PER_GAME,
        # AWAY_LOSS_POINTS_FOR_PER_GAME,
        # AWAY_LOSS_POINTS_AGAINST_PER_GAME,
        # AWAY_AWAYWIN_POINTS_FOR_PER_GAME,
        # AWAY_AWAYWIN_POINTS_AGAINST_PER_GAME,
        # AWAY_AWAYLOSS_POINTS_FOR_PER_GAME,
        # AWAY_AWAYLOSS_POINTS_AGAINST_PER_GAME,
    ],
    "GAME_homeWin",
    [0.05, 0.95],
    HOME_WIN_WEIGHT_FUNCTION,
)

MODEL_HOME_SPREAD = Model(
    [
        # constant
        CONSTANT,
        # home team params
        HOME_POINTS_FOR_PER_GAME,
        HOME_POINTS_AGAINST_PER_GAME,
        HOME_HOME_POINTS_FOR_PER_GAME,
        HOME_HOME_POINTS_AGAINST_PER_GAME,
        HOME_WIN_POINTS_FOR_PER_GAME,
        HOME_WIN_POINTS_AGAINST_PER_GAME,
        HOME_LOSS_POINTS_FOR_PER_GAME,
        HOME_LOSS_POINTS_AGAINST_PER_GAME,
        HOME_HOMEWIN_POINTS_FOR_PER_GAME,
        HOME_HOMEWIN_POINTS_AGAINST_PER_GAME,
        HOME_HOMELOSS_POINTS_FOR_PER_GAME,
        HOME_HOMELOSS_POINTS_AGAINST_PER_GAME,
        # away team params
        AWAY_POINTS_FOR_PER_GAME,
        AWAY_POINTS_AGAINST_PER_GAME,
        AWAY_AWAY_POINTS_FOR_PER_GAME,
        AWAY_AWAY_POINTS_AGAINST_PER_GAME,
        AWAY_WIN_POINTS_FOR_PER_GAME,
        AWAY_WIN_POINTS_AGAINST_PER_GAME,
        AWAY_LOSS_POINTS_FOR_PER_GAME,
        AWAY_LOSS_POINTS_AGAINST_PER_GAME,
        AWAY_AWAYWIN_POINTS_FOR_PER_GAME,
        AWAY_AWAYWIN_POINTS_AGAINST_PER_GAME,
        AWAY_AWAYLOSS_POINTS_FOR_PER_GAME,
        AWAY_AWAYLOSS_POINTS_AGAINST_PER_GAME,
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
        HOME_HOME_POINTS_FOR_PER_GAME,
        HOME_HOME_POINTS_AGAINST_PER_GAME,
        HOME_WIN_POINTS_FOR_PER_GAME,
        HOME_WIN_POINTS_AGAINST_PER_GAME,
        HOME_LOSS_POINTS_FOR_PER_GAME,
        HOME_LOSS_POINTS_AGAINST_PER_GAME,
        HOME_HOMEWIN_POINTS_FOR_PER_GAME,
        HOME_HOMEWIN_POINTS_AGAINST_PER_GAME,
        HOME_HOMELOSS_POINTS_FOR_PER_GAME,
        HOME_HOMELOSS_POINTS_AGAINST_PER_GAME,
        # away team params
        AWAY_POINTS_FOR_PER_GAME,
        AWAY_POINTS_AGAINST_PER_GAME,
        AWAY_AWAY_POINTS_FOR_PER_GAME,
        AWAY_AWAY_POINTS_AGAINST_PER_GAME,
        AWAY_WIN_POINTS_FOR_PER_GAME,
        AWAY_WIN_POINTS_AGAINST_PER_GAME,
        AWAY_LOSS_POINTS_FOR_PER_GAME,
        AWAY_LOSS_POINTS_AGAINST_PER_GAME,
        AWAY_AWAYWIN_POINTS_FOR_PER_GAME,
        AWAY_AWAYWIN_POINTS_AGAINST_PER_GAME,
        AWAY_AWAYLOSS_POINTS_FOR_PER_GAME,
        AWAY_AWAYLOSS_POINTS_AGAINST_PER_GAME,
    ],
    "GAME_total",
    [0, None],
)
