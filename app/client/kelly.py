import numpy as np
from scipy.optimize import minimize


def american_odds_to_payout(odds: int):
    """
    The american odds (+/- in hundreds) to the payout on a $1 stake, assuming it wins.
    This is *payout*, not profit, so the bet slip displays correctly when printed out to CLI or client
    """
    if odds >= 100:
        return 1 + np.abs(odds) / 100
    elif odds <= -100:
        return 1 + 100 / np.abs(odds)
    return None


'''
def american_odds_to_probability(odds: int):
    """
    The american odds (+/- in hundreds) to probability between 0 and 1
    # TODO delete/comment out, this may not be necessary
    """
    if odds >= 100:
        return 100 / (abs(odds) + 100)
    elif odds <= -100:
        return abs(odds) / (abs(odds) + 100)
    return None
'''


def single_kelly_fraction(prob: float, odds: int, growth=False):
    """
    outputs the standard kelly fraction, for one wager and one wager only
    """
    b = american_odds_to_payout(odds)
    f = (prob * b - 1) / (b - 1)
    if growth:
        return f, single_kelly_growth(prob, b, f)
    else:
        return f


def single_kelly_growth(prob: float, b: float, f: float):
    # just the growth parameter
    return prob * np.log(1 + f * (b - 1)) + (1 - prob) * np.log(1 - f)


def solve_multi_kelly_fractions():
    raise NotImplementedError
    # looks like we need to try and solve for an x (lambda/Lagrange multiplier) such that the sum of f_i equals 1.
    # We can supposedly solve for each f_i from the lambda, and use a bisection method(?) to find the optimal x*
    # the, f* is the outputs from x*
    #
    # looks like scipy.optimize.minimize can do KKT with minimizing an objective f(x) and inequality constraints g(x) >= 0
    # result = minimize(objective, initial_guess, method='SLSQP', constraints=[constraints]); {'type': 'ineq', 'fun': constraint1}
    # so for f(x) = -sum(single_kelly_growths) <- These need to be partials of single kelly growths...
    # with g_1(x) = 1 - sum(f_i) >= 0
    # and g_i+1(x) = f_i >= 0
    # ... figure it out from there...


class OptimalBet:
    def __init__(self, gameDict: dict):
        gameTag = gameDict["gameTag"]
        awayTeam = gameDict["awayTeam"]
        homeTeam = gameDict["homeTeam"]
        current_best = {
            "gameTag": gameTag,
            "team": None,
            "bet_type": None,
            "line": None,
            "odds": None,
            "probability_of_win": None,
            "payout_if_win": None,
            "fraction": None,
            "bankroll_growth": 0,
        }

        # TODO implement point spread bet evaluation/comparison, once available
        pass

        # evaluate moneyline bet(s)
        away_ml_kelly_fraction, away_ml_kelly_growth = single_kelly_fraction(gameDict["away_win_pr"], gameDict["away_ml_odds"], growth=True)
        home_ml_kelly_fraction, home_ml_kelly_growth = single_kelly_fraction(gameDict["home_win_pr"], gameDict["home_ml_odds"], growth=True)
        if away_ml_kelly_fraction > home_ml_kelly_fraction:  # this checks which sign is bigger
            if (away_ml_kelly_fraction > 0) & (away_ml_kelly_growth > current_best["bankroll_growth"]):
                # this checks that the proposed wager is positive kelly and is better than the current best wager for the game. If so, update
                current_best = {
                    "gameTag": gameTag,
                    "team": awayTeam,
                    "bet_type": "moneyline",
                    "line": True,
                    "odds": gameDict["away_ml_odds"],
                    "probability_of_win": gameDict["away_win_pr"],
                    "payout_if_win": american_odds_to_payout(gameDict["away_ml_odds"]),
                    "fraction": away_ml_kelly_fraction,
                    "bankroll_growth": away_ml_kelly_growth,
                }
        else:
            if (home_ml_kelly_fraction > 0) & (home_ml_kelly_growth > current_best["bankroll_growth"]):
                # this checks that the proposed wager is positive kelly and is better than the current best wager for the game. If so, update
                current_best = {
                    "gameTag": gameTag,
                    "team": homeTeam,
                    "bet_type": "moneyline",
                    "line": True,
                    "odds": gameDict["home_ml_odds"],
                    "probability_of_win": gameDict["home_win_pr"],
                    "payout_if_win": american_odds_to_payout(gameDict["home_ml_odds"]),
                    "fraction": home_ml_kelly_fraction,
                    "bankroll_growth": home_ml_kelly_growth,
                }

        # TODO implement point total bet evaluation evaluation/comparison, once available
        pass

        self.current_best = current_best


class BettingSlip:
    def __init__(self, prediction_json: dict, bankroll: float = 1.0):
        self.prediction_json = prediction_json  # maybe don't even need to store this...
        self.bankroll = bankroll

        # go through each item in the prediction json procedurally, then select valid (winning) bets and store them to the class
        self.date = prediction_json["date"]
        raw_bets = [OptimalBet(game) for game in prediction_json["predictions"]]
        self.bets = [bet for bet in raw_bets if bet.current_best["bet_type"] is not None]

        if np.sum([bet.current_best["fraction"] for bet in self.bets]) <= 1:
            print("------- YAY No adjustments necessary!")
        else:
            print("<!:::!> darn. gotta figure something out.")

        # TODO verify that the kelly fractions for each optimal bet don't sum to greater than 1, and if so, adjust/solve these
        # honestly, we could just scale them all back to 1 but keep the proportions the same. That way, we honor the user's desired spending.
        # but first, let's just see if we can find a day that for -110 or 100 gives us a >1 then go from there.

    def pretty_print(self):
        raise NotImplementedError
