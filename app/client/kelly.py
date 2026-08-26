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
    # b = american_odds_to_payout(odds)
    return prob * np.log(1 + f * (b - 1)) + (1 - prob) * np.log(1 - f)


class Bet:
    def __init__(
        self,
        gameTag: str = None,
        team: str = None,
        bet_type: str = None,
        line=None,
        odds: int = None,
        prob: float = None,
    ):
        self.gameTag = gameTag
        self.team = team
        self.bet_type = bet_type
        self.line = line
        self.odds = odds
        self.prob = prob
        self.fraction = None
        self.growth = None


def bets_from_game_json(game: dict):
    # point spread for both teams
    # away_spread = Bet()
    # home_spread = Bet()

    # moneyline for both teams
    away_moneyline = Bet(
        gameTag=game["gameTag"],
        team=game["awayTeam"],
        bet_type="moneyline",
        line=True,
        odds=game["away_ml_odds"],
        prob=game["away_win_pr"],
    )
    home_moneyline = Bet(
        gameTag=game["gameTag"],
        team=game["homeTeam"],
        bet_type="moneyline",
        line=True,
        odds=game["home_ml_odds"],
        prob=game["home_win_pr"],
    )

    # point total for over and under
    # over = Bet()
    # under = Bet()

    return away_moneyline, home_moneyline


class BettingSlip:
    def __init__(self, prediction_json: dict, bankroll: float = 1.0):
        self.prediction_json = prediction_json  # maybe don't even need to store this...
        self.bankroll = bankroll

        # go through each item in the prediction json procedurally, then select valid (winning) bets and store them to the class
        self.date = prediction_json["date"]
        self.candidate_bets = [bet for game in prediction_json["predictions"] for bet in bets_from_game_json(game)]
        for candidate in self.candidate_bets:
            print(candidate.__dict__)
        # self.solve_multi_kelly_fractions()
        # TODO fix the solve_multi_kelly_fractions dilemma
        # is it scipy minimize? or is it the binary section search?

    def solve_multi_kelly_fractions(self):
        f0 = np.array([bet.current_best["fraction"] for bet in self.bets])
        print("INITIAL F*:", f0, sum(f0))

        def objective_function(fn: np.ndarray):
            growth_rates = [
                single_kelly_growth(bet.current_best["probability_of_win"], bet.current_best["payout_if_win"], fn[i])
                for i, bet in enumerate(self.bets)
            ]
            return -1 * np.sum(growth_rates)

        def constraint_function(fn: np.ndarray):
            return 1 - np.sum(fn)

        constraints = {"type": "ineq", "fun": constraint_function}
        bounds = [(0, 1) for i in range(len(f0))]
        result = minimize(objective_function, f0, method="SLSQP", constraints=constraints, bounds=bounds)
        print("RESULT:", result.x, sum(result.x))
        print(result.multipliers)  # TODO check this from home computer, since this isn't working on python 3.10 dev computer

    def pretty_print(self):
        raise NotImplementedError
