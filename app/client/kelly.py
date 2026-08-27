import os
import json
import numpy as np

with open(os.path.join("app", "data", "config.json"), "r") as file:
    TOLERANCE = json.load(file)["BISECTION_SEARCH_TOLERANCE"]


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
        self.payout = american_odds_to_payout(odds)
        self._single_fraction = self._kelly_fraction()
        self._single_growth = self._kelly_growth(self._single_fraction)
        self._multi_fraction = None
        self._multi_Value = None
        self.fraction = None

    def _kelly_fraction(self):
        return np.max([(self.prob * self.payout - 1) / (self.payout - 1), 0])

    def _kelly_growth(self, f):
        return self.prob * np.log(1 + f * (self.payout - 1)) + (1 - self.prob) * np.log(1 - f)

    def _expected_value(self):
        return self.prob * self.payout - 1

    def _lagrange_fraction(self, lam):
        if lam == 0:
            return self._kelly_fraction()
        elif lam >= self._expected_value():
            return 0
        else:
            a = self.payout * lam - lam
            b = 2 * lam - self.payout * lam - self.payout + 1
            c = self.payout * self.prob - lam - 1
            # zero_plus = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            zero_minus = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            return zero_minus

    def _lagrange_Value(self, f, lam):
        return self._kelly_growth(f) - lam * f


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
        # self.prediction_json = prediction_json  # maybe don't even need to store this...
        self.game_list = [pred["gameTag"] for pred in prediction_json["predictions"]]
        self.bankroll = bankroll
        self.bets = [None]

        # go through each item in the prediction json procedurally, then select valid (winning) bets and store them to the class
        self.date = prediction_json["date"]
        candidate_bets = [bet for game in prediction_json["predictions"] for bet in bets_from_game_json(game)]

        # check the single kelly slate, and if sum of the fractions are < 1 then accept it
        # if not, move on to multi kelly
        # clean up the bet objects either way
        single_kelly_candidates = []
        for gameTag in self.game_list:
            best_bet = max([candidate for candidate in candidate_bets if candidate.gameTag == gameTag], key=lambda bet: bet._single_growth)
            if best_bet._single_fraction > 0:
                single_kelly_candidates.append(best_bet)
        if sum([bet._single_fraction for bet in single_kelly_candidates]) <= 1:
            self.cleanup_slate(single_kelly_candidates, mode="single")
        else:
            multi_kelly_candidates = self.bisection_search(candidate_bets)
            self.cleanup_slate(multi_kelly_candidates, mode="multi")

    def bisection_search(self, candidate_bets):
        # find the initial lambda_max
        lambda_min = 0
        lambda_max = np.max([bet._expected_value() for bet in candidate_bets])

        # bisection search lies inside while condition
        while lambda_max - lambda_min > TOLERANCE:
            # recalculate values for all candidate bets
            lam = (lambda_max + lambda_min) / 2
            for bet in candidate_bets:
                bet._multi_fraction = bet._lagrange_fraction(lam)
                bet._multi_Value = bet._lagrange_Value(bet._multi_fraction, lam)

            # select new slate
            multi_kelly_candidates = []
            for gameTag in self.game_list:
                best_bet = max([candidate for candidate in candidate_bets if candidate.gameTag == gameTag], key=lambda bet: bet._multi_Value)
                if best_bet._multi_fraction > 0:
                    multi_kelly_candidates.append(best_bet)

            # adjust bisection search
            if sum([bet._multi_fraction for bet in multi_kelly_candidates]) > 1:
                lambda_min = lam
            else:
                lambda_max = lam

        return multi_kelly_candidates

    def cleanup_slate(self, bet_list, mode):
        for bet in bet_list:
            if mode == "single":
                bet.fraction = bet._single_fraction
            elif mode == "multi":
                bet.fraction = bet._multi_fraction
            else:
                raise Exception(f"Unable to clean up betting slip, invalid mode invoked {mode}")
        self.bets = bet_list

    def pretty_print(self):
        # TODO pretty print the bet slip. It's the final step before releaseing v0.0.1
        raise NotImplementedError
