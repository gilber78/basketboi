import numpy as np
from scipy import optimize


def american_odds_to_payout(odds: int):
    """
    The american odds (+/- in hundreds) to the payout on a $1 stake, assuming it wins.
    This is *payout*, not profit, so the bet slip displays correctly when printed out to CLI or client
    """
    if odds >= 100:
        return 1 + abs(odds) / 100
    elif odds <= -100:
        return 1 + 100 / abs(odds)
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
        return f, prob * np.log(1 + f * (b - 1)) + (1 - prob) * np.log(1 - f)
    else:
        return f


def multi_kelly_fraction():
    raise NotImplementedError


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
            "payout_if_win": 0,
            "fraction": 0,
            "bankroll_growth": 0,
        }

        # TODO implement spread bet evaluation/comparison, once available
        pass

        # evaluate moneyline bet(s)
        away_ml_kelly_fraction, away_ml_kelly_growth = single_kelly_fraction(gameDict["away_win_pr"], gameDict["away_ml_odds"], growth=True)
        home_ml_kelly_fraction, home_ml_kelly_growth = single_kelly_fraction(gameDict["home_win_pr"], gameDict["home_ml_odds"], growth=True)
        if away_ml_kelly_fraction > home_ml_kelly_fraction:  # this checks which sign is bigger
            if away_ml_kelly_growth > current_best["bankroll_growth"]:
                current_best = {
                    "gameTag": gameTag,
                    "team": awayTeam,
                    "bet_type": "moneyline",
                    "line": True,
                    "payout_if_win": american_odds_to_payout(gameDict["away_ml_odds"]),
                    "fraction": away_ml_kelly_fraction,
                    "bankroll_growth": away_ml_kelly_growth,
                }
        else:
            if home_ml_kelly_growth > current_best["bankroll_growth"]:
                current_best = {
                    "gameTag": gameTag,
                    "team": homeTeam,
                    "bet_type": "moneyline",
                    "line": True,
                    "payout_if_win": american_odds_to_payout(gameDict["home_ml_odds"]),
                    "fraction": home_ml_kelly_fraction,
                    "bankroll_growth": home_ml_kelly_growth,
                }

        # TODO implement total bet evaluation, once available
        pass

        self.current_best = current_best


class BettingSlip:
    def __init__(self, prediction_json: dict, bankroll: float = 1.0):
        self.prediction_json = prediction_json  # maybe don't even need to store this...
        self.bankroll = bankroll

        # go through each item in the prediction json procedurally
        self.date = prediction_json["date"]
        self.bets = [OptimalBet(game) for game in prediction_json["predictions"]]

        # TODO verify that the kelly fractions for each optimal bet don't sum to greater than 1, and if so, adjust/solve these

    def pretty_print(self):
        raise NotImplementedError


if __name__ == "__main__":  # for testing only...
    test_underdog_odds = 133
    test_favorite_odds = -150
    model_favorite_prob = 0.5
    print(f"Underdog is going off at {test_underdog_odds:+d}")
    print(f"Favorite is going off at {test_favorite_odds:+d}")
    # print("-----")
    # print(f"Underdog has a {american_odds_to_probability(test_underdog_odds):.2%} chance")
    # print(f"Favorite has a {american_odds_to_probability(test_favorite_odds):.2%} chance")
    print("-----")
    print(f"$100 on the underdog will pay ${round(100*american_odds_to_payout(test_underdog_odds))}")
    print(f"$100 on the favorite will pay ${round(100*american_odds_to_payout(test_favorite_odds))}")
    print("-----")
    print(
        f"If the favorite is predicted {model_favorite_prob:.2%} to win by the model, bet ${100*single_kelly_fraction(model_favorite_prob, test_underdog_odds)} on the underdog"
    )
    print(
        f"If the favorite is predicted {model_favorite_prob:.2%} to win by the model, bet ${100*single_kelly_fraction(model_favorite_prob, test_favorite_odds)} on the favorite"
    )
