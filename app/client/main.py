import os
import sys
import json
import argparse
from pathlib import Path
from pprint import pprint

with open(os.path.join("app", "data", "config.json"), "r") as file:
    config = json.load(file)
    os.environ["SEASON_PATH"] = os.path.join(config["DATA_DOWNLOAD_PATH"], "seasons")
with open(config["KAGGLE_API_TOKEN_PATH"], "r") as file:
    os.environ["KAGGLE_USERNAME"] = file.readline()
    os.environ["KAGGLE_KEY"] = file.readline()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kelly import BettingSlip
from basketboi_client import process_server_output_json
from server.build_models import build_models
from server.functions import print_current_season, validate_game_tag
from server.basketboi_server import predictions_from_date
from server.download_and_sort_data import download_and_sort_data  # this import has to come last

parser = argparse.ArgumentParser("BASKETBOI cli function")
parser.add_argument(
    "--input-json",
    "-i",
    dest="input",
    type=str,
    default=os.path.join("app", "client", "input_template.json"),
    help="input json from which predictions will be calculated",
)
ARGS = parser.parse_args()


def get_cli_inputs(i, pred):
    # set up cli strings to print
    gameDict = validate_game_tag(pred["gameTag"])
    away_living_string = f"      {pred['awayTeam']}        "
    home_living_string = f"      {pred['homeTeam']}        "

    # get needed user inputs and print to terminal, update strings appropriately
    print(f"\033[1m   GAME {i+1:02d}\033[0m")
    print(away_living_string)
    print(home_living_string)

    # TODO put back input for spreads and spread odds
    """
    away_spread_line = float(
        input(f"\033[A\033[A{away_living_string}        <- Enter {gameDict['awayTeamAbbreviation']} point spread line\r{away_living_string} ")
    )
    away_living_string += f"{away_spread_line:>+6.1f} "
    print("\r\033[A" + away_living_string + "\033[K")
    home_spread_line = -1 * away_spread_line
    home_living_string += f"{home_spread_line:>+6.1f} "
    print(home_living_string + "\033[K")
    pred["away_spread_line"] = away_spread_line
    pred["home_spread_line"] = home_spread_line
    away_spread_odds = int(
        input(f"\033[A\033[A{away_living_string}        <- Enter {gameDict['awayTeamAbbreviation']} point spread odds\r{away_living_string} ")
    )
    away_living_string += f"{away_spread_odds:>+5d}    "
    print("\r\033[A" + away_living_string + "\033[K")
    home_spread_odds = int(input(f"{home_living_string}        <- Enter {gameDict['homeTeamAbbreviation']} point spread odds\r{home_living_string} "))
    home_living_string += f"{home_spread_odds:>+5d}    "
    print("\r\033[A" + home_living_string + "\033[K")
    pred["away_spread_odds"] = away_spread_odds
    pred["home_spread_odds"] = home_spread_odds
    """
    away_living_string += f"+/-.5  xxxx    "
    home_living_string += f"+/-.5  xxxx    "

    # input for moneyline odds
    away_ml_odds = int(input(f"\033[A\033[A{away_living_string}        <- Enter {gameDict['awayTeamAbbreviation']} ML odds\r{away_living_string} "))
    away_living_string += f"{away_ml_odds:>+6d}    " + " o000.5  xxxx"  # TODO delete when the totals are back
    print("\r\033[A" + away_living_string + "\033[K")
    home_ml_odds = int(input(f"{home_living_string}        <- Enter {gameDict['homeTeamAbbreviation']} ML odds\r{home_living_string} "))
    home_living_string += f"{home_ml_odds:>+6d}    " + " u000.5  xxxx"  # TODO delete when the totals are back
    print("\r\033[A" + home_living_string + "\033[K")
    pred["away_ml_odds"] = away_ml_odds
    pred["home_ml_odds"] = home_ml_odds

    # TODO put back input for totals and totals odds
    """
    point_total_line = float(input(f"\033[A\033[A{away_living_string}        <- Enter {gameTag} point total line\r{away_living_string} "))
    away_living_string += f" o{point_total_line:>4.1f} "
    print("\r\033[A" + away_living_string + "\033[K")
    home_living_string += f" u{point_total_line:>4.1f} "
    print(home_living_string + "\033[K")
    pred["point_total_line"] = point_total_line
    total_over_odds = int(input(f"\033[A\033[A{away_living_string}        <- Enter {gameTag} point total over odds\r{away_living_string} "))
    away_living_string += f"{total_over_odds:>+5d}    "
    print("\r\033[A" + away_living_string + "\033[K")
    total_under_odds = int(input(f"{home_living_string}        <- Enter {gameTag} point total under odds\r{home_living_string} "))
    home_living_string += f"{total_under_odds:>+5d}    "
    print("\r\033[A" + home_living_string + "\033[K")
    pred["total_over_odds"] = total_over_odds
    pred["total_under_odds"] = total_under_odds
    """


def main():
    print("===== WELCOME TO BASKETBOI! =====")

    # update the data models
    if config["ALLOW_DATA_DOWNLOAD"]:
        download_and_sort_data(config)  # donwload/sort raw data, if necessary
        build_models(config)  # create plots that are passed as part of debug for model
    else:
        print("!! Downloads halted by supplied config !!")
        print("-> No model updates initiated as a result")
    print_current_season()
    print()

    # get game related predictions
    with open(ARGS.input, "r") as file:
        input_json = json.load(file)
    output_json = process_server_output_json(predictions_from_date(input_json["date"], input_json["games"]))

    # get betting input for each available or requested game
    bankroll = 0
    if output_json["predictions"] is not None:
        print(f"There are {len(output_json['predictions'])} games on {output_json['date']}:")
        print("\033[4m#                     SPREAD        ML            TOTAL                                            #\033[0m")
        while True:
            try:
                for i, pred in enumerate(output_json["predictions"]):
                    get_cli_inputs(i, pred)
                if input("Please verify that the above odds sheet is correct. If so, press enter. ") == "":
                    while True:
                        bankroll = float(input("Great! What's the most you're willing to risk today (enter a dollar amount)? $"))
                        if bankroll >= 1:
                            print()
                            break
                        else:
                            print("Please enter a bet number greater than or equal to $1.00")
                    break
                else:
                    print("Trying again!\n")
            except Exception as e:
                print(e, "please try again.")
                exit()
    else:
        print(f"There are no predictions available for games on {output_json['date']}:\n")

    # get ev/kelly fractions for each bet, make suggested bet slip, return to user
    betting_slip = BettingSlip(output_json, bankroll)
    """ debug stuff I don't wanna delete yet
    print("===== output json ====")
    pprint(output_json)
    print()
    print("----- betting slip -----")
    for bet in betting_slip.bets:
        if bet is not None:
            pprint(bet.__dict__)
        else:
            pprint(bet)
    print(f"percent bankroll consumed: {sum([bet.fraction for bet in betting_slip.bets])}")
    print(f"number of good bets: {len(betting_slip.bets)}")
    print()
    """

    # pretty print the betting slip
    betting_slip.pretty_print()
    print("All done! May the odds be ever in your favor :)")
    print()


if __name__ == "__main__":
    main()
