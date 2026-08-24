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
    # get important server output information
    gameTag = pred["gameTag"]
    gameDict = validate_game_tag(gameTag)
    home_win_pr = pred["home_win_pr"]
    away_win_pr = 1 - home_win_pr
    pred["away_win_pr"] = away_win_pr
    away_living_string = f"      {gameDict['awayTeamAbbreviation']}        "
    home_living_string = f"      {gameDict['homeTeamAbbreviation']}        "

    # get needed user inputs and print to terminal, update strings appropriately
    print(f"--- GAME {i+1:02d}")
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

    # TODO put input for totals
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
    print("----- WELCOME TO THE BASKETBALL COMPUTER THINGY -----")

    # update the data models
    download_and_sort_data(config)  # donwload/sort raw data, if necessary
    build_models(config)  # create plots that are passed as part of debug for model
    print_current_season()
    print()

    # get game related predictions
    with open(ARGS.input, "r") as file:
        input_json = json.load(file)
    output_json = predictions_from_date(input_json["date"], input_json["games"])

    # get betting input for each available or requested game
    print(f"There are {len(output_json['predictions'])} games on {output_json['date']}:")
    print("\033[4m                      SPREAD        ML            TOTAL     \033[0m")
    while True:
        for i, pred in enumerate(output_json["predictions"]):
            get_cli_inputs(i, pred)
        if input("Please verify that the above odds sheet is correct. Ifs so, press enter.  ") == "":
            try:
                bankroll = float(input("Great! What's the most you're willing to risk today (enter a dollar amount)? $"))
                print()
                break
            except Exception as e:
                print(e, "please try again.")
                exit()
        else:
            print("Trying again!\n")

    # TODO get ev/kelly fracctions for each bet, make suggested bet slitp
    pprint(output_json)
    pprint(bankroll)


if __name__ == "__main__":
    main()
