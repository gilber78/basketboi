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
from server.functions import print_current_season
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


def main():
    print("----- WELCOME TO THE BASKETBALL COMPUTER THINGY -----")

    # update the data models
    download_and_sort_data(config)  # donwload/sort raw data, if necessary
    build_models(config)  # create plots that are passed as part of debug for model
    print_current_season()
    print()

    # get game related predictions
    with open(ARGS.input, "r") as file:
        input = json.load(file)
    output = predictions_from_date(input["date"], input["games"])
    pprint(output)

    # get kelly fractions to bet for each, and which side to bet...
    # TODO work this problem - need a way to input stuff per game and get kelly fractions via CLI


if __name__ == "__main__":
    main()
