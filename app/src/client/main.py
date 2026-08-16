import os
import sys
import json
from pathlib import Path

with open(os.path.join("app", "data", "config.json"), "r") as file:
    config = json.load(file)
    os.environ["SEASON_PATH"] = os.path.join(config["DATA_DOWNLOAD_PATH"], "seasons")
with open(config["KAGGLE_API_TOKEN_PATH"], "r") as file:
    os.environ["KAGGLE_USERNAME"] = file.readline()
    os.environ["KAGGLE_KEY"] = file.readline()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.build_models import build_models
from server.functions import print_current_season
from server.download_and_sort_data import download_and_sort_data  # this import has to come last


def update():
    print("----- WELCOME TO THE BASKETBALL COMPUTER THINGY -----")
    download_and_sort_data(config)  # donwload/sort raw data, if necessary
    build_models(config)  # create plots that are passed as part of debug for model


if __name__ == "__main__":
    update()
    print_current_season()
