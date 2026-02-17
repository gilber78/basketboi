import os
import json

with open("app/data/config.json", "r") as file:
    config = json.load(file)
    os.environ["SEASON_PATH"] = os.path.join(config["DATA_DOWNLOAD_PATH"], "seasons")
with open(config["KAGGLE_API_TOKEN_PATH"], "r") as file:
    os.environ["KAGGLE_API_TOKEN"] = file.read()

import matplotlib.pyplot as plt
from build_models import build_models
from download_and_sort_data import download_and_sort_data  # this import has to come last


def main():
    print("----- WELCOME TO THE BASKETBALL COMPUTER THINGY -----")
    download_and_sort_data(config)  # donwload/sort raw data, if necessary
    build_models(config)  # create plots that are passed as part of debug for model
    if config["DEBUG"]:
        plt.show()  # show plots, if in debug mode


if __name__ == "__main__":
    main()
