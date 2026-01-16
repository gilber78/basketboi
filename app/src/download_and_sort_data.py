import os
import json
import datetime
import pandas as pd

# from tempfile import TemporaryDirectory

with open("app/data/config.json", "r") as file:
    config = json.load(file)
with open("app\\data\\name_to_id.json", "r") as file:
    ids = json.load(file)
with open(config["KAGGLE_API_TOKEN_PATH"], "r") as file:
    KAGGLE_API_TOKEN = file.read()
os.environ["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN

import kaggle

# DATA_DATE_FORMAT_STRING = "%Y-%m-%d %H:%M:%S"
COLUMNS_TO_KEEP = [
    "gameDateTimeEst",
    "hometeamCity",
    "hometeamName",
    "hometeamId",
    "awayteamCity",
    "awayteamName",
    "awayteamId",
    "homeScore",
    "awayScore",
    "winner",
    "gameType",
    # "seriesGameNumber",
    # TODO do playoffs include these stats, or can we predict something else from this?
]
SEASON_PATH = os.path.join(config["DATA_DOWNLOAD_PATH"], "seasons")
DOWNLOAD_TIME_FILEPATH = os.path.join(config["DATA_DOWNLOAD_PATH"], "last_download")


def download_csv(path=config["DATA_DOWNLOAD_PATH"], force=True, quiet=False):
    kaggle.api.dataset_download_file(dataset=config["DATASET_NAME"], file_name=config["DATA_FILE_NAME"], path=path, force=force, quiet=quiet)
    # TODO put the full games.csv into only a temporary folder, then sort and delete the raw data post sorting
    with open(DOWNLOAD_TIME_FILEPATH, "w") as file:
        file.write(str(datetime.datetime.now(datetime.timezone.utc).isoformat()))


def sort_data_by_season(df, path=SEASON_PATH):
    # loop throught raw dataframe, save raw data to a folder named after the year(s) in question
    # assumes the first day of the season occurs after August 1st of that year, and concludes before August 1st the following year
    os.makedirs(path, exist_ok=True)
    for i in range(config["MIN_SEASON_YEAR"], datetime.date.today().year + 1):
        season_df = df[(df["gameDateTimeEst"] > f"{i}-08-01 00:00:00") & (df["gameDateTimeEst"] < f"{i+1}-08-01 00:00:00")]
        if not season_df.empty:
            dir_path = os.path.join(path, f"{i}-{i+1}")
            file_path = os.path.join(dir_path, f"{i}-{i+1}_raw.csv")
            os.makedirs(dir_path, exist_ok=True)
            season_df.to_csv(file_path, index=False)
            print(f"Saved {i}-{i+1} game data to {file_path}")


### MAIN FUNCTION/PROCESS FOR THIS FILE
def download_and_sort_data():
    # determine if we should request kaggle and get a new set of data
    os.makedirs(config["DATA_DOWNLOAD_PATH"], exist_ok=True)
    if os.path.isfile(DOWNLOAD_TIME_FILEPATH) and not config["FORCE_FILE_UPDATE"]:
        print(f"Using previously downloaded {config['DATA_FILE_NAME']} from XX:XX:XXZ")  # TODO say when you last downloaded games.csv
    else:
        download_csv()
        print(f"Downloaded {config['DATA_FILE_NAME']}")
    # TODO if the last download time is older than a certain number of hours or we force it to update, update
    # and if force update, remove everything inside the config data download path

    # trim data to only needed columns, sort by season year(s)
    print("Sorting game data by season")
    raw_df = pd.read_csv(os.path.join(config["DATA_DOWNLOAD_PATH"], config["DATA_FILE_NAME"]), low_memory=False)[COLUMNS_TO_KEEP]
    raw_df = raw_df[(raw_df["gameType"] != "Preseason") & (raw_df["gameType"] != "Play-in Tournament")]
    id_list = list(ids.values())
    raw_df = raw_df.query("(hometeamId in @id_list) & (awayteamId in @id_list)")
    sort_data_by_season(raw_df)
    # TODO ^^^expand sorting logic to eliminate teams and games that don't count toward standings, particularly NBA Cup Finals and Global Games

    # thread each season folder, do the rest of preprocessing in parallel, then wait and return
