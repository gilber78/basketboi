import os
import json
import kaggle
import pickle
import datetime
import pandas as pd
from NBASeason import NBASeason

from constants import MAXSIZE
from functions import DATA_DATE_FORMAT_STRING, DATA_TIME_FORMAT_STRING, get_current_season_year

with open(os.path.join("app", "data", "team_data.json"), "r") as file:
    id_list = [v["id"] for v in json.load(file).values()]

GAMES_COLUMNS_TO_KEEP = [
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
    # TODO does playoff series length influence the stats/models? Do we treat them as a separate data set? Do we include them in the data set?
]

ODDS_COLUMNS_TO_KEEP = []


def download_csv(path, file_name, download_time_filepath, dataset, quiet=False):
    kaggle.api.dataset_download_file(dataset=dataset, file_name=file_name, path=path, force=True, quiet=quiet)
    with open(download_time_filepath, "w") as file:
        file.write(datetime.datetime.now(datetime.timezone.utc).isoformat())


def sort_data_by_season(games_df: pd.DataFrame, path, min_season_year, reset_time_filepath, full=False):
    # loop throught raw dataframe, save raw data to a folder named after the year(s) in question
    os.makedirs(path, exist_ok=True)
    current_season_year = get_current_season_year()
    if full:
        with open(reset_time_filepath, "w") as file:
            file.write(datetime.datetime.now(datetime.timezone.utc).isoformat())
    for year in range(min_season_year if full else current_season_year, current_season_year + 1):
        # assumes the first day of the season occurs after September 1st of that year, and concludes before September 1st the following year, split to avoid dataframe lock
        season_games_df = games_df[
            (games_df["gameDateTimeEst"] > f"{year}-09-01 00:00:00") & (games_df["gameDateTimeEst"] < f"{year+1}-09-01 00:00:00")
        ]
        if not season_games_df.empty:
            # helper nums
            dir_path = os.path.join(path, f"{year}-{year+1}")
            raw_file_path = os.path.join(dir_path, f"{year}-{year+1}_raw.csv")
            full_file_path = os.path.join(dir_path, f"{year}-{year+1}_full.csv")
            class_file_path = os.path.join(dir_path, f"{year}-{year+1}_season.pkl")
            season = NBASeason(season_games_df)
            full_df = season.generate_and_save_full_season_data()

            # save data
            os.makedirs(dir_path, exist_ok=True)
            season_games_df.to_csv(raw_file_path, index=False)
            full_df.to_csv(full_file_path, index=False)
            with open(class_file_path, "wb") as file:
                pickle.dump(season, file, pickle.HIGHEST_PROTOCOL)
            print(f"Saved {year}-{year+1} game data to {dir_path}")


### MAIN FUNCTION/PROCESS FOR THIS FILE
def download_and_sort_data(config):
    # read from config
    DOWNLOAD_TIME_FILEPATH = os.path.join(config["DATA_DOWNLOAD_PATH"], "last_download")
    RESET_TIME_FILEPATH = os.path.join(config["DATA_DOWNLOAD_PATH"], "last_reset")
    LAST_DOWNLOAD_TIME_FORMAT_STRING = DATA_DATE_FORMAT_STRING + " " + DATA_TIME_FORMAT_STRING

    # determine if we should request kaggle and get a new set of data
    if os.path.isfile(DOWNLOAD_TIME_FILEPATH):
        with open(DOWNLOAD_TIME_FILEPATH) as file:
            last_download_time = datetime.datetime.fromisoformat(file.read())
            download_time_delta_hrs = (datetime.datetime.now(datetime.timezone.utc) - last_download_time).total_seconds() // 3600
    else:
        download_time_delta_hrs = MAXSIZE
    if os.path.isfile(RESET_TIME_FILEPATH):
        with open(RESET_TIME_FILEPATH) as file:
            last_reset_time = datetime.datetime.fromisoformat(file.read())
            reset_time_delta_days = (datetime.datetime.now(datetime.timezone.utc) - last_reset_time).total_seconds() // 86400
    else:
        reset_time_delta_days = MAXSIZE
    skip_download = download_time_delta_hrs < config["MAX_FILE_AGE_HRS"] and os.path.isdir(config["DATA_DOWNLOAD_PATH"])
    full_download = (
        reset_time_delta_days >= config["MAX_RESET_AGE_DAYS"] or config["FORCE_FILE_UPDATE"] or not os.path.isdir(config["DATA_DOWNLOAD_PATH"])
    )

    # download and sorting logic
    os.makedirs(config["DATA_DOWNLOAD_PATH"], exist_ok=True)
    if skip_download and not full_download:
        print(
            f"Using previously downloaded {config['GAME_DATA_FILE_NAME']} and {config['ODDS_DATA_FILE_NAME']} from {last_download_time.astimezone().strftime(LAST_DOWNLOAD_TIME_FORMAT_STRING)}"
        )
        return
    else:
        download_csv(
            path=config["DATA_DOWNLOAD_PATH"],
            file_name=config["GAME_DATA_FILE_NAME"],
            download_time_filepath=DOWNLOAD_TIME_FILEPATH,
            dataset=config["GAME_DATASET_NAME"],
        )
        print(f"Downloaded {config['GAME_DATA_FILE_NAME']}")
        """
        download_csv(
            path=config["DATA_DOWNLOAD_PATH"],
            file_name=config["ODDS_DATA_FILE_NAME"],
            download_time_filepath=DOWNLOAD_TIME_FILEPATH,
            dataset=config["ODDS_DATASET_NAME"],
        )
        print(f"Downloaded {config['ODDS_DATA_FILE_NAME']}")
        """

        # trim data to only needed columns
        print("Performing batch sort/cleanup on game data")
        games_raw_df = pd.read_csv(os.path.join(config["DATA_DOWNLOAD_PATH"], config["GAME_DATA_FILE_NAME"]), low_memory=False)[GAMES_COLUMNS_TO_KEEP]
        games_raw_df = games_raw_df[(games_raw_df["homeScore"] != 0) & (games_raw_df["awayScore"] != 0) & (games_raw_df["gameType"] != "Preseason")]
        if not config["INCLUDE_PLAYOFFS"]:
            games_raw_df = games_raw_df[(games_raw_df["gameType"] != "Play-in Tournament") & (games_raw_df["gameType"] != "Playoffs")]
        games_raw_df = games_raw_df.query("(hometeamId in @id_list) & (awayteamId in @id_list)")

        print("Performing batch sort/cleanup on odds data")
        # odds_raw_df = pd.read_csv(os.path.join(config["DATA_DOWNLOAD_PATH"], config["ODDS_DATA_FILE_NAME"]), low_memory=False)[ODDS_COLUMNS_TO_KEEP]
        # TODO the odds database doesn't have the data we need for 2024-2026. Put back once you find a suitable dataset.

        # season sort by year
        print("Sorting game data by season")
        if full_download:
            print("Performing full data update")
        else:
            print("Updating current season only")
        sort_data_by_season(
            games_raw_df,
            path=os.environ["SEASON_PATH"],
            min_season_year=config["MIN_SEASON_YEAR"],
            reset_time_filepath=RESET_TIME_FILEPATH,
            full=full_download,
        )
        # TODO sort data by season here
        # TODO figure out how to pick games back from sorted data...
