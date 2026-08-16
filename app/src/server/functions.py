import os
import pickle
import datetime

from server.constants import DAYS_PER_YEAR

DATA_DATE_FORMAT_STRING = "%Y-%m-%d"
DATA_TIME_FORMAT_STRING = "%H:%M:%S"


def get_day_from_full_time(start_time: str):
    return start_time.split(" ")[0]


def get_time_from_full_time(start_time: str):
    return start_time.split(" ")[1]


def get_season_year(date: str = None):
    if date is None:
        today = datetime.date.today()
    else:
        today = datetime.datetime.strptime(date, DATA_DATE_FORMAT_STRING).date()
    if today.month >= 9:
        return today.year
    else:
        return today.year - 1


def get_list_wins_and_losses(data: list):
    return sum(data), len(data) - sum(data)


def increment_day(start_day: str, inc: int = 1):
    start_day_object = datetime.datetime.strptime(start_day, DATA_DATE_FORMAT_STRING)
    end_day_object = start_day_object + datetime.timedelta(days=inc)
    end_day = datetime.datetime.strftime(end_day_object, DATA_DATE_FORMAT_STRING)
    return end_day


def fractional_year_since(x_date: str, ref_date: str):
    x_date_object = datetime.datetime.strptime(x_date, DATA_DATE_FORMAT_STRING)
    ref_date_object = datetime.datetime.strptime(ref_date, DATA_DATE_FORMAT_STRING)
    return (x_date_object - ref_date_object).days / DAYS_PER_YEAR


def print_current_season():
    year = get_season_year()
    # TODO fix this to use the most recent season if the target pickle doesn't exist. But that's a problem for September.
    with open(os.path.join("app", "data", "games", "seasons", f"{year}-{year+1}", f"{year}-{year+1}_season.pkl"), "rb") as file:
        test_season = pickle.load(file)
    print()
    test_season.pretty_print()


def validate_game_tag(gameTag: str):
    gameTagList = gameTag.split(" ")
    if len(gameTagList) == 3 and len(gameTagList[0]) == 3 and gameTagList[1] == "@" and len(gameTagList[2]) == 3:
        return {"awayTeamAbbreviation": gameTagList[0].upper(), "homeTeamAbbreviation": gameTagList[2].upper()}
    else:
        return {"awayTeamAbbreviation": None, "homeTeamAbbreviation": None}
