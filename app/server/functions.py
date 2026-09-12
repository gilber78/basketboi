"""
BASKETBOI

Copyright © 2026 Eric Gilbertson. All rights reserved.
See LICENSE.md for permitted use.
"""

import os
import pickle
import datetime

# Data strings/constants
DATA_DATE_FORMAT_STRING = "%Y-%m-%d"
DATA_TIME_FORMAT_STRING = "%H:%M:%S"
DAYS_PER_YEAR = 365.2425


def get_day_from_full_time(start_time: str):
    """
    Extract just the game date from the game start time (day + time)
    """
    return start_time.split(" ")[0]


def get_time_from_full_time(start_time: str):
    """
    Get game time from the full game start time (day + time)
    """
    return start_time.split(" ")[1]


def get_season_year(date: str = None):
    """
    Get the season year from the game date. Convention is that the 2025-2026 season is labeled 2025
    """
    if date is None:
        today = datetime.date.today()
    else:
        today = datetime.datetime.strptime(date, DATA_DATE_FORMAT_STRING).date()
    if today.month >= 9:
        return today.year
    else:
        return today.year - 1


def get_list_wins_and_losses(data: list):
    """
    Get the number of wins and losses from a list. This list is calculated from the last 10 series, listing whether or not the team won or lost.
    """
    return sum(data), len(data) - sum(data)


def increment_day(start_day: str, inc: int = 1):
    """
    increment datettime string by converting it to a datetime in between
    """
    start_day_object = datetime.datetime.strptime(start_day, DATA_DATE_FORMAT_STRING)
    end_day_object = start_day_object + datetime.timedelta(days=inc)
    end_day = datetime.datetime.strftime(end_day_object, DATA_DATE_FORMAT_STRING)
    return end_day


def fractional_year_since(x_date: str, ref_date: str):
    """
    Get fractional year between a current date and a reference date
    """
    x_date_object = datetime.datetime.strptime(x_date, DATA_DATE_FORMAT_STRING)
    ref_date_object = datetime.datetime.strptime(ref_date, DATA_DATE_FORMAT_STRING)
    return (x_date_object - ref_date_object).days / DAYS_PER_YEAR


def print_current_season():
    """
    Pretty print the most recent season object to the console
    """
    year = get_season_year()
    try:
        with open(os.path.join("app", "data", "games", "seasons", f"{year}-{year+1}", f"{year}-{year+1}_season.pkl"), "rb") as file:
            test_season = pickle.load(file)
        print()
        test_season.pretty_print()
    except FileNotFoundError as e:
        print()
        print(e)
        print(f"NO AVAILABLE SEASON DATA FOR {year}-{year+1} TO DISPLAY")


def validate_game_tag(gameTag: str):
    """
    Confirm that game tag strings conform to the proper form
    """
    gameTagList = gameTag.split(" ")
    if len(gameTagList) == 3 and len(gameTagList[0]) == 3 and gameTagList[1] == "@" and len(gameTagList[2]) == 3:
        return {"awayTeamAbbreviation": gameTagList[0].upper(), "homeTeamAbbreviation": gameTagList[2].upper()}
    else:
        return {"awayTeamAbbreviation": None, "homeTeamAbbreviation": None}
