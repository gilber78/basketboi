import pickle
import datetime

DATA_DATE_FORMAT_STRING = "%Y-%m-%d"
DATA_TIME_FORMAT_STRING = "%H:%M:%S"


def get_day_from_full_time(start_time):
    return start_time.split(" ")[0]


def get_time_from_full_time(start_time):
    return start_time.split(" ")[1]


def get_current_season_year():
    today = datetime.date.today()
    if today.month >= 8:
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


def print_current_season():
    year = get_current_season_year()
    with open(f"app\\data\\games\\seasons\\{year}-{year+1}\\{year}-{year+1}_season.pkl", "rb") as file:
        test_season = pickle.load(file)
    print()
    test_season.pretty_print()
