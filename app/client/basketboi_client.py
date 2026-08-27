import sys
import requests
from pathlib import Path
from pprint import pprint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.functions import validate_game_tag

"""
Documentation goes here for client
"""


def process_server_output_json(output_json):
    # this function takes the minimal json returned from the server and fleshes it out with more complete data for each game available
    if output_json["predictions"] is None:
        return output_json

    for pred in output_json["predictions"]:
        gameDict = validate_game_tag(pred["gameTag"])
        pred["awayTeam"] = gameDict["awayTeamAbbreviation"]
        pred["homeTeam"] = gameDict["homeTeamAbbreviation"]
        home_win_pr = pred["home_win_pr"]
        away_win_pr = 1 - home_win_pr
        pred["away_win_pr"] = away_win_pr

    return output_json


if __name__ == "__main__":
    response = requests.get("http://localhost:8000")
    print("Status code:", response.status_code)
    print("Response:")
    pprint(response.json())
    print(response.url)
    print()

    response = requests.get("http://localhost:8000/echo", params={"message": "Hello from the client!"})
    print("Status code:", response.status_code)
    print("Response:")
    pprint(response.json())
    print(response.url)
    print()

    response = requests.get("http://localhost:8000/predictions/", params={"date": "2018-01-15"})
    print("Status code:", response.status_code)
    print("Response:")
    pprint(process_server_output_json(response.json()))
    print(response.url)
    print()

    response = requests.get("http://localhost:8000/predictions/", params={"date": "2018-05-15"})
    print("Status code:", response.status_code)
    print("Response:")
    pprint(process_server_output_json(response.json()))
    print(response.url)
    print()

    response = requests.get(
        "http://localhost:8000/predictions/",
        params={"date": "2018-05-15", "games": ["LAL @ MIN", "DEN @ MIL"]},
    )
    print("Status code:", response.status_code)
    print("Response:")
    pprint(process_server_output_json(response.json()))
    print(response.url)

    # http://localhost:8000/predictions?date=YYYY-MM-DD&games=AWY+%40+HME
