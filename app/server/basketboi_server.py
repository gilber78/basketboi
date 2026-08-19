"""
TO BUILD OUT THE SERVER:
- each endpoint needs the @app.get() or @app.post() decorator, depending on which the endpoint is expecting to accept for the endpoint. This defines the url the client(s) can hit
- / is root
- the endpoint becomes localhost:8000/{endpoint here}
- then, you can define the function to invoke that returns a json containing all the necessary returns for the client to read and display properly
"""

import os
import sys
import json
import pickle
from typing import List
from pathlib import Path
from fastapi import FastAPI, Query
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.Models import Model
from server.NBASeason import NBASeason
from server.functions import get_season_year

app = FastAPI()

with open(os.path.join("app", "data", "config.json"), "r") as file:
    config = json.load(file)


class PredictionRequest(BaseModel):
    date: str
    games: list  # [str]


@app.get("/")
def root():
    return {"message": "Hello from the Basketboi server!", "name": __name__}


@app.get("/echo/")
def echo(message: str):
    return {"message": message}


@app.get("/predictions/")
def get_predictions_full(date: str, games: List[str] = Query(None)):
    return predictions_from_date(date, games)


def predictions_from_date(date: str, gameTags: list = None):  # strictly for debugging purposes
    # get necessary input data from pickled season class
    try:
        year = get_season_year(date)
        with open(
            os.path.join(os.path.join(config["DATA_DOWNLOAD_PATH"], "seasons"), f"{year}-{year+1}", f"{year}-{year+1}_season.pkl"), "rb"
        ) as file:
            target_season = pickle.load(file)
        target_season.reset_statistics()
        predictions_df = target_season.generate_game_slate_df(date, gameTags)
        if predictions_df is None:
            raise Exception("Cannot find predictions requested")
    except Exception as e:
        print(e)
        return {"date": date, "predictions": None}

    # fetch pickled model classes
    with open(os.path.join(config["MODEL_SAVE_PATH"], "MODEL_HOME_WIN_PR.pkl"), "rb") as file:
        MODEL_HOME_WIN_PR = pickle.load(file)
    # with open(os.path.join(config["MODEL_SAVE_PATH"], "MODEL_HOME_SPREAD.pkl"), "rb") as file:
    #     MODEL_HOME_SPREAD = pickle.load(file)
    # with open(os.path.join(config["MODEL_SAVE_PATH"], "MODEL_TOTAL_SCORE.pkl"), "rb") as file:
    #     MODEL_TOTAL_SCORE = pickle.load(file)

    # calculate predictions and store in dict to return
    predictions_list = []
    for _, row in predictions_df.iterrows():
        predictions_list.append(
            {
                "gameTag": row["GAME_gameTag"],
                "home_win_pr": MODEL_HOME_WIN_PR.value(row, apply_mask=True)[0],
                # MODEL_HOME_SPREAD
                # MODEL_TOTAL_SCORE
            }
        )

    return {"date": date, "predictions": predictions_list}
