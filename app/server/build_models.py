import os
import pickle
import pandas as pd
from server.Models import Model, MODEL_HOME_WIN_PR  # , MODEL_HOME_SPREAD, MODEL_TOTAL_SCORE


def save_model_to_disk(model: Model, path: str):
    with open(path, "wb") as file:
        pickle.dump(model, file, pickle.HIGHEST_PROTOCOL)
        print(f"Saved {model} model data to {path}")


def build_models(config):
    # get reference data
    ref_data = pd.concat(
        [
            pd.read_csv(os.path.join(os.environ["SEASON_PATH"], dir, f"{dir}_full.csv"))
            for dir in os.listdir(os.environ["SEASON_PATH"])
            if (int(dir.split("-")[0]) >= config["MIN_REFERENCE_DATA_YEAR"])
        ],
        ignore_index=True,
    )

    # calculate out all the models
    MODEL_HOME_WIN_PR.calculate_model(ref_data)
    # MODEL_HOME_SPREAD.calculate_model(ref_data)
    # MODEL_TOTAL_SCORE.calculate_model(ref_data)

    # save models to the disk
    os.makedirs(config["MODEL_SAVE_PATH"], exist_ok=True)
    save_model_to_disk(MODEL_HOME_WIN_PR, os.path.join(config["MODEL_SAVE_PATH"], "MODEL_HOME_WIN_PR.pkl"))
    # save_model_to_disk(MODEL_HOME_SPREAD, os.path.join(config["MODEL_SAVE_PATH"], "MODEL_HOME_SPREAD.pkl"))
    # save_model_to_disk(MODEL_TOTAL_SCORE, os.path.join(config["MODEL_SAVE_PATH"], "MODEL_TOTAL_SCORE.pkl"))
