# BASKETBOI
idk what this name will be, just roll with it.
mostly a test to get used to dockering

## Setup
### API credentials
To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile settings and select 'Generate New Token'. Copy the generated token, paste and save it to a file named /app/data/access_token. This will allow you to connect to Kaggle and download the database when the program is run.

### Server-client [in progress]
- Install the dependencies from requirements.txt [coming soon]
- Install uvicorn, which runs the server `pip install uvicorn`
- On the machine meant to host the functionality, run `uvicorn app.src.server.server:app --host 127.0.0.1 --port 8000` from the project level directory. If you are debugging, feel free to add the `--reload` flag so that updates to server.py are processed in real time
- then, running `client.py` should connect appropriately

### Optimization before each season
In order to properly calibrate the model before each season start (and possibly during the season...), do the following steps:

- optimize model parameters for data starting the 5 seasons previous. For example, to prepare for the 2026/27 season, the first reference season should be 2020 and the first test season should be 2024. This gives two seasons of test data
- Then, bump up one year so the season starts with 5 seasons of data before the first games (TBD?)

## Extra note
- Before pushing to git, make sure all of the following run (this is my really bad version of "regression test", if it were...):
    - `black .`
    - `python app/src/main.py`
    - `python app/src/optimize.py`
    - `python app/src/server/server.py` (both from cli and uvicorn, in concert with `python app/src/client/client.py`)

TODO refactor server everything to exist inside server/client, remove src folder
