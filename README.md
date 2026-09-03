# Basketball Automated Statistical Knowledge Engine for Tracking Bets, Odds, & Intelligence
Otherwise known as: BASKETBOI!

This program is a pet project meant to improve my full-stack development. Its *secondary* purpose is to provide a model for sports betting on NBA games. The model(s) present take historical information about team performance heading into a game (wins/losses, points per game, current streak, etc) and make predictions based on the average expected performance of both teams. It assigns probabilities and takes user inputs about betting odds being laid, and informs the user which wagers are most favorable for their bankroll and how much money to place on each.

## Setup - API credentials
To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile settings and select 'Generate New Token'. Copy the generated token, paste and save it to a file named /app/data/access_token. This will allow you to connect to Kaggle and download the database when the program is run.

## To run from CLI
1) From the top level of this directory, use the the file located in `app\client\input_template.json` and create the inputs for a given day. For games that happened in the past, the program will automatically find those games. For games happening today or in the future, you must specify a list of game strings in the format `"AWY @ HME"` and have the json dated correctly
2) Simply call `python app/client/main.py -i /path/to/json`
3) Follow the prompts after the program downloads the newest NBA data and updates the model(s) to enter the odds for every game (only moneyline supported for now)
4) Double check the odds sheet and press enter, as prompted
5) Enter the total bankroll/daily outlay you wish to use, when prompted
6) The program will print a list of suggested bets and some supporting data. If no profitable bets were found, the program will output `"NO BETS RECOMMENDED TODAY"` in red bold. Otherwise, simply place the recommended bets and best of luck!
7) https://www.oddsshopper.com/tools/betting-calculators/odds can be used to convert kalshi prices to american odds, if needed while using the web interface.

*Please remember that gambling of any kind should be done at your own risk, this tool is not proven to offer accurate predictions or sound financial advice. Call 1-800-GAMBLER if you or someone you know struggles with gambling addiction*

## Development Notes
### Server-client [in progress]
- Install the dependencies from requirements.txt [coming soon]
- Install uvicorn, which runs the server `pip install uvicorn`
- On the machine meant to host the functionality, run `uvicorn app.server.basketboi_server:app --host 127.0.0.1 --port 8000` from the project level directory. If you are debugging, feel free to add the `--reload` flag so that updates to server.py are processed in real time
- then, running `client.py` should connect appropriately

### Optimization before each season
In order to properly calibrate the model before each season start (and possibly during the season...), do the following steps:

- Start by selecting model year parameters for data starting the 5 seasons previous. For example, to prepare for the 2026/27 season, the first reference season should be 2020 and the first test season should be 2024. This gives 5 (total) seasons worth of reference data and 2 seasons worth of test data, and sufficient room to find optimal weight function combos
- Run the optimizer_day_by_day inside `app/optim/optimze.py` to acquire the best z/b parameters for each model
- update the values inside `app/data/config.json` to reflect the otpimal values before tagging a season-specific release
- (optional) verify the polynomials by calling the functions inside optimize.py with `debug_debug_plots` set True, but I forsee this not being a super common modification.

## Extra note
- Before pushing to git, make sure all of the following run (this is my really bad version of "regression test", if it were...):
    - `black .`
    - `python app/client/main.py`
    - `time python app/optim/optimize.py`
    - `python app/server/basketboi_server.py` (both from cli and uvicorn, in concert with `python app/client/basketboi_client.py`)

#### TODO must make minor tweaks and tag release before 10/20/2026, and get an MVP before the start of this season. This could be
- (high) finalize, for once and for all, with the term by term comparisons (with historical weighting considered...)
- (med) accept both Kalshi price in CLI, along with American odds (configurable)?
- (low) argparseify optimize.py