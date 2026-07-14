# BASKETBOI
idk what this name will be, just roll with it.
mostly a test to get used to dockering

## Setup
### API credentials
To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile settings and select 'Generate New Token'. Copy the generated token, paste and save it to a file named /app/data/access_token. This will allow you to connect to Kaggle and download the database when the program is run.

### Optimization before each season
In order to properly calibrate the model before each season start (and possibly during the season...), do the following steps:

- optimize model parameters for data starting the 5 seasons previous. For example, to prepare for the 2026/27 season, the first reference season should be 2020 and the first test season should be 2024. This gives two seasons of test data
- Then, bump up one year so the season starts with 5 seasons of data before the first games (TBD?)