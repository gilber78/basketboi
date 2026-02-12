import pandas as pd
from functions import get_day_from_full_time, get_time_from_full_time


class NBAGame:
    def __init__(self, data: pd.Series):
        try:
            self.gameDate = get_day_from_full_time(data["gameDateTimeEst"])
            self.gameTime = get_time_from_full_time(data["gameDateTimeEst"])
            self.homeTeam = data["hometeamCity"] + " " + data["hometeamName"]
            self.homeTeamId = data["hometeamId"]
            self.awayTeam = data["awayteamCity"] + " " + data["awayteamName"]
            self.awayTeamId = data["awayteamId"]
            self.homeScore = data["homeScore"]
            self.awayScore = data["awayScore"]
            self.total = self.awayScore + self.homeScore
            self.spread = self.awayScore - self.homeScore
            self.winner = data["winner"]
            self.gameType = data["gameType"]
        except Exception as e:
            print(f"An error occurred while initializing NBAGame object from file: {e}")

    def pretty_print(self):
        print(f"{self.gameDate} {self.gameTime} - {self.gameType}")
        print(f"    {self.awayTeam:<26} | {self.awayScore:>3d}  {self.total:>3d}")
        print(f"    {self.homeTeam:<26} | {self.homeScore:>3d}  {self.spread:>+3d}")
