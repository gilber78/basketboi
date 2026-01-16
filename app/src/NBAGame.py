import pandas as pd


class NBAGame:
    def __init__(self, data: pd.Series):
        try:
            self.gameDate = data["gameDateTimeEst"].split(" ")[0]
            self.gameTime = data["gameDateTimeEst"].split(" ")[1]
            self.homeTeam = data["hometeamCity"] + " " + data["hometeamName"]
            self.homeTeamId = data["hometeamId"]
            self.awayTeam = data["awayteamCity"] + " " + data["awayteamName"]
            self.awayTeamId = data["awayteamId"]
            self.homeScore = data["homeScore"]
            self.awayScore = data["awayScore"]
            self.winner = data["winner"]
            self.gameType = data["gameType"]
        except Exception as e:
            print(f"An error occurred while initializing NBAGame object from file: {e}")

    def pretty_print(self):  # TODO expand NBAGame pretty print
        print(f"{self.gameDate} {self.gameTime} | {self.gameType}")
        print(f"    {self.awayTeam:<28} {self.awayScore:>3d}")
        print(f"    {self.homeTeam:<28} {self.homeScore:>3d}")
