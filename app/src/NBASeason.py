import pandas as pd
from NBAGame import NBAGame
from NBATeam import NBATeam


class NBASeason:
    def __init__(self, data: pd.DataFrame):
        self.startDate = data["gameDateTimeEst"].min().split(" ")[0]
        self.endDate = data["gameDateTimeEst"].max().split(" ")[0]
        self.gameList = [NBAGame(line) for _, line in data.iterrows()]
        self.teamList = self._generate_list_of_teams()

    def _generate_list_of_teams(self):
        teamList = []
        for game in self.gameList:
            if (game.homeTeam, game.homeTeamId) not in teamList:
                teamList.append((game.homeTeam, game.homeTeamId))
            if (game.awayTeam, game.awayTeamId) not in teamList:
                teamList.append((game.awayTeam, game.awayTeamId))
        return [NBATeam(*args) for args in teamList]

    def pretty_print(self):  # TODO expand NBASeason pretty print
        print(f"{self.startDate.split('-')[0]}-{self.endDate.split('-')[0]} NBA SEASON:")
        print(f"    Number of games: {len(self.gameList)}")
        print(f"    Number of teams: {len(self.teamList)}")


if __name__ == "__main__":
    season_list = [NBASeason(pd.read_csv(f"app\\data\\games\seasons\\{i}-{i+1}\\{i}-{i+1}_raw.csv")) for i in range(1946, 2026)]
    for season in season_list:
        season.pretty_print()
