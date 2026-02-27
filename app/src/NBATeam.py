import json
from NBAGame import NBAGame
from functions import get_list_wins_and_losses

with open("app\\data\\team_data.json", "r") as file:
    team_data = json.load(file)


class NBATeam:
    def __init__(self, teamName):
        # immutable instance variables, these never change
        self.teamName = teamName
        self.teamId = team_data[teamName]["id"]
        self.teamAbbreviation = team_data[teamName]["abv"]
        self.reset_statistics()

    def reset_statistics(self):
        # instance variables that get updated as we play throught games
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.points_for = 0
        self.points_against = 0
        self.streak = 0
        self.last10 = []

        # specific to home games
        self.home_games_played = 0
        self.home_wins = 0
        self.home_losses = 0
        self.home_points_for = 0
        self.home_points_against = 0
        self.home_streak = 0
        self.home_last10 = []

        # specific to away games
        self.away_games_played = 0
        self.away_wins = 0
        self.away_losses = 0
        self.away_points_for = 0
        self.away_points_against = 0
        self.away_streak = 0
        self.away_last10 = []

        # specific to wins
        self.win_points_for = 0
        self.win_points_against = 0

        # specific to losses
        self.loss_points_for = 0
        self.loss_points_against = 0

        # specific to home wins
        self.homewin_points_for = 0
        self.homewin_points_against = 0

        # specific to home losses
        self.homeloss_points_for = 0
        self.homeloss_points_against = 0

        # specific to away wins
        self.awaywin_points_for = 0
        self.awaywin_points_against = 0

        # specific to away losses
        self.awayloss_points_for = 0
        self.awayloss_points_against = 0

    def update_game(self, game: NBAGame):
        # determine parameters of the game to tell what stats to update how
        switch = {
            "home": True if self.teamId == game.homeTeamId else False,
            "win": True if self.teamId == game.winner else False,
        }

        # apply updates - I wish there aas a better way to document this...
        self.games_played += 1
        self.last10.append(switch["win"])
        if len(self.last10) > 10:
            self.last10.pop(0)
        if switch["home"]:
            self.home_games_played += 1
            self.points_for += game.homeScore
            self.home_points_for += game.homeScore
            self.points_against += game.awayScore
            self.home_points_against += game.awayScore
            self.home_last10.append(switch["win"])
            if len(self.home_last10) > 10:
                self.home_last10.pop(0)
            if switch["win"]:
                self.wins += 1
                self.home_wins += 1
                if self.streak < 0:
                    self.streak = 0
                self.streak += 1
                if self.home_streak < 0:
                    self.home_streak = 0
                self.home_streak += 1
                self.win_points_for += game.homeScore
                self.win_points_against += game.awayScore
                self.homewin_points_for += game.homeScore
                self.homewin_points_against += game.awayScore
            else:
                self.losses += 1
                self.home_losses += 1
                if self.streak > 0:
                    self.streak = 0
                self.streak -= 1
                if self.home_streak > 0:
                    self.home_streak = 0
                self.home_streak -= 1
                self.loss_points_for += game.homeScore
                self.loss_points_against += game.awayScore
                self.homeloss_points_for += game.homeScore
                self.homeloss_points_against += game.awayScore
        else:
            self.away_games_played += 1
            self.points_for += game.awayScore
            self.away_points_for += game.awayScore
            self.points_against += game.homeScore
            self.away_points_against += game.homeScore
            self.away_last10.append(switch["win"])
            if len(self.away_last10) > 10:
                self.away_last10.pop(0)
            if switch["win"]:
                self.wins += 1
                self.away_wins += 1
                if self.streak < 0:
                    self.streak = 0
                self.streak += 1
                if self.away_streak < 0:
                    self.away_streak = 0
                self.away_streak += 1
                self.win_points_for += game.awayScore
                self.win_points_against += game.homeScore
                self.awaywin_points_for += game.awayScore
                self.awaywin_points_against += game.homeScore
            else:
                self.losses += 1
                self.away_losses += 1
                if self.streak > 0:
                    self.streak = 0
                self.streak -= 1
                if self.away_streak > 0:
                    self.away_streak = 0
                self.away_streak -= 1
                self.loss_points_for += game.awayScore
                self.loss_points_against += game.homeScore
                self.awayloss_points_for += game.awayScore
                self.awayloss_points_against += game.homeScore

    def pretty_print(self, header=True):
        if header:
            print("\033[4m                            W    L     PCT    HOME    AWAY    PFPG    PAPG    DIFF   STRK    L10   \033[0m")
        last10 = get_list_wins_and_losses(self.last10)
        print(
            f"{self.teamName:<26} "
            + f"{self.wins:>2d}   "
            + f"{self.losses:>2d}   "
            + f"{0 if self.games_played == 0 else self.wins/self.games_played:>.3f}   "
            + f"{self.home_wins:>2d}-{self.home_losses:<2d}   "
            + f"{self.away_wins:>2d}-{self.away_losses:<2d}   "
            + f"{0 if self.games_played == 0 else self.points_for/self.games_played:>5.1f}   "
            + f"{0 if self.games_played == 0 else self.points_against/self.games_played:>5.1f}   "
            + f"{0 if self.away_games_played == 0 else (self.points_for - self.points_against)/self.games_played:>+5.1f}    "
            + f"{' ' if abs(self.streak) < 10 else ''}{'W' if self.streak >= 0 else 'L'}{abs(self.streak):<2d}  "
            + f"{' ' if abs(last10[0]) >= 10 else ''}{last10[0]:>2d}-{last10[1]:<2d}"
        )
