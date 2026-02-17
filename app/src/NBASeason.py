import pandas as pd
from NBAGame import NBAGame
from NBATeam import NBATeam
from functions import get_day_from_full_time, increment_day, get_list_wins_and_losses


def create_game_data_series(awayTeam: NBATeam, homeTeam: NBATeam, game: NBAGame = None):
    # TODO expand this to percentages, ratios, etc that get used in modeling to save space
    HOME_last10 = get_list_wins_and_losses(homeTeam.last10)
    HOME_home_last10 = get_list_wins_and_losses(homeTeam.home_last10)
    # HOME_away_last10 = get_list_wins_and_losses(homeTeam.away_last10)
    AWAY_last10 = get_list_wins_and_losses(awayTeam.last10)
    # AWAY_home_last10 = get_list_wins_and_losses(awayTeam.home_last10)
    AWAY_away_last10 = get_list_wins_and_losses(awayTeam.away_last10)
    series_dict = {
        "HOME_games_played": homeTeam.games_played,
        "HOME_wins": homeTeam.wins,
        "HOME_losses": homeTeam.losses,
        "HOME_points_for": homeTeam.points_for,
        "HOME_points_against": homeTeam.points_against,
        "HOME_streak": homeTeam.streak,
        "HOME_last10_w": HOME_last10[0],
        "HOME_last10_l": HOME_last10[1],
        "HOME_home_games_played": homeTeam.home_games_played,
        "HOME_home_wins": homeTeam.home_wins,
        "HOME_home_losses": homeTeam.home_losses,
        "HOME_home_points_for": homeTeam.home_points_for,
        "HOME_home_points_against": homeTeam.home_points_against,
        "HOME_home_streak": homeTeam.home_streak,
        "HOME_home_last10_w": HOME_home_last10[0],
        "HOME_home_last10_l": HOME_home_last10[1],
        # "HOME_away_games_played": homeTeam.away_games_played,
        # "HOME_away_wins": homeTeam.away_wins,
        # "HOME_away_losses": homeTeam.away_losses,
        # "HOME_away_points_for": homeTeam.away_points_for,
        # "HOME_away_points_against": homeTeam.away_points_against,
        # "HOME_away_streak": homeTeam.away_streak,
        # "HOME_away_last10_w": HOME_away_last10[0],
        # "HOME_away_last10_l": HOME_away_last10[1],
        "HOME_win_points_for": homeTeam.win_points_for,
        "HOME_win_points_against": homeTeam.win_points_against,
        "HOME_loss_points_for": homeTeam.loss_points_for,
        "HOME_loss_points_against": homeTeam.loss_points_against,
        "HOME_homewin_points_for": homeTeam.homewin_points_for,
        "HOME_homewin_points_against": homeTeam.homewin_points_against,
        "HOME_homeloss_points_for": homeTeam.homeloss_points_for,
        "HOME_homeloss_points_against": homeTeam.homeloss_points_against,
        # "HOME_awaywin_points_for": homeTeam.awaywin_points_for,
        # "HOME_awaywin_points_against": homeTeam.awaywin_points_against,
        # "HOME_awayloss_points_for": homeTeam.awayloss_points_for,
        # "HOME_awayloss_points_against": homeTeam.awayloss_points_against,
        "AWAY_games_played": awayTeam.games_played,
        "AWAY_wins": awayTeam.wins,
        "AWAY_losses": awayTeam.losses,
        "AWAY_points_for": awayTeam.points_for,
        "AWAY_points_against": awayTeam.points_against,
        "AWAY_streak": awayTeam.streak,
        "AWAY_last10_w": AWAY_last10[0],
        "AWAY_last10_l": AWAY_last10[1],
        # "AWAY_home_games_played": awayTeam.home_games_played,
        # "AWAY_home_wins": awayTeam.home_wins,
        # "AWAY_home_losses": awayTeam.home_losses,
        # "AWAY_home_points_for": awayTeam.home_points_for,
        # "AWAY_home_points_against": awayTeam.home_points_against,
        # "AWAY_home_streak": awayTeam.home_streak,
        # "AWAY_home_last10_w": AWAY_home_last10[0],
        # "AWAY_home_last10_l": AWAY_home_last10[1],
        "AWAY_away_games_played": awayTeam.away_games_played,
        "AWAY_away_wins": awayTeam.away_wins,
        "AWAY_away_losses": awayTeam.away_losses,
        "AWAY_away_points_for": awayTeam.away_points_for,
        "AWAY_away_points_against": awayTeam.away_points_against,
        "AWAY_away_streak": awayTeam.away_streak,
        "AWAY_away_last10_w": AWAY_away_last10[0],
        "AWAY_away_last10_l": AWAY_away_last10[1],
        "AWAY_win_points_for": awayTeam.win_points_for,
        "AWAY_win_points_against": awayTeam.win_points_against,
        "AWAY_loss_points_for": awayTeam.loss_points_for,
        "AWAY_loss_points_against": awayTeam.loss_points_against,
        # "AWAY_homewin_points_for": awayTeam.homewin_points_for,
        # "AWAY_homewinwin_points_against": awayTeam.homewin_points_against,
        # "AWAY_homeloss_points_for": awayTeam.homeloss_points_for,
        # "AWAY_homeloss_points_against": awayTeam.homeloss_points_against,
        "AWAY_awaywin_points_for": awayTeam.awaywin_points_for,
        "AWAY_awaywin_points_against": awayTeam.awaywin_points_against,
        "AWAY_awayloss_points_for": awayTeam.awayloss_points_for,
        "AWAY_awayloss_points_against": awayTeam.awayloss_points_against,
    }
    if game != None:
        game_dict = {
            "GAME_homeScore": game.homeScore,
            "GAME_awayScore": game.awayScore,
            "GAME_total": game.total,
            "GAME_spread": game.spread,
            "GAME_homeWin": 1 if game.winner == homeTeam.teamId else 0,
        }
    else:
        game_dict = {
            "GAME_homeScore": None,
            "GAME_awayScore": None,
            "GAME_total": None,
            "GAME_spread": None,
            "GAME_homeWin": None,
        }
    return pd.DataFrame([series_dict | game_dict])


class NBASeason:
    def __init__(self, data: pd.DataFrame):
        self.startDate = get_day_from_full_time(data["gameDateTimeEst"].min())
        self.endDate = get_day_from_full_time(data["gameDateTimeEst"].max())
        self.gameList = [NBAGame(line) for _, line in data.iterrows()]
        self.teamList = self.generate_list_of_teams()

    def generate_list_of_teams(self):
        teamList = []
        for game in self.gameList:
            if (game.homeTeam, game.homeTeamId) not in teamList:
                teamList.append((game.homeTeam, game.homeTeamId))
            if (game.awayTeam, game.awayTeamId) not in teamList:
                teamList.append((game.awayTeam, game.awayTeamId))
        return [NBATeam(*args) for args in teamList]

    def generate_and_save_full_season_data(self):
        current_day = self.startDate
        df = pd.DataFrame()
        while current_day <= self.endDate:
            # get list of games that occurred on this day, in order
            current_game_list = [game for game in self.gameList if game.gameDate == current_day]
            for game in current_game_list:
                awayTeam = next(team for team in self.teamList if team.teamId == game.awayTeamId)
                homeTeam = next(team for team in self.teamList if team.teamId == game.homeTeamId)
                df = pd.concat([df, create_game_data_series(awayTeam=awayTeam, homeTeam=homeTeam, game=game)], ignore_index=True)
                awayTeam.update_game(game)
                homeTeam.update_game(game)
            # to move to the next day in the sequence
            current_day = increment_day(current_day)
        return df

    def pretty_print(self):
        sorted_standings = sorted(
            self.teamList,
            key=lambda team: (0 if team.games_played == 0 else team.wins / team.games_played, team.points_for - team.points_against),
            reverse=True,
        )
        print(f"{self.startDate.split('-')[0]}-{self.endDate.split('-')[0]} NBA SEASON:")
        sorted_standings[0].pretty_print(header=True)
        for team in sorted_standings[1:]:
            team.pretty_print(header=False)
