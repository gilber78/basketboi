import pandas as pd


class NBATeam:
    def __init__(self, teamName, teamId):
        # immutable instance variables, these never change
        self.teamName = teamName
        self.teamId = teamId
        self.teamAbbreviation = None  # TODO how to pull this from json without loading it a bunch of times

        # instance variables that get updated as we play throught games, are resettable
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.points_for = 0
        self.points_against = 0
        self.streak = 0
        self.last10 = []

        # specific to home (resettable)
        self.home_games_played = 0
        self.home_wins = 0
        self.home_losses = 0
        self.home_points_for = 0
        self.home_points_against = 0
        self.home_streak = 0
        self.home_last10 = []

        # specific to away (resettable)
        self.away_games_played = 0
        self.away_wins = 0
        self.away_losses = 0
        self.away_points_for = 0
        self.away_points_against = 0
        self.away_streak = 0
        self.away_last10 = []
