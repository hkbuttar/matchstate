from pydantic import BaseModel


class MatchSummary(BaseModel):
    match_id: int
    date: str
    home_team: str
    away_team: str
    final_home_goals: int
    final_away_goals: int
    final_result: str


class MatchEvent(BaseModel):
    minute: int
    kind: str  # "goal" | "own_goal" | "red_card" | "substitution"
    team: str
    description: str


class MatchDetail(BaseModel):
    match_id: int
    date: str
    home_team: str
    away_team: str
    final_home_goals: int
    final_away_goals: int
    final_result: str
    home_formation: int | None
    away_formation: int | None
    events: list[MatchEvent]


class TrajectoryPoint(BaseModel):
    minute: int
    home_goals: int
    away_goals: int
    static_home_win: float
    static_draw: float
    static_away_win: float
    bayesian_home_win: float
    bayesian_draw: float
    bayesian_away_win: float
    gbm_home_win: float
    gbm_draw: float
    gbm_away_win: float


class MatchTrajectory(BaseModel):
    match_id: int
    home_team: str
    away_team: str
    points: list[TrajectoryPoint]


class BigMoment(BaseModel):
    minute: int
    gbm_swing: float
    static_swing: float
    gbm_home_win: float
    static_home_win: float
    events: str
