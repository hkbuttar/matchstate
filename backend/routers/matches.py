import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.schemas import BigMoment, MatchDetail, MatchEvent, MatchSummary, MatchTrajectory, TrajectoryPoint
from backend.state import ensure_ready, state
from data.team_names import to_football_data
from models.gbm import predict_proba_dicts

router = APIRouter(prefix="/matches", tags=["matches"])


@router.get("", response_model=list[MatchSummary])
def list_matches():
    ensure_ready()
    df = state.features_df.drop_duplicates("match_id").sort_values("match_id")
    out = []
    for r in df.itertuples():
        meta = state.match_meta.get(r.match_id, {})
        out.append(
            MatchSummary(
                match_id=r.match_id,
                date=meta.get("match_date", ""),
                home_team=r.home_team,
                away_team=r.away_team,
                final_home_goals=r.final_home_goals,
                final_away_goals=r.final_away_goals,
                final_result=r.final_result,
            )
        )
    return out


def _match_row(match_id: int):
    df = state.features_df[state.features_df["match_id"] == match_id]
    if df.empty:
        raise HTTPException(404, f"match_id {match_id} not found")
    return df


def _match_events(match_id: int) -> dict:
    parsed = state.match_events.get(str(match_id))
    if parsed is None:
        raise HTTPException(404, f"match_id {match_id} not found")
    return parsed


@router.get("/{match_id}", response_model=MatchDetail)
def match_detail(match_id: int):
    ensure_ready()
    df = _match_row(match_id)
    first = df.iloc[0]
    meta = state.match_meta.get(match_id, {})
    parsed = _match_events(match_id)

    events: list[MatchEvent] = []
    for g in parsed["goals"]:
        events.append(MatchEvent(minute=g["minute"], kind="goal", team=g["team"], description=f"Goal, {g['team']}"))
    for r in parsed["red_cards"]:
        events.append(MatchEvent(
            minute=r["minute"], kind="red_card", team=r["team"],
            description=f"{r['card_type']}, {r['player_name']} ({r['team']})",
        ))
    for s in parsed["substitutions"]:
        events.append(MatchEvent(
            minute=s["minute"], kind="substitution", team=s["team"],
            description=f"{s['player_off_name']} -> {s['player_on_name']} ({s['team']})",
        ))
    events.sort(key=lambda e: e.minute)

    return MatchDetail(
        match_id=match_id,
        date=meta.get("match_date", ""),
        home_team=first.home_team,
        away_team=first.away_team,
        final_home_goals=int(first.final_home_goals),
        final_away_goals=int(first.final_away_goals),
        final_result=first.final_result,
        home_formation=int(first.home_formation) if pd.notna(first.home_formation) else None,
        away_formation=int(first.away_formation) if pd.notna(first.away_formation) else None,
        events=events,
    )


@router.get("/{match_id}/trajectory", response_model=MatchTrajectory)
def match_trajectory(match_id: int):
    ensure_ready()
    df = _match_row(match_id).sort_values("minute").reset_index(drop=True)
    home, away = df["home_team"].iloc[0], df["away_team"].iloc[0]
    home_fd, away_fd = to_football_data(home), to_football_data(away)

    gbm_probs = predict_proba_dicts(state.gbm_model, df)
    points = []
    for row, gbm_p in zip(df.itertuples(), gbm_probs):
        static_p = state.static_model.in_game_probabilities(home_fd, away_fd, row.minute, row.home_goals, row.away_goals)
        bayes_p = state.bayesian_model.in_game_probabilities(home_fd, away_fd, row.minute, row.home_goals, row.away_goals)
        points.append(
            TrajectoryPoint(
                minute=row.minute,
                home_goals=row.home_goals,
                away_goals=row.away_goals,
                static_home_win=static_p["home_win"], static_draw=static_p["draw"], static_away_win=static_p["away_win"],
                bayesian_home_win=bayes_p["home_win"], bayesian_draw=bayes_p["draw"], bayesian_away_win=bayes_p["away_win"],
                gbm_home_win=gbm_p["home_win"], gbm_draw=gbm_p["draw"], gbm_away_win=gbm_p["away_win"],
            )
        )
    return MatchTrajectory(match_id=match_id, home_team=home, away_team=away, points=points)


@router.get("/{match_id}/big-moments", response_model=list[BigMoment])
def match_big_moments(match_id: int, n: int = 5):
    traj = match_trajectory(match_id)
    home = traj.home_team
    parsed = _match_events(match_id)
    goals = parsed["goals"]

    def annotate(minute: int) -> str:
        tags = []
        for g in goals:
            if g["minute"] == minute:
                tags.append(f"GOAL ({'home' if g['team'] == home else 'away'}, {g['team']})")
        for r in parsed["red_cards"]:
            if r["minute"] == minute:
                tags.append(f"RED CARD ({'home' if r['team'] == home else 'away'}, {r['player_name']})")
        for s in parsed["substitutions"]:
            if s["minute"] == minute:
                tags.append(f"SUB ({'home' if s['team'] == home else 'away'}: {s['player_off_name']} -> {s['player_on_name']})")
        return "; ".join(tags)

    gbm_arr = np.array([[p.gbm_home_win, p.gbm_draw, p.gbm_away_win] for p in traj.points])
    static_arr = np.array([[p.static_home_win, p.static_draw, p.static_away_win] for p in traj.points])
    gbm_swing = np.concatenate([[0.0], np.abs(np.diff(gbm_arr, axis=0)).sum(axis=1) * 0.5])
    static_swing = np.concatenate([[0.0], np.abs(np.diff(static_arr, axis=0)).sum(axis=1) * 0.5])

    top_idx = np.argsort(-gbm_swing)[:n]
    top_idx = sorted(top_idx.tolist())

    return [
        BigMoment(
            minute=traj.points[i].minute,
            gbm_swing=float(gbm_swing[i]),
            static_swing=float(static_swing[i]),
            gbm_home_win=traj.points[i].gbm_home_win,
            static_home_win=traj.points[i].static_home_win,
            events=annotate(traj.points[i].minute),
        )
        for i in top_idx
    ]
