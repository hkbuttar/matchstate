const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export interface MatchSummary {
  match_id: number;
  date: string;
  home_team: string;
  away_team: string;
  final_home_goals: number;
  final_away_goals: number;
  final_result: string;
}

export interface MatchEvent {
  minute: number;
  kind: "goal" | "own_goal" | "red_card" | "substitution";
  team: string;
  description: string;
}

export interface MatchDetail extends MatchSummary {
  home_formation: number | null;
  away_formation: number | null;
  events: MatchEvent[];
}

export interface TrajectoryPoint {
  minute: number;
  home_goals: number;
  away_goals: number;
  static_home_win: number;
  static_draw: number;
  static_away_win: number;
  bayesian_home_win: number;
  bayesian_draw: number;
  bayesian_away_win: number;
  gbm_home_win: number;
  gbm_draw: number;
  gbm_away_win: number;
}

export interface MatchTrajectory {
  match_id: number;
  home_team: string;
  away_team: string;
  points: TrajectoryPoint[];
}

export interface BigMoment {
  minute: number;
  gbm_swing: number;
  static_swing: number;
  gbm_home_win: number;
  static_home_win: number;
  events: string;
}

export interface SeasonSummary {
  season: string;
  n_matches: number;
  home_adv: number;
  rho: number;
}

export interface SeasonDetail {
  season: string;
  n_matches: number;
  home_adv: number;
  rho: number;
  log_likelihood: number;
  attack: Record<string, number>;
  defense: Record<string, number>;
}

export interface BayesianTrajectoryRow {
  team: string;
  period: number;
  attack: number;
  defense: number;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`GET ${path} failed: ${res.status} ${body}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  matches: () => getJSON<MatchSummary[]>("/matches"),
  match: (id: number) => getJSON<MatchDetail>(`/matches/${id}`),
  trajectory: (id: number) => getJSON<MatchTrajectory>(`/matches/${id}/trajectory`),
  bigMoments: (id: number) => getJSON<BigMoment[]>(`/matches/${id}/big-moments`),
  seasons: () => getJSON<SeasonSummary[]>("/seasons"),
  season: (season: string) => getJSON<SeasonDetail>(`/seasons/${encodeURIComponent(season)}`),
  bayesianTrajectory: () => getJSON<BayesianTrajectoryRow[]>("/seasons/2015-16/bayesian-trajectory"),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  result: <T = any>(name: string) => getJSON<T>(`/results/${name}`),
};
