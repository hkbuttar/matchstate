import Link from "next/link";
import { api } from "@/lib/api";

interface GbmComparisonRow {
  model: string;
  brier: number;
  logloss: number;
}

function StatTile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border p-5" style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}>
      <div className="text-sm" style={{ color: "var(--text-muted)" }}>
        {label}
      </div>
      <div className="mt-1 text-3xl font-semibold" style={{ color: "var(--text-primary)" }}>
        {value}
      </div>
      {sub && (
        <div className="mt-1 text-xs" style={{ color: "var(--text-secondary)" }}>
          {sub}
        </div>
      )}
    </div>
  );
}

export default async function HomePage() {
  const [matches, seasons, gbmComparison] = await Promise.all([
    api.matches(),
    api.seasons(),
    api.result<GbmComparisonRow[]>("gbm-comparison"),
  ]);

  const bestBrier = Math.min(...gbmComparison.map((r) => r.brier));

  return (
    <div className="space-y-10">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight" style={{ color: "var(--text-primary)" }}>
          MatchState
        </h1>
        <p className="mt-2 max-w-2xl text-sm" style={{ color: "var(--text-secondary)" }}>
          Live win probability model for EPL matches. Dixon-Coles Poisson baseline, hierarchical Bayesian team
          strength, possession-value momentum, and gradient boosting, benchmarked against real betting-market odds
          with bootstrap-validated calibration.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatTile label="Matches with event data" value={String(matches.length)} sub="2015/16 season, StatsBomb" />
        <StatTile label="EPL seasons (results)" value={String(seasons.length)} sub="1993/94-2025/26" />
        <StatTile label="Best in-game Brier" value={bestBrier.toFixed(4)} sub="95 held-out test matches" />
        <StatTile label="Models compared" value="4" sub="Dixon-Coles, Bayesian, GBM, market" />
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Link
          href="/matches"
          className="rounded-lg border p-5 transition hover:opacity-80"
          style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}
        >
          <div className="font-medium" style={{ color: "var(--text-primary)" }}>
            Match explorer
          </div>
          <div className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
            Live win-probability trajectories per minute, with big-moment annotations (goals, red cards,
            substitutions) for any of the 380 matches.
          </div>
        </Link>
        <Link
          href="/calibration"
          className="rounded-lg border p-5 transition hover:opacity-80"
          style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}
        >
          <div className="font-medium" style={{ color: "var(--text-primary)" }}>
            Calibration
          </div>
          <div className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
            Reliability diagrams per outcome class, raw vs. isotonic-calibrated.
          </div>
        </Link>
        <Link
          href="/market"
          className="rounded-lg border p-5 transition hover:opacity-80"
          style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}
        >
          <div className="font-medium" style={{ color: "var(--text-primary)" }}>
            Betting-market benchmark
          </div>
          <div className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
            Every model vs. de-vigged Pinnacle closing odds, pre-match and in-game.
          </div>
        </Link>
        <Link
          href="/seasons"
          className="rounded-lg border p-5 transition hover:opacity-80"
          style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}
        >
          <div className="font-medium" style={{ color: "var(--text-primary)" }}>
            Team strength history
          </div>
          <div className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
            Home advantage and fitted team strength across all 33 EPL seasons.
          </div>
        </Link>
      </div>
    </div>
  );
}
