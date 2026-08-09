import { type LineSeries } from "@/components/LineChart";
import { HomeAdvantageChart, StrengthEvolutionChart } from "@/components/SeasonCharts";
import { api } from "@/lib/api";

// Bayesian trajectory rows are keyed by football-data.co.uk team names
// (see baseline/data.py), not StatsBomb's -- "Leicester", not "Leicester City".
const HIGHLIGHT_TEAMS = ["Leicester", "Aston Villa", "Man City"];
const HIGHLIGHT_COLOR: Record<string, string> = {
  Leicester: "var(--series-1)",
  "Aston Villa": "var(--series-2)",
  "Man City": "var(--series-3)",
};

export default async function SeasonsPage() {
  const [seasons, trajectory] = await Promise.all([api.seasons(), api.bayesianTrajectory()]);

  const orderedSeasons = [...seasons].sort((a, b) => a.season.localeCompare(b.season));
  const seasonIndex = new Map(orderedSeasons.map((s, i) => [s.season, i]));

  const homeAdvSeries: LineSeries[] = [
    {
      key: "home_adv",
      label: "Home advantage",
      color: "var(--series-1)",
      data: orderedSeasons.map((s) => ({ x: seasonIndex.get(s.season)!, y: s.home_adv })),
    },
  ];

  const teams = Array.from(new Set(trajectory.map((r) => r.team)));
  const strengthSeries: LineSeries[] = teams.map((team) => {
    const rows = trajectory.filter((r) => r.team === team).sort((a, b) => a.period - b.period);
    const highlighted = HIGHLIGHT_TEAMS.includes(team);
    return {
      key: team,
      label: team,
      color: highlighted ? HIGHLIGHT_COLOR[team] : "var(--text-muted)",
      muted: !highlighted,
      data: rows.map((r) => ({ x: r.period, y: r.attack - r.defense })),
    };
  });
  // draw highlighted series last so they render on top of the muted ones
  strengthSeries.sort((a, b) => Number(a.muted ?? false) - Number(b.muted ?? false));

  return (
    <div className="space-y-10">
      <div>
        <h1 className="text-xl font-semibold tracking-tight" style={{ color: "var(--text-primary)" }}>
          Team Strength History
        </h1>
        <p className="mt-1 max-w-3xl text-sm" style={{ color: "var(--text-secondary)" }}>
          Home advantage fitted independently for all 33 EPL seasons, and the hierarchical Bayesian model&apos;s
          within-season strength evolution for 2015/16 (Leicester&apos;s title-winning season).
        </p>
      </div>

      <section>
        <h2 className="mb-1 font-medium" style={{ color: "var(--text-primary)" }}>
          Home advantage by season
        </h2>
        <p className="mb-3 text-sm" style={{ color: "var(--text-secondary)" }}>
          Positive in every single season -- and notably lower in 2020/21 (index {seasonIndex.get("2020/21")}), the
          empty-stadiums COVID season, recovered from goals data alone.
        </p>
        <HomeAdvantageChart homeAdvSeries={homeAdvSeries} seasonLabels={orderedSeasons.map((s) => s.season)} />
      </section>

      <section>
        <h2 className="mb-1 font-medium" style={{ color: "var(--text-primary)" }}>
          Within-season strength evolution, 2015/16
        </h2>
        <p className="mb-3 text-sm" style={{ color: "var(--text-secondary)" }}>
          All 20 clubs plotted; Leicester, Aston Villa, and Manchester City highlighted. Leicester&apos;s strength
          rises every single period -- driven mainly by defense, matching the real history of that title win.
        </p>
        <StrengthEvolutionChart strengthSeries={strengthSeries} />
      </section>
    </div>
  );
}
