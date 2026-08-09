import { api } from "@/lib/api";
import LineChart, { type Annotation, type LineSeries } from "@/components/LineChart";

const EVENT_COLOR: Record<string, string> = {
  goal: "var(--status-good)",
  own_goal: "var(--status-good)",
  red_card: "var(--status-critical)",
  substitution: "var(--text-muted)",
};

export default async function MatchDetailPage(props: PageProps<"/matches/[id]">) {
  const { id } = await props.params;
  const matchId = Number(id);

  const [detail, trajectory] = await Promise.all([api.match(matchId), api.trajectory(matchId)]);

  const series: LineSeries[] = [
    {
      key: "static",
      label: "Static Dixon-Coles",
      color: "var(--series-1)",
      data: trajectory.points.map((p) => ({ x: p.minute, y: p.static_home_win })),
    },
    {
      key: "bayesian",
      label: "Hierarchical Bayesian",
      color: "var(--series-3)",
      data: trajectory.points.map((p) => ({ x: p.minute, y: p.bayesian_home_win })),
    },
    {
      key: "gbm",
      label: "Gradient Boosting",
      color: "var(--series-2)",
      data: trajectory.points.map((p) => ({ x: p.minute, y: p.gbm_home_win })),
    },
  ];

  // only goals and red cards annotate the chart -- substitutions are frequent
  // enough to clutter a 90+ minute timeline without adding much signal here
  const annotations: Annotation[] = detail.events
    .filter((e) => e.kind === "goal" || e.kind === "own_goal" || e.kind === "red_card")
    .map((e) => ({ x: e.minute, label: e.description, color: EVENT_COLOR[e.kind] }));

  return (
    <div className="space-y-8">
      <div>
        <div className="text-sm" style={{ color: "var(--text-muted)" }}>
          {detail.date}
        </div>
        <h1 className="text-2xl font-semibold tracking-tight" style={{ color: "var(--text-primary)" }}>
          {detail.home_team} {detail.final_home_goals}-{detail.final_away_goals} {detail.away_team}
        </h1>
      </div>

      <section>
        <h2 className="mb-1 font-medium" style={{ color: "var(--text-primary)" }}>
          P(home win) by minute, all three models
        </h2>
        <p className="mb-3 text-sm" style={{ color: "var(--text-secondary)" }}>
          Red lines mark goals and red cards. Watch how the static/Bayesian models barely react to red cards (they
          only see score and time) while gradient boosting jumps sharply -- see{" "}
          <code>models/README.md</code> for the full finding.
        </p>
        <LineChart series={series} yDomain={[0, 1]} xLabel="Minute" yLabel="P(home win)" annotations={annotations} />
      </section>

      <section>
        <h2 className="mb-3 font-medium" style={{ color: "var(--text-primary)" }}>
          Match events
        </h2>
        <ul className="space-y-1.5 text-sm">
          {detail.events.map((e, i) => (
            <li key={i} className="flex gap-3">
              <span className="w-10 shrink-0 tabular-nums" style={{ color: "var(--text-muted)" }}>
                {e.minute}&apos;
              </span>
              <span
                className="inline-block h-2 w-2 shrink-0 translate-y-1.5 rounded-full"
                style={{ background: EVENT_COLOR[e.kind] }}
              />
              <span style={{ color: "var(--text-secondary)" }}>{e.description}</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
