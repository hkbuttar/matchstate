import BarChart, { type Bar } from "@/components/BarChart";
import { api } from "@/lib/api";

interface Metric {
  brier: number;
  logloss: number;
}
interface StepNineResult {
  prematch: Record<string, Metric>;
  ingame_overall: Record<string, Metric>;
}
interface CI {
  point: number;
  ci_lo: number;
  ci_hi: number;
}
interface StepTenResult {
  per_model_ci: Record<string, CI>;
  pairwise_diffs: Record<string, { point_diff: number; ci_lo: number; ci_hi: number; significant_at_95: boolean }>;
}

const MODEL_ORDER = ["market", "static", "bayesian", "gbm"];
const MODEL_LABEL: Record<string, string> = {
  market: "Market (+ naive adj.)",
  static: "Static Dixon-Coles",
  bayesian: "Hierarchical Bayesian",
  gbm: "Gradient Boosting",
};
const MODEL_COLOR: Record<string, string> = {
  market: "var(--series-4)",
  static: "var(--series-1)",
  bayesian: "var(--series-3)",
  gbm: "var(--series-2)",
};

export default async function MarketPage() {
  const [stepNine, stepTen] = await Promise.all([
    api.result<StepNineResult>("market-comparison"),
    api.result<StepTenResult>("ingame-bootstrap"),
  ]);

  const prematchBars: Bar[] = MODEL_ORDER.map((m) => ({
    key: m,
    label: MODEL_LABEL[m],
    value: stepNine.prematch[m].brier,
    color: MODEL_COLOR[m],
  }));

  const ingameBars: Bar[] = MODEL_ORDER.map((m) => ({
    key: m,
    label: MODEL_LABEL[m],
    value: stepTen.per_model_ci[m].point,
    color: MODEL_COLOR[m],
    ciLo: stepTen.per_model_ci[m].ci_lo,
    ciHi: stepTen.per_model_ci[m].ci_hi,
  }));

  return (
    <div className="space-y-10">
      <div>
        <h1 className="text-xl font-semibold tracking-tight" style={{ color: "var(--text-primary)" }}>
          Betting-Market Benchmark
        </h1>
        <p className="mt-1 max-w-3xl text-sm" style={{ color: "var(--text-secondary)" }}>
          Every model against de-vigged Pinnacle closing odds, on the same 95 held-out test matches. The market
          wins pre-match, as expected -- but the in-game gap narrows substantially, and per the bootstrap (Step
          10), is not statistically significant. See <code>market/README.md</code> and{" "}
          <code>backtest/README.md</code>.
        </p>
      </div>

      <section>
        <h2 className="mb-3 font-medium" style={{ color: "var(--text-primary)" }}>
          Pre-match Brier score
        </h2>
        <BarChart bars={prematchBars} yLabel="Brier score (lower is better)" />
      </section>

      <section>
        <h2 className="mb-1 font-medium" style={{ color: "var(--text-primary)" }}>
          In-game Brier score, with 95% bootstrap confidence intervals
        </h2>
        <p className="mb-3 text-sm" style={{ color: "var(--text-secondary)" }}>
          Vertical bars show the match-block bootstrap 95% CI. Every interval overlaps every other --
          none of these differences are statistically significant except GBM vs. Bayesian and market vs. GBM (see
          table below).
        </p>
        <BarChart bars={ingameBars} yLabel="Brier score (lower is better)" />
      </section>

      <section>
        <h2 className="mb-3 font-medium" style={{ color: "var(--text-primary)" }}>
          Pairwise significance (match-block bootstrap, n=2000)
        </h2>
        <div className="overflow-x-auto rounded-lg border" style={{ borderColor: "var(--border)" }}>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left" style={{ background: "var(--surface-1)", color: "var(--text-muted)" }}>
                <th className="px-4 py-2 font-medium">Comparison</th>
                <th className="px-4 py-2 font-medium">Difference</th>
                <th className="px-4 py-2 font-medium">95% CI</th>
                <th className="px-4 py-2 font-medium">Result</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(stepTen.pairwise_diffs).map(([key, d]) => (
                <tr key={key} className="border-t" style={{ borderColor: "var(--border)" }}>
                  <td className="px-4 py-2" style={{ color: "var(--text-primary)" }}>
                    {key.replace("_minus_", " vs. ")}
                  </td>
                  <td className="px-4 py-2 tabular-nums" style={{ color: "var(--text-secondary)" }}>
                    {d.point_diff >= 0 ? "+" : ""}
                    {d.point_diff.toFixed(4)}
                  </td>
                  <td className="px-4 py-2 tabular-nums" style={{ color: "var(--text-secondary)" }}>
                    [{d.ci_lo.toFixed(4)}, {d.ci_hi.toFixed(4)}]
                  </td>
                  <td
                    className="px-4 py-2 font-medium"
                    style={{ color: d.significant_at_95 ? "var(--status-critical)" : "var(--text-muted)" }}
                  >
                    {d.significant_at_95 ? "Significant" : "Not significant"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
