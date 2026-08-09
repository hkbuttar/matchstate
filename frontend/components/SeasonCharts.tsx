"use client";

import LineChart, { type LineSeries } from "@/components/LineChart";

// Client wrapper: the xTickFormat callbacks LineChart needs can't cross the
// server/client boundary as props (React can't serialize functions from a
// Server Component into a "use client" child) -- this component receives
// only plain, serializable data and builds those closures locally instead.

export function HomeAdvantageChart({ homeAdvSeries, seasonLabels }: { homeAdvSeries: LineSeries[]; seasonLabels: string[] }) {
  return (
    <LineChart
      series={homeAdvSeries}
      yLabel="Fitted home advantage"
      xTickFormat={(v) => seasonLabels[Math.round(v)] ?? ""}
      xDomain={[0, seasonLabels.length - 1]}
    />
  );
}

export function StrengthEvolutionChart({ strengthSeries }: { strengthSeries: LineSeries[] }) {
  return (
    <LineChart
      series={strengthSeries}
      yLabel="Overall strength (attack - defense)"
      xLabel="Period (0 = start of season, 7 = end)"
      xTickFormat={(v) => String(Math.round(v))}
    />
  );
}
