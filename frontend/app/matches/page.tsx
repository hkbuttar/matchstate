import Link from "next/link";
import { api } from "@/lib/api";

export default async function MatchesPage() {
  const matches = await api.matches();
  const sorted = [...matches].sort((a, b) => a.date.localeCompare(b.date));

  return (
    <div>
      <h1 className="text-xl font-semibold tracking-tight" style={{ color: "var(--text-primary)" }}>
        Matches
      </h1>
      <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
        All 380 matches of the 2015/16 EPL season. Click a match for its live win-probability trajectory.
      </p>

      <div className="mt-6 overflow-x-auto rounded-lg border" style={{ borderColor: "var(--border)" }}>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left" style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}>
              <th className="px-4 py-2 font-medium" style={{ color: "var(--text-muted)" }}>
                Date
              </th>
              <th className="px-4 py-2 font-medium" style={{ color: "var(--text-muted)" }}>
                Home
              </th>
              <th className="px-4 py-2 text-center font-medium" style={{ color: "var(--text-muted)" }}>
                Score
              </th>
              <th className="px-4 py-2 font-medium" style={{ color: "var(--text-muted)" }}>
                Away
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((m) => (
              <tr key={m.match_id} className="border-b last:border-0" style={{ borderColor: "var(--border)" }}>
                <td className="px-4 py-2 tabular-nums" style={{ color: "var(--text-secondary)" }}>
                  {m.date}
                </td>
                <td className="px-4 py-2" style={{ color: "var(--text-primary)" }}>
                  {m.home_team}
                </td>
                <td className="px-4 py-2 text-center">
                  <Link
                    href={`/matches/${m.match_id}`}
                    className="tabular-nums font-medium hover:underline"
                    style={{ color: "var(--series-1)" }}
                  >
                    {m.final_home_goals}-{m.final_away_goals}
                  </Link>
                </td>
                <td className="px-4 py-2" style={{ color: "var(--text-primary)" }}>
                  {m.away_team}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
