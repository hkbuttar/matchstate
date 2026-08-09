import Image from "next/image";
import { api } from "@/lib/api";

interface ClassMetric {
  brier: number;
  ece: number;
}
interface CalibrationRow {
  model: string;
  classes: Record<string, { raw: ClassMetric; platt: ClassMetric; isotonic: ClassMetric }>;
  best_method_by_mean_ece: string;
}

const MODEL_LABEL: Record<string, string> = {
  static: "Static Dixon-Coles",
  bayesian: "Hierarchical Bayesian",
  gbm: "Gradient Boosting",
};
const CLASS_LABEL: Record<string, string> = { home_win: "Home win", draw: "Draw", away_win: "Away win" };
const CLASS_ORDER = ["home_win", "draw", "away_win"];

export default async function CalibrationPage() {
  const results = await api.result<CalibrationRow[]>("calibration");

  return (
    <div className="space-y-10">
      <div>
        <h1 className="text-xl font-semibold tracking-tight" style={{ color: "var(--text-primary)" }}>
          Probability Calibration
        </h1>
        <p className="mt-1 max-w-3xl text-sm" style={{ color: "var(--text-secondary)" }}>
          Platt scaling and isotonic regression, fit on a 43-match calibration split disjoint from both model
          fitting and the 95-match test set, reported per outcome class since draws are the class most likely to be
          miscalibrated. Isotonic regression clearly helps <strong>home win</strong> calibration, but makes{" "}
          <strong>draw</strong> and <strong>away win</strong> calibration worse across every model -- traced to only
          9 draw-outcome matches in the calibration split. See <code>calibration/README.md</code> for the full
          finding.
        </p>
      </div>

      {results.map((r) => (
        <section key={r.model}>
          <h2 className="mb-3 font-medium" style={{ color: "var(--text-primary)" }}>
            {MODEL_LABEL[r.model] ?? r.model}
          </h2>
          <div className="overflow-hidden rounded-lg border" style={{ borderColor: "var(--border)" }}>
            <Image
              src={`/calibration/${r.model}_reliability.png`}
              alt={`Reliability diagram for ${MODEL_LABEL[r.model] ?? r.model}`}
              width={1600}
              height={533}
              className="w-full"
            />
          </div>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left" style={{ color: "var(--text-muted)" }}>
                  <th className="py-1 pr-4 font-medium">Class</th>
                  <th className="py-1 pr-4 font-medium">Raw ECE</th>
                  <th className="py-1 pr-4 font-medium">Platt ECE</th>
                  <th className="py-1 pr-4 font-medium">Isotonic ECE</th>
                </tr>
              </thead>
              <tbody>
                {CLASS_ORDER.map((cls) => {
                  const v = r.classes[cls];
                  const bestIsIso = v.isotonic.ece < v.raw.ece;
                  return (
                    <tr key={cls} className="border-t" style={{ borderColor: "var(--border)" }}>
                      <td className="py-1.5 pr-4" style={{ color: "var(--text-primary)" }}>
                        {CLASS_LABEL[cls]}
                      </td>
                      <td className="py-1.5 pr-4 tabular-nums" style={{ color: "var(--text-secondary)" }}>
                        {v.raw.ece.toFixed(4)}
                      </td>
                      <td className="py-1.5 pr-4 tabular-nums" style={{ color: "var(--text-secondary)" }}>
                        {v.platt.ece.toFixed(4)}
                      </td>
                      <td
                        className="py-1.5 pr-4 tabular-nums font-medium"
                        style={{ color: bestIsIso ? "var(--status-good)" : "var(--status-critical)" }}
                      >
                        {v.isotonic.ece.toFixed(4)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      ))}
    </div>
  );
}
