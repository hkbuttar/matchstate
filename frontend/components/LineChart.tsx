"use client";

import { useMemo, useRef, useState } from "react";

export interface LineSeries {
  key: string;
  label: string;
  color: string;
  data: { x: number; y: number }[];
  muted?: boolean; // emphasis mode: render in de-emphasis gray, thinner
}

export interface Annotation {
  x: number;
  label: string;
  color?: string;
}

interface LineChartProps {
  series: LineSeries[];
  height?: number;
  xDomain?: [number, number];
  yDomain?: [number, number];
  xLabel?: string;
  yLabel?: string;
  yTickFormat?: (v: number) => string;
  xTickFormat?: (v: number) => string;
  annotations?: Annotation[];
  referenceLine?: boolean; // y = x diagonal, for reliability diagrams
  yTicks?: number[];
}

const MARGIN = { top: 16, right: 88, bottom: 36, left: 48 };

export default function LineChart({
  series,
  height = 340,
  xDomain,
  yDomain,
  xLabel,
  yLabel,
  yTickFormat = (v) => v.toFixed(2),
  xTickFormat = (v) => String(v),
  annotations = [],
  referenceLine = false,
  yTicks,
}: LineChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const width = 720; // SVG viewBox scales this proportionally to the container's actual rendered width
  const [hoverX, setHoverX] = useState<number | null>(null);

  const allX = series.flatMap((s) => s.data.map((d) => d.x));
  const allY = series.flatMap((s) => s.data.map((d) => d.y));
  const [xMin, xMax] = xDomain ?? [Math.min(...allX), Math.max(...allX)];
  const [yMin, yMax] = yDomain ?? [Math.min(0, ...allY), Math.max(...allY)];

  // The right margin must fit the single-series direct end-label when one
  // will be drawn (2+ series rely on the legend instead, see labelableKeys
  // below, and don't need margin reserved for a label at all). ~6.2px/char
  // is a conservative estimate for 11px system-ui sans; never clip instead
  // of guessing too small.
  const singleSeriesLabel = series.filter((s) => !s.muted).length === 1 ? series.find((s) => !s.muted)?.label : null;
  const rightMargin = singleSeriesLabel ? Math.max(MARGIN.right, 24 + singleSeriesLabel.length * 6.2) : MARGIN.right;

  const plotW = width - MARGIN.left - rightMargin;
  const plotH = height - MARGIN.top - MARGIN.bottom;

  const sx = (x: number) => MARGIN.left + ((x - xMin) / (xMax - xMin || 1)) * plotW;
  const sy = (y: number) => MARGIN.top + plotH - ((y - yMin) / (yMax - yMin || 1)) * plotH;

  const ticksY = yTicks ?? niceTicks(yMin, yMax, 5);
  const ticksX = niceTicks(xMin, xMax, 6);

  function handleMove(e: React.MouseEvent<SVGSVGElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * width;
    const x = xMin + ((px - MARGIN.left) / plotW) * (xMax - xMin);
    if (x < xMin || x > xMax) {
      setHoverX(null);
      return;
    }
    setHoverX(x);
  }

  // Direct end-labels are only useful when there's no legend to carry
  // identity instead (the single-series case). With 2+ series a legend is
  // always rendered (see below), and long labels risk colliding where lines
  // converge (all models near-certain by full time) or being clipped by the
  // right margin -- per the dataviz skill, the safe fallback for converging
  // series is the legend + tooltip, not stacked or clipped text.
  const nonMutedCount = series.filter((s) => !s.muted).length;
  const labelableKeys = useMemo(() => {
    if (nonMutedCount >= 2) return new Set<string>();
    return new Set(series.filter((s) => !s.muted).map((s) => s.key));
  }, [series, nonMutedCount]);

  const nearestX = useMemo(() => {
    if (hoverX === null || series.length === 0 || series[0].data.length === 0) return null;
    const xs = series[0].data.map((d) => d.x);
    let best = xs[0];
    let bestDist = Infinity;
    for (const x of xs) {
      const d = Math.abs(x - hoverX);
      if (d < bestDist) {
        bestDist = d;
        best = x;
      }
    }
    return best;
  }, [hoverX, series]);

  const nonMutedSeries = series.filter((s) => !s.muted);

  return (
    <div ref={containerRef} className="w-full">
      {nonMutedSeries.length >= 2 && (
        <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-xs">
          {nonMutedSeries.map((s) => (
            <div key={s.key} className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-3" style={{ background: s.color }} />
              <span style={{ color: "var(--text-secondary)" }}>{s.label}</span>
            </div>
          ))}
        </div>
      )}
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        role="img"
        onMouseMove={handleMove}
        onMouseLeave={() => setHoverX(null)}
        style={{ background: "var(--surface-1)" }}
      >
        {/* gridlines */}
        {ticksY.map((t) => (
          <g key={`gy-${t}`}>
            <line
              x1={MARGIN.left}
              x2={width - rightMargin}
              y1={sy(t)}
              y2={sy(t)}
              stroke="var(--gridline)"
              strokeWidth={1}
            />
            <text x={MARGIN.left - 8} y={sy(t)} textAnchor="end" dominantBaseline="middle" fontSize={11} fill="var(--text-muted)">
              {yTickFormat(t)}
            </text>
          </g>
        ))}
        {ticksX.map((t) => (
          <text
            key={`gx-${t}`}
            x={sx(t)}
            y={height - MARGIN.bottom + 18}
            textAnchor="middle"
            fontSize={11}
            fill="var(--text-muted)"
          >
            {xTickFormat(t)}
          </text>
        ))}

        {/* axis baseline */}
        <line
          x1={MARGIN.left}
          x2={width - rightMargin}
          y1={height - MARGIN.bottom}
          y2={height - MARGIN.bottom}
          stroke="var(--axis)"
          strokeWidth={1}
        />

        {referenceLine && (
          <line
            x1={sx(Math.max(xMin, yMin))}
            y1={sy(Math.max(xMin, yMin))}
            x2={sx(Math.min(xMax, yMax))}
            y2={sy(Math.min(xMax, yMax))}
            stroke="var(--axis)"
            strokeWidth={1.5}
          />
        )}

        {/* annotations: vertical event markers */}
        {annotations.map((a, i) => (
          <g key={`ann-${i}`}>
            <line
              x1={sx(a.x)}
              x2={sx(a.x)}
              y1={MARGIN.top}
              y2={height - MARGIN.bottom}
              stroke={a.color ?? "var(--status-critical)"}
              strokeWidth={1.5}
              strokeOpacity={0.55}
            />
          </g>
        ))}

        {/* series lines */}
        {series.map((s) => {
          const d = s.data.map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.x)} ${sy(p.y)}`).join(" ");
          const color = s.muted ? "var(--text-muted)" : s.color;
          const last = s.data[s.data.length - 1];
          return (
            <g key={s.key}>
              <path d={d} fill="none" stroke={color} strokeWidth={s.muted ? 1.5 : 2} strokeLinejoin="round" strokeLinecap="round" opacity={s.muted ? 0.35 : 1} />
              {last && !s.muted && (
                <>
                  <circle cx={sx(last.x)} cy={sy(last.y)} r={4} fill={color} stroke="var(--surface-1)" strokeWidth={2} />
                  {labelableKeys.has(s.key) && (
                    <text x={sx(last.x) + 8} y={sy(last.y)} dominantBaseline="middle" fontSize={11} fill="var(--text-secondary)">
                      {s.label}
                    </text>
                  )}
                </>
              )}
            </g>
          );
        })}

        {/* crosshair */}
        {nearestX !== null && (
          <line x1={sx(nearestX)} x2={sx(nearestX)} y1={MARGIN.top} y2={height - MARGIN.bottom} stroke="var(--axis)" strokeWidth={1} />
        )}

        {xLabel && (
          <text x={MARGIN.left + plotW / 2} y={height - 2} textAnchor="middle" fontSize={11} fill="var(--text-muted)">
            {xLabel}
          </text>
        )}
        {yLabel && (
          <text
            x={-(MARGIN.top + plotH / 2)}
            y={12}
            textAnchor="middle"
            fontSize={11}
            fill="var(--text-muted)"
            transform="rotate(-90)"
          >
            {yLabel}
          </text>
        )}
      </svg>

      {nearestX !== null && (
        <div className="mt-1 rounded border px-3 py-2 text-xs" style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}>
          <div className="mb-1 font-medium" style={{ color: "var(--text-primary)" }}>
            {xTickFormat(nearestX)}
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1">
            {/* emphasis mode: the tooltip reads out the highlighted series only --
                muted series are background context, not meant for a per-value readout */}
            {(series.some((s) => s.muted) ? series.filter((s) => !s.muted) : series).map((s) => {
              const point = s.data.reduce((best, p) => (Math.abs(p.x - nearestX) < Math.abs(best.x - nearestX) ? p : best), s.data[0]);
              return (
                <div key={s.key} className="flex items-center gap-1.5">
                  <span className="inline-block h-0.5 w-3" style={{ background: s.muted ? "var(--text-muted)" : s.color }} />
                  <span style={{ color: "var(--text-secondary)" }}>{s.label}</span>
                  <span className="font-semibold tabular-nums" style={{ color: "var(--text-primary)" }}>
                    {point ? point.y.toFixed(3) : "-"}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function niceTicks(min: number, max: number, count: number): number[] {
  if (min === max) return [min];
  const range = max - min;
  const step = niceNum(range / (count - 1), true);
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 1e-6; v += step) {
    ticks.push(Math.round(v * 1e6) / 1e6);
  }
  return ticks;
}

function niceNum(range: number, round: boolean): number {
  const exponent = Math.floor(Math.log10(range));
  const fraction = range / 10 ** exponent;
  let niceFraction: number;
  if (round) {
    if (fraction < 1.5) niceFraction = 1;
    else if (fraction < 3) niceFraction = 2;
    else if (fraction < 7) niceFraction = 5;
    else niceFraction = 10;
  } else {
    if (fraction <= 1) niceFraction = 1;
    else if (fraction <= 2) niceFraction = 2;
    else if (fraction <= 5) niceFraction = 5;
    else niceFraction = 10;
  }
  return niceFraction * 10 ** exponent;
}
