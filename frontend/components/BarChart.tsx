"use client";

import { useState } from "react";

export interface Bar {
  key: string;
  label: string;
  value: number;
  color: string;
  ciLo?: number;
  ciHi?: number;
}

interface BarChartProps {
  bars: Bar[];
  height?: number;
  valueFormat?: (v: number) => string;
  yLabel?: string;
}

const MARGIN = { top: 16, right: 24, bottom: 56, left: 56 };
const BAR_MAX_WIDTH = 24;

export default function BarChart({ bars, height = 320, valueFormat = (v) => v.toFixed(4), yLabel }: BarChartProps) {
  const [hoverKey, setHoverKey] = useState<string | null>(null);
  const width = 640;

  const allVals = bars.flatMap((b) => [b.value, b.ciLo ?? b.value, b.ciHi ?? b.value]);
  const yMax = Math.max(...allVals) * 1.15;
  const yMin = 0;

  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;
  const slot = plotW / bars.length;
  const barWidth = Math.min(BAR_MAX_WIDTH, slot * 0.5);

  const sy = (y: number) => MARGIN.top + plotH - ((y - yMin) / (yMax - yMin || 1)) * plotH;

  const ticksY = niceTicksLinear(yMin, yMax, 5);

  return (
    <div className="w-full">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" style={{ background: "var(--surface-1)" }}>
        {ticksY.map((t) => (
          <g key={t}>
            <line x1={MARGIN.left} x2={width - MARGIN.right} y1={sy(t)} y2={sy(t)} stroke="var(--gridline)" strokeWidth={1} />
            <text x={MARGIN.left - 8} y={sy(t)} textAnchor="end" dominantBaseline="middle" fontSize={11} fill="var(--text-muted)">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        <line x1={MARGIN.left} x2={width - MARGIN.right} y1={sy(0)} y2={sy(0)} stroke="var(--axis)" strokeWidth={1} />

        {bars.map((b, i) => {
          const cx = MARGIN.left + slot * i + slot / 2;
          const barTop = sy(b.value);
          const barBottom = sy(0);
          const isHover = hoverKey === b.key;
          return (
            <g
              key={b.key}
              onMouseEnter={() => setHoverKey(b.key)}
              onMouseLeave={() => setHoverKey(null)}
              style={{ cursor: "pointer" }}
            >
              {/* hit target wider than the bar */}
              <rect x={cx - slot / 2} y={MARGIN.top} width={slot} height={plotH} fill="transparent" />

              {b.ciLo !== undefined && b.ciHi !== undefined && (
                <line x1={cx} x2={cx} y1={sy(b.ciHi)} y2={sy(b.ciLo)} stroke="var(--text-muted)" strokeWidth={1.5} />
              )}

              <rect
                x={cx - barWidth / 2}
                y={barTop}
                width={barWidth}
                height={barBottom - barTop}
                rx={4}
                fill={b.color}
                opacity={isHover ? 0.85 : 1}
              />
              <text x={cx} y={barTop - 8} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--text-primary)">
                {valueFormat(b.value)}
              </text>
              <text x={cx} y={height - MARGIN.bottom + 18} textAnchor="middle" fontSize={11} fill="var(--text-secondary)">
                {b.label}
              </text>
            </g>
          );
        })}

        {yLabel && (
          <text
            x={-(MARGIN.top + plotH / 2)}
            y={14}
            textAnchor="middle"
            fontSize={11}
            fill="var(--text-muted)"
            transform="rotate(-90)"
          >
            {yLabel}
          </text>
        )}
      </svg>
    </div>
  );
}

function niceTicksLinear(min: number, max: number, count: number): number[] {
  if (min === max) return [min];
  const range = max - min;
  const rough = range / (count - 1);
  const mag = 10 ** Math.floor(Math.log10(rough));
  const norm = rough / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const ticks: number[] = [];
  for (let v = 0; v <= max + step * 1e-6; v += step) ticks.push(Math.round(v * 1e6) / 1e6);
  return ticks;
}
