"use client";

// DecisionGauge — a circular risk gauge showing the fraud probability and
// the resulting action decision, with color-coded bands.

const COLORS: Record<string, string> = {
  allow: "#2dd4bf",
  review: "#f59e0b",
  hold: "#f43f5e",
};

export default function DecisionGauge({
  probability,
  action,
}: {
  probability: number;
  action: string;
}) {
  const pct = Math.round(probability * 100);
  const color = COLORS[action] ?? "#94a3b8";
  const r = 84;
  const circ = 2 * Math.PI * r;
  const filled = circ * probability;

  return (
    <div className="gauge" style={{ ["--gauge" as string]: color }}>
      <svg viewBox="0 0 200 200" width="220" height="220">
        <circle
          className="gauge-track"
          cx="100"
          cy="100"
          r={r}
          fill="none"
          strokeWidth="16"
        />
        <circle
          className="gauge-fill"
          cx="100"
          cy="100"
          r={r}
          fill="none"
          strokeWidth="16"
          strokeDasharray={`${filled} ${circ}`}
          strokeLinecap="round"
          transform="rotate(-90 100 100)"
        />
        <text x="100" y="108" textAnchor="middle" className="gauge-value">
          {pct}%
        </text>
        <text x="100" y="132" textAnchor="middle" className="gauge-label">
          fraud probability
        </text>
      </svg>
      <div className="gauge-action" style={{ background: color }}>
        {action.toUpperCase()}
      </div>
    </div>
  );
}
