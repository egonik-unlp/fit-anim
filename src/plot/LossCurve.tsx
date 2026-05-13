import React, { useMemo } from "react";
import { linearScale, niceTicks, fmt } from "./scales";

type Props = {
  history: Array<{ step: number; loss: number }>;
  /** if true, render y-axis in log scale (default true since loss spans orders of magnitude) */
  logScale?: boolean;
};

const W = 1000;
const H = 220;
const PAD = { top: 36, right: 28, bottom: 38, left: 64 };

export function LossCurve({ history, logScale = true }: Props) {
  const visible = useMemo(() => history.filter((p) => Number.isFinite(p.loss) && p.loss > 0), [history]);

  // x range
  const minStep = visible.length ? visible[0]!.step : 0;
  const maxStep = visible.length ? Math.max(visible[visible.length - 1]!.step, minStep + 1) : 1;

  // y transform — use log10 of loss when logScale
  const yTransform = (v: number) => (logScale ? Math.log10(Math.max(v, 1e-12)) : v);

  // y range
  let yLo = +Infinity, yHi = -Infinity;
  for (const p of visible) {
    const y = yTransform(p.loss);
    if (y < yLo) yLo = y;
    if (y > yHi) yHi = y;
  }
  if (!Number.isFinite(yLo) || !Number.isFinite(yHi)) {
    yLo = 0;
    yHi = 1;
  }
  if (yHi - yLo < 1e-6) {
    yLo -= 0.5;
    yHi += 0.5;
  }
  // small padding
  const yPad = (yHi - yLo) * 0.08;
  yLo -= yPad;
  yHi += yPad;

  const sx = linearScale([minStep, maxStep], [PAD.left, W - PAD.right]);
  const sy = linearScale([yLo, yHi], [H - PAD.bottom, PAD.top]);

  const path = useMemo(() => {
    if (visible.length === 0) return "";
    let d = "";
    for (let i = 0; i < visible.length; i++) {
      const p = visible[i]!;
      d += (i === 0 ? "M" : "L") + sx(p.step).toFixed(2) + " " + sy(yTransform(p.loss)).toFixed(2) + " ";
    }
    return d;
  }, [visible, sx, sy, logScale]);

  // Ticks
  const xTicks = niceTicks(minStep, maxStep, 6);
  // y ticks in transformed space → convert back to label values
  const yTicksT = niceTicks(yLo, yHi, 4);
  const yTickLabel = (t: number) => (logScale ? Math.pow(10, t) : t);

  const last = visible[visible.length - 1];

  return (
    <svg
      id="loss-svg"
      xmlns="http://www.w3.org/2000/svg"
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      style={{ display: "block", background: "white" }}
    >
      <style>{`
        .loss-axis { stroke: #111; stroke-width: 1; fill: none; }
        .loss-tick { font: 11px ui-sans-serif, system-ui, sans-serif; fill: #555; }
        .loss-line { stroke: #111; stroke-width: 1.6; fill: none; }
        .loss-fill { fill: #111; opacity: 0.04; }
        .loss-dot { fill: #111; }
        .loss-title { font: 10px ui-sans-serif, system-ui, sans-serif; fill: #888; text-transform: uppercase; letter-spacing: 0.16em; }
        .loss-value { font: 13px ui-monospace, "SF Mono", Menlo, monospace; fill: #111; font-variant-numeric: tabular-nums; }
        .loss-axis-lbl { font: italic 11px "Iowan Old Style", Palatino, Georgia, serif; fill: #555; }
      `}</style>

      {/* title strip */}
      <text className="loss-title" x={PAD.left} y={20}>loss curve {logScale ? "(log scale)" : ""}</text>
      {last && (
        <text className="loss-value" x={W - PAD.right} y={20} textAnchor="end">
          {`step ${last.step}    L = ${last.loss.toExponential(3)}`}
        </text>
      )}

      {/* axes */}
      <line className="loss-axis" x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} />
      <line className="loss-axis" x1={PAD.left} y1={H - PAD.bottom} x2={PAD.left} y2={PAD.top} />

      {/* x ticks */}
      {xTicks.map((t, i) => (
        <g key={`xt${i}`}>
          <line className="loss-axis" x1={sx(t)} y1={H - PAD.bottom} x2={sx(t)} y2={H - PAD.bottom + 4} />
          <text className="loss-tick" x={sx(t)} y={H - PAD.bottom + 18} textAnchor="middle">{Math.round(t)}</text>
        </g>
      ))}

      {/* y ticks */}
      {yTicksT.map((t, i) => (
        <g key={`yt${i}`}>
          <line className="loss-axis" x1={PAD.left - 4} y1={sy(t)} x2={PAD.left} y2={sy(t)} />
          <text className="loss-tick" x={PAD.left - 8} y={sy(t) + 4} textAnchor="end">{fmt(yTickLabel(t))}</text>
        </g>
      ))}

      {/* line + last-step dot */}
      <path className="loss-line" d={path} />
      {last && (
        <circle className="loss-dot" cx={sx(last.step)} cy={sy(yTransform(last.loss))} r={3} />
      )}

      {/* axis labels */}
      <text className="loss-axis-lbl" x={(PAD.left + W - PAD.right) / 2} y={H - 6} textAnchor="middle">step</text>
      <text className="loss-axis-lbl" x={16} y={(PAD.top + H - PAD.bottom) / 2} transform={`rotate(-90 16 ${(PAD.top + H - PAD.bottom) / 2})`} textAnchor="middle">L(θ)</text>
    </svg>
  );
}
