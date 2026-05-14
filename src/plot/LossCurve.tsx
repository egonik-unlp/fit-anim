import React, { useMemo } from "react";
import { linearScale, niceTicks, fmt } from "./scales";

export type LossPoint = { step: number; train: number; test?: number };

type Props = {
  history: LossPoint[];
  /** if true, render y-axis in log scale (default true since loss spans orders of magnitude) */
  logScale?: boolean;
};

const W = 1000;
const H = 220;
const PAD = { top: 36, right: 28, bottom: 38, left: 64 };

export function LossCurve({ history, logScale = true }: Props) {
  const hasTest = useMemo(
    () => history.some((p) => p.test !== undefined && Number.isFinite(p.test)),
    [history]
  );

  const yTransform = (v: number) => (logScale ? Math.log10(Math.max(v, 1e-12)) : v);

  // Pre-compute visible train / test points (filter out non-positive in log mode).
  const visibleTrain = useMemo(
    () => history.filter((p) => Number.isFinite(p.train) && (!logScale || p.train > 0)),
    [history, logScale]
  );
  const visibleTest = useMemo(
    () =>
      history.filter(
        (p) => p.test !== undefined && Number.isFinite(p.test) && (!logScale || p.test! > 0)
      ),
    [history, logScale]
  );

  const allVisible = visibleTrain.length + visibleTest.length > 0;
  const firstStep = allVisible
    ? Math.min(
        visibleTrain[0]?.step ?? Infinity,
        visibleTest[0]?.step ?? Infinity
      )
    : 0;
  const lastStep = allVisible
    ? Math.max(
        visibleTrain[visibleTrain.length - 1]?.step ?? -Infinity,
        visibleTest[visibleTest.length - 1]?.step ?? -Infinity
      )
    : 1;
  const minStep = firstStep;
  const maxStep = Math.max(lastStep, minStep + 1);

  // y range from union of both series
  let yLo = +Infinity, yHi = -Infinity;
  for (const p of visibleTrain) {
    const y = yTransform(p.train);
    if (y < yLo) yLo = y;
    if (y > yHi) yHi = y;
  }
  for (const p of visibleTest) {
    const y = yTransform(p.test!);
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
  const yPad = (yHi - yLo) * 0.08;
  yLo -= yPad;
  yHi += yPad;

  const sx = linearScale([minStep, maxStep], [PAD.left, W - PAD.right]);
  const sy = linearScale([yLo, yHi], [H - PAD.bottom, PAD.top]);

  const buildPath = (pts: LossPoint[], key: "train" | "test") => {
    if (pts.length === 0) return "";
    let d = "";
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i]!;
      const v = key === "train" ? p.train : p.test!;
      d += (i === 0 ? "M" : "L") + sx(p.step).toFixed(2) + " " + sy(yTransform(v)).toFixed(2) + " ";
    }
    return d;
  };
  const trainPath = useMemo(() => buildPath(visibleTrain, "train"), [visibleTrain, sx, sy, logScale]);
  const testPath = useMemo(() => buildPath(visibleTest, "test"), [visibleTest, sx, sy, logScale]);

  const xTicks = niceTicks(minStep, maxStep, 6);
  const yTicksT = niceTicks(yLo, yHi, 4);
  const yTickLabel = (t: number) => (logScale ? Math.pow(10, t) : t);

  const lastTrain = visibleTrain[visibleTrain.length - 1];
  const lastTest = visibleTest[visibleTest.length - 1];

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
        .loss-line-test { stroke: #1e6bb8; stroke-width: 1.4; fill: none; opacity: 0.95; }
        .loss-fill { fill: #111; opacity: 0.04; }
        .loss-dot { fill: #111; }
        .loss-dot-test { fill: #1e6bb8; }
        .loss-title { font: 10px ui-sans-serif, system-ui, sans-serif; fill: #888; text-transform: uppercase; letter-spacing: 0.16em; }
        .loss-value { font: 13px ui-monospace, "SF Mono", Menlo, monospace; fill: #111; font-variant-numeric: tabular-nums; }
        .loss-value-test { fill: #1e6bb8; }
        .loss-axis-lbl { font: italic 11px "Iowan Old Style", Palatino, Georgia, serif; fill: #555; }
      `}</style>

      <text className="loss-title" x={PAD.left} y={20}>loss curve {logScale ? "(log scale)" : ""}</text>
      {lastTrain && (
        <text className="loss-value" x={W - PAD.right} y={20} textAnchor="end">
          {hasTest
            ? `step ${lastTrain.step}    train ${lastTrain.train.toExponential(3)}`
            : `step ${lastTrain.step}    L = ${lastTrain.train.toExponential(3)}`}
        </text>
      )}
      {hasTest && lastTest && (
        <text className="loss-value loss-value-test" x={W - PAD.right} y={36} textAnchor="end">
          {`test ${lastTest.test!.toExponential(3)}`}
        </text>
      )}

      <line className="loss-axis" x1={PAD.left} y1={H - PAD.bottom} x2={W - PAD.right} y2={H - PAD.bottom} />
      <line className="loss-axis" x1={PAD.left} y1={H - PAD.bottom} x2={PAD.left} y2={PAD.top} />

      {xTicks.map((t, i) => (
        <g key={`xt${i}`}>
          <line className="loss-axis" x1={sx(t)} y1={H - PAD.bottom} x2={sx(t)} y2={H - PAD.bottom + 4} />
          <text className="loss-tick" x={sx(t)} y={H - PAD.bottom + 18} textAnchor="middle">{Math.round(t)}</text>
        </g>
      ))}

      {yTicksT.map((t, i) => (
        <g key={`yt${i}`}>
          <line className="loss-axis" x1={PAD.left - 4} y1={sy(t)} x2={PAD.left} y2={sy(t)} />
          <text className="loss-tick" x={PAD.left - 8} y={sy(t) + 4} textAnchor="end">{fmt(yTickLabel(t))}</text>
        </g>
      ))}

      <path className="loss-line" d={trainPath} />
      {hasTest && <path className="loss-line-test" d={testPath} />}
      {lastTrain && (
        <circle className="loss-dot" cx={sx(lastTrain.step)} cy={sy(yTransform(lastTrain.train))} r={3} />
      )}
      {hasTest && lastTest && (
        <circle className="loss-dot-test" cx={sx(lastTest.step)} cy={sy(yTransform(lastTest.test!))} r={3} />
      )}

      <text className="loss-axis-lbl" x={(PAD.left + W - PAD.right) / 2} y={H - 6} textAnchor="middle">step</text>
      <text className="loss-axis-lbl" x={16} y={(PAD.top + H - PAD.bottom) / 2} transform={`rotate(-90 16 ${(PAD.top + H - PAD.bottom) / 2})`} textAnchor="middle">L(θ)</text>
    </svg>
  );
}
