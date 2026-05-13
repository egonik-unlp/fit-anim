import React, { useMemo } from "react";
import type { Dataset } from "../data/presets";
import type { Model, Params } from "../models/types";
import { linearScale, niceTicks, fmt } from "./scales";

export type ResidualView = "segments" | "histogram" | "strips";

type Props = {
  model: Model;
  params: Params;
  data: Dataset;
  loss: number;
  step: number;
  view: ResidualView;
  showLossText: boolean;
  showAxes: boolean;
  showPolyEquation: boolean;
  showMlpDiagram: boolean;
  showErrorEquation: boolean;
};

// Outer SVG: 1000 x 600. Histogram view consumes a right strip.
const W = 1000;
const H = 600;
const HIST_W = 240;
const HIST_GAP = 24;

export function Plot({ model, params, data, loss, step, view, showLossText, showAxes, showPolyEquation, showMlpDiagram, showErrorEquation }: Props) {
  const isMlp = model.id.startsWith("mlp-");
  const equationVisible = isMlp ? showMlpDiagram : showPolyEquation;
  // The MLP diagram is taller than the polynomial equation, so the top margin grows when it's shown.
  const padTop = !equationVisible ? 32 : isMlp ? 170 : 78;
  const padBottom = showErrorEquation ? 100 : 48;
  const PAD = { top: padTop, right: 32, bottom: padBottom, left: 56 };

  const histActive = view === "histogram";
  const plotRight = histActive ? W - PAD.right - HIST_W - HIST_GAP : W - PAD.right;
  const x = linearScale([data.xMin, data.xMax], [PAD.left, plotRight]);
  const y = linearScale([data.yMin, data.yMax], [H - PAD.bottom, PAD.top]);

  // densely-sampled model curve
  const curvePath = useMemo(() => {
    const N = 400;
    let d = "";
    for (let i = 0; i <= N; i++) {
      const xv = data.xMin + (i / N) * (data.xMax - data.xMin);
      const yv = model.forward(xv, params);
      d += (i === 0 ? "M" : "L") + x(xv).toFixed(2) + " " + y(yv).toFixed(2) + " ";
    }
    return d;
  }, [model, params, data, plotRight]);

  // per-point predictions and residuals
  const preds = useMemo(() => data.xs.map((xv) => model.forward(xv, params)), [model, params, data]);
  const residuals = useMemo(() => data.ys.map((yv, i) => yv - preds[i]!), [data.ys, preds]);

  const xTicks = showAxes ? niceTicks(data.xMin, data.xMax, 5) : [];
  const yTicks = showAxes ? niceTicks(data.yMin, data.yMax, 5) : [];

  return (
    <svg
      id="plot-svg"
      xmlns="http://www.w3.org/2000/svg"
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      style={{ display: "block", background: "white" }}
    >
      <style>{`
        .axis { stroke: #111; stroke-width: 1; fill: none; }
        .tick-label { font: 12px ui-sans-serif, system-ui, sans-serif; fill: #333; }
        .data-dot { fill: #b8b8b8; }
        .pred-dot { fill: #111; }
        .resid-seg { stroke: #888; stroke-width: 0.8; opacity: 0.55; }
        .model-curve { stroke: #111; stroke-width: 1.8; fill: none; }
        .strip { fill: #111; opacity: 0.06; }
        .hist-bar { fill: #c8c8c8; }
        .hist-gauss { stroke: #111; stroke-width: 1.5; fill: none; }
        .hist-axis { stroke: #999; stroke-width: 0.8; }
        .info-text { font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; fill: #111; }
        .panel-title { font: 12px ui-sans-serif, system-ui, sans-serif; fill: #555; }
        .eq-text { font-family: "Iowan Old Style", "Palatino Linotype", Palatino, "Times New Roman", Georgia, serif; font-size: 22px; fill: #111; }
        .eq-text .num { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 19px; font-variant-numeric: tabular-nums; fill: #111; }
        .eq-text .var { font-style: italic; }
        .eq-text .op  { fill: #555; }
        .eq-text .sup { font-size: 0.7em; }
        .diag-edge-pos { stroke: #111; fill: none; }
        .diag-edge-neg { stroke: #b0b0b0; fill: none; stroke-dasharray: 2 2; }
        .diag-hidden { fill: #fff; stroke: #333; stroke-width: 1; }
        .diag-io { fill: #fff; stroke: #111; stroke-width: 1.6; }
        .diag-io-label { font-family: "Iowan Old Style", Palatino, Georgia, serif; font-style: italic; font-size: 13px; fill: #111; }
        .diag-cap { font: 10px ui-sans-serif, system-ui, sans-serif; fill: #888; text-transform: uppercase; letter-spacing: 0.12em; }
        .diag-ellipsis { font: 16px ui-sans-serif, system-ui, sans-serif; fill: #888; }
        .diag-tanh { font: italic 11px "Iowan Old Style", Palatino, Georgia, serif; fill: #444; }
      `}</style>

      {equationVisible && <EquationOverlay model={model} params={params} />}
      {showErrorEquation && <ErrorEquation loss={loss} n={data.xs.length} cx={W / 2} cy={H - 32} />}

      {/* axes */}
      {showAxes && (
        <g>
          <line className="axis" x1={PAD.left} y1={H - PAD.bottom} x2={plotRight} y2={H - PAD.bottom} />
          <line className="axis" x1={PAD.left} y1={H - PAD.bottom} x2={PAD.left} y2={PAD.top} />
          {xTicks.map((t, i) => (
            <g key={`xt${i}`}>
              <line className="axis" x1={x(t)} y1={H - PAD.bottom} x2={x(t)} y2={H - PAD.bottom + 4} />
              <text className="tick-label" x={x(t)} y={H - PAD.bottom + 18} textAnchor="middle">{fmt(t)}</text>
            </g>
          ))}
          {yTicks.map((t, i) => (
            <g key={`yt${i}`}>
              <line className="axis" x1={PAD.left - 4} y1={y(t)} x2={PAD.left} y2={y(t)} />
              <text className="tick-label" x={PAD.left - 8} y={y(t) + 4} textAnchor="end">{fmt(t)}</text>
            </g>
          ))}
        </g>
      )}

      {/* density strips view (drawn behind everything else) */}
      {view === "strips" && (
        <DensityStrips
          model={model}
          params={params}
          data={data}
          residuals={residuals}
          x={x}
          y={y}
          plotLeft={PAD.left}
          plotRight={plotRight}
        />
      )}

      {/* residual segments view */}
      {view === "segments" && (
        <g>
          {data.xs.map((xv, i) => (
            <line
              key={`r${i}`}
              className="resid-seg"
              x1={x(xv)}
              y1={y(data.ys[i]!)}
              x2={x(xv)}
              y2={y(preds[i]!)}
            />
          ))}
        </g>
      )}

      {/* data cloud */}
      <g>
        {data.xs.map((xv, i) => (
          <circle key={`d${i}`} className="data-dot" cx={x(xv)} cy={y(data.ys[i]!)} r={2.6} />
        ))}
      </g>

      {/* model predictions at each x (small ticks on the curve) — only in segments view, to keep the others uncluttered */}
      {view === "segments" && (
        <g>
          {data.xs.map((xv, i) => (
            <circle key={`p${i}`} className="pred-dot" cx={x(xv)} cy={y(preds[i]!)} r={1.4} />
          ))}
        </g>
      )}

      {/* model curve */}
      <path className="model-curve" d={curvePath} />

      {/* histogram panel */}
      {histActive && (
        <ResidualHistogram
          residuals={residuals}
          left={plotRight + HIST_GAP}
          right={W - PAD.right}
          top={PAD.top}
          bottom={H - PAD.bottom}
        />
      )}

      {/* info text — bottom-left of the plot frame, deferential to the equation up top */}
      {showLossText && (
        <g>
          <text className="info-text" x={PAD.left + 4} y={H - PAD.bottom - 10}>
            {`step ${step}   loss ${loss.toExponential(2)}`}
          </text>
        </g>
      )}
    </svg>
  );
}

// --- error function equation (bottom) ---
function ErrorEquation({ loss, n, cx, cy }: { loss: number; n: number; cx: number; cy: number }) {
  // L(θ) = (1/N) Σᵢ ( yᵢ − f(xᵢ; θ) )²   =   <live value>
  return (
    <g>
      <text className="eq-text" x={cx} y={cy} textAnchor="middle" style={{ fontSize: 19 }}>
        <tspan className="var">L</tspan>
        <tspan>(</tspan>
        <tspan className="var">θ</tspan>
        <tspan>)</tspan>
        <tspan className="op">{"  =  "}</tspan>
        <tspan>{"(1/"}</tspan>
        <tspan className="var">N</tspan>
        <tspan>{") "}</tspan>
        <tspan style={{ fontSize: "1.3em" }}>{"∑"}</tspan>
        <tspan className="sup" dy="0.35em">{`i=1..${n}`}</tspan>
        <tspan dy="-0.35em">{" ("}</tspan>
        <tspan className="var">y</tspan>
        <tspan className="sup" dy="0.35em">i</tspan>
        <tspan dy="-0.35em">{" "}</tspan>
        <tspan className="op">−</tspan>
        <tspan>{" f("}</tspan>
        <tspan className="var">x</tspan>
        <tspan className="sup" dy="0.35em">i</tspan>
        <tspan dy="-0.35em">{"; "}</tspan>
        <tspan className="var">θ</tspan>
        <tspan>{"))"}</tspan>
        <tspan className="sup" dy="-0.55em">2</tspan>
        <tspan dy="0.55em" className="op">{"   =   "}</tspan>
        <tspan className="num">{loss.toExponential(3)}</tspan>
      </text>
    </g>
  );
}

// --- live equation overlay ---
function EquationOverlay({ model, params }: { model: Model; params: Params }) {
  const isPoly = model.id.startsWith("poly-");
  const cx = W / 2;
  if (isPoly) {
    return <PolyEquation params={params} cx={cx} cy={44} />;
  }
  // MLP: extract hidden width from the model id ("mlp-16" → 16) and draw the diagram.
  const H_units = Number.parseInt(model.id.split("-")[1] ?? "0", 10) || 0;
  return <MlpDiagram params={params} hidden={H_units} cx={cx} cy={86} />;
}

function fmtCoef(v: number): string {
  const a = Math.abs(v);
  if (a < 0.005) return "0.00";
  if (a >= 100) return v.toExponential(1).replace("e", "·10^");
  return a.toFixed(2);
}

function PolyEquation({ params, cx, cy }: { params: Params; cx: number; cy: number }) {
  // Build: f(x) = a0 [± a1 x] [± a2 x²] ...
  const parts: React.ReactNode[] = [];
  // LHS
  parts.push(<tspan key="lhs1">f(</tspan>);
  parts.push(<tspan key="lhs2" className="var">x</tspan>);
  parts.push(<tspan key="lhs3">{") = "}</tspan>);

  for (let k = 0; k < params.length; k++) {
    const a = params[k]!;
    const negative = a < 0;
    if (k === 0) {
      if (negative) parts.push(<tspan key={`s${k}`} className="op">{"−"}</tspan>);
    } else {
      parts.push(<tspan key={`s${k}`} className="op">{negative ? " − " : " + "}</tspan>);
    }
    parts.push(<tspan key={`m${k}`} className="num">{fmtCoef(a)}</tspan>);
    if (k >= 1) {
      parts.push(<tspan key={`x${k}`}>{" "}</tspan>);
      parts.push(<tspan key={`xx${k}`} className="var">x</tspan>);
      if (k >= 2) {
        // superscript via dy, then reset
        parts.push(
          <tspan key={`p${k}`} className="sup" dy="-0.55em">{k}</tspan>
        );
        parts.push(<tspan key={`pr${k}`} dy="0.55em">{""}</tspan>);
      }
    }
  }

  return (
    <text className="eq-text" x={cx} y={cy} textAnchor="middle">
      {parts}
    </text>
  );
}

function MlpDiagram({ params, hidden, cx, cy }: { params: Params; hidden: number; cx: number; cy: number }) {
  if (hidden <= 0) return null;
  const DW = 420;
  const DH = 130;
  const left = cx - DW / 2;
  const top = cy - DH / 2;

  const MAX_VISIBLE = 12;
  const ellipsis = hidden > MAX_VISIBLE;
  const slots: Array<{ idx: number; kind: "node" | "dots" }> = [];
  if (ellipsis) {
    for (let i = 0; i < 5; i++) slots.push({ idx: i, kind: "node" });
    slots.push({ idx: -1, kind: "dots" });
    for (let i = hidden - 5; i < hidden; i++) slots.push({ idx: i, kind: "node" });
  } else {
    for (let i = 0; i < hidden; i++) slots.push({ idx: i, kind: "node" });
  }

  const inX = left + 36;
  const outX = left + DW - 36;
  const ioY = top + DH / 2;
  const hidX = left + DW / 2;
  const colTop = top + 10;
  const colBot = top + DH - 10;
  const slotY = (i: number) =>
    slots.length === 1 ? (colTop + colBot) / 2 : colTop + (i / (slots.length - 1)) * (colBot - colTop);

  const w1 = params.slice(0, hidden);
  const w2 = params.slice(2 * hidden, 3 * hidden);
  let maxAbs = 1e-6;
  for (const v of w1) if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
  for (const v of w2) if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
  const widthFor = (w: number) => 0.35 + (Math.abs(w) / maxAbs) * 2.0;
  const opacityFor = (w: number) => 0.18 + (Math.abs(w) / maxAbs) * 0.7;

  const edges: React.ReactNode[] = [];
  const nodes: React.ReactNode[] = [];

  slots.forEach((s, i) => {
    const yy = slotY(i);
    if (s.kind === "dots") {
      nodes.push(
        <text key="dots" x={hidX} y={yy + 5} textAnchor="middle" className="diag-ellipsis">
          {"⋮"}
        </text>
      );
      return;
    }
    const w1v = w1[s.idx]!;
    const w2v = w2[s.idx]!;
    edges.push(
      <line
        key={`e1-${s.idx}`}
        className={w1v >= 0 ? "diag-edge-pos" : "diag-edge-neg"}
        x1={inX + 9}
        y1={ioY}
        x2={hidX - 5}
        y2={yy}
        strokeWidth={widthFor(w1v)}
        opacity={opacityFor(w1v)}
      />
    );
    edges.push(
      <line
        key={`e2-${s.idx}`}
        className={w2v >= 0 ? "diag-edge-pos" : "diag-edge-neg"}
        x1={hidX + 5}
        y1={yy}
        x2={outX - 9}
        y2={ioY}
        strokeWidth={widthFor(w2v)}
        opacity={opacityFor(w2v)}
      />
    );
    nodes.push(<circle key={`n-${s.idx}`} className="diag-hidden" cx={hidX} cy={yy} r={4} />);
  });

  return (
    <g>
      {edges}
      <circle className="diag-io" cx={inX} cy={ioY} r={11} />
      <text className="diag-io-label" x={inX} y={ioY + 4} textAnchor="middle">x</text>
      {nodes}
      <circle className="diag-io" cx={outX} cy={ioY} r={11} />
      <text className="diag-io-label" x={outX} y={ioY + 4} textAnchor="middle">f</text>
      <text className="diag-cap" x={inX} y={top + DH + 14} textAnchor="middle">input</text>
      <text className="diag-cap" x={hidX} y={top + DH + 14} textAnchor="middle">
        {`hidden  ·  ${hidden}`}
      </text>
      <text className="diag-tanh" x={hidX} y={top - 6} textAnchor="middle">tanh</text>
      <text className="diag-cap" x={outX} y={top + DH + 14} textAnchor="middle">output</text>
    </g>
  );
}


// --- density strips ---
type StripsProps = {
  model: Model;
  params: Params;
  data: Dataset;
  residuals: number[];
  x: (v: number) => number;
  y: (v: number) => number;
  plotLeft: number;
  plotRight: number;
};

function DensityStrips({ model, params, data, residuals, x, y, plotLeft, plotRight }: StripsProps) {
  const sigma = useMemo(() => {
    if (residuals.length === 0) return 1;
    let m = 0;
    for (const r of residuals) m += r;
    m /= residuals.length;
    let v = 0;
    for (const r of residuals) v += (r - m) * (r - m);
    return Math.sqrt(v / residuals.length) || 1e-6;
  }, [residuals]);

  // Render ~24 strips across x range.
  const strips: React.ReactNode[] = [];
  const N = 24;
  // strip width in pixels: half spacing
  const spacing = (plotRight - plotLeft) / N;
  const stripHalf = spacing * 0.42;
  for (let i = 0; i <= N; i++) {
    const xv = data.xMin + (i / N) * (data.xMax - data.xMin);
    const cy = model.forward(xv, params);
    // sample density vertically
    const M = 30;
    const path: string[] = [];
    // build left edge then right edge polygon for a smooth strip
    // density at y: exp(-(y-mu)^2 / (2 sigma^2))
    const cxPix = x(xv);
    // left side, top to bottom
    for (let j = 0; j <= M; j++) {
      const t = j / M;
      const yv = data.yMin + t * (data.yMax - data.yMin);
      const d = Math.exp(-((yv - cy) ** 2) / (2 * sigma * sigma));
      const wpx = stripHalf * d;
      path.push(`${j === 0 ? "M" : "L"} ${(cxPix - wpx).toFixed(2)} ${y(yv).toFixed(2)}`);
    }
    // right side, bottom to top
    for (let j = M; j >= 0; j--) {
      const t = j / M;
      const yv = data.yMin + t * (data.yMax - data.yMin);
      const d = Math.exp(-((yv - cy) ** 2) / (2 * sigma * sigma));
      const wpx = stripHalf * d;
      path.push(`L ${(cxPix + wpx).toFixed(2)} ${y(yv).toFixed(2)}`);
    }
    path.push("Z");
    strips.push(<path key={`s${i}`} className="strip" d={path.join(" ")} />);
  }
  return <g>{strips}</g>;
}

// --- residual histogram + gaussian overlay ---
type HistProps = {
  residuals: number[];
  left: number;
  right: number;
  top: number;
  bottom: number;
};

function ResidualHistogram({ residuals, left, right, top, bottom }: HistProps) {
  const w = right - left;
  const h = bottom - top;
  if (residuals.length === 0) return null;
  let mean = 0;
  for (const r of residuals) mean += r;
  mean /= residuals.length;
  let v = 0;
  for (const r of residuals) v += (r - mean) * (r - mean);
  const sigma = Math.sqrt(v / residuals.length) || 1e-6;

  // Fixed range so bars stay anchored as fit improves (the bunching-up around 0 is the visual story).
  let amax = 0;
  for (const r of residuals) amax = Math.max(amax, Math.abs(r));
  // pick a stable-ish range: round up to nice number from largest seen residual at init.
  // But we don't track init here; use 4*sigma_initial-ish heuristic via amax with a floor.
  const range = Math.max(amax * 1.1, sigma * 4, 0.2);
  const xMin = -range;
  const xMax = range;

  const bins = 24;
  const counts = new Array<number>(bins).fill(0);
  for (const r of residuals) {
    if (r < xMin || r > xMax) continue;
    let bi = Math.floor(((r - xMin) / (xMax - xMin)) * bins);
    if (bi >= bins) bi = bins - 1;
    counts[bi]!++;
  }
  const binW = (xMax - xMin) / bins;
  // normalise to density so the overlay matches scale
  const density = counts.map((c) => c / (residuals.length * binW));
  // Gaussian density values along x grid
  const G = 80;
  const gxs: number[] = [];
  const gys: number[] = [];
  for (let i = 0; i <= G; i++) {
    const xv = xMin + (i / G) * (xMax - xMin);
    const d = Math.exp(-((xv - mean) ** 2) / (2 * sigma * sigma)) / (sigma * Math.sqrt(2 * Math.PI));
    gxs.push(xv);
    gys.push(d);
  }
  const yMax = Math.max(...density, ...gys) * 1.1 || 1;

  const sx = linearScale([xMin, xMax], [left, right]);
  const sy = linearScale([0, yMax], [bottom, top]);

  return (
    <g>
      <text className="panel-title" x={left} y={top - 10}>residuals (y − f(x))</text>
      {/* baseline */}
      <line className="hist-axis" x1={left} y1={bottom} x2={right} y2={bottom} />
      {/* x=0 marker */}
      <line className="hist-axis" x1={sx(0)} y1={top} x2={sx(0)} y2={bottom} strokeDasharray="2 3" />
      {/* bars */}
      {density.map((d, i) => {
        const xL = xMin + i * binW;
        const xR = xL + binW;
        const pxL = sx(xL);
        const pxR = sx(xR);
        const pyT = sy(d);
        return (
          <rect
            key={`b${i}`}
            className="hist-bar"
            x={pxL + 0.5}
            y={pyT}
            width={Math.max(0, pxR - pxL - 1)}
            height={Math.max(0, bottom - pyT)}
          />
        );
      })}
      {/* gaussian overlay */}
      <path
        className="hist-gauss"
        d={gxs.map((xv, i) => `${i === 0 ? "M" : "L"} ${sx(xv).toFixed(2)} ${sy(gys[i]!).toFixed(2)}`).join(" ")}
      />
      <text className="tick-label" x={right} y={bottom + 16} textAnchor="end">
        {`σ ≈ ${fmt(sigma)}`}
      </text>
    </g>
  );
}
