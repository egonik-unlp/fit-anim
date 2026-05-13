import { mulberry32, gaussian } from "../models/rng";

export type Dataset = {
  xs: number[];
  ys: number[];
  /** the noise-free truth, useful for plotting reference if we ever want it */
  truth: (x: number) => number;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
};

export type PresetId = "line" | "parabola" | "cubic" | "sine" | "step";

const TRUTHS: Record<PresetId, (x: number) => number> = {
  line: (x) => 0.6 * x + 0.2,
  parabola: (x) => 0.7 * x * x - 0.3,
  cubic: (x) => 0.5 * x * x * x - 0.4 * x,
  sine: (x) => Math.sin(Math.PI * x),
  step: (x) => (x < 0 ? -0.6 : 0.6),
};

export const PRESET_LABELS: Record<PresetId, string> = {
  line: "Line",
  parabola: "Parabola",
  cubic: "Cubic",
  sine: "Sine",
  step: "Step",
};

export function makeDataset(
  preset: PresetId,
  opts: { n?: number; noise?: number; seed?: number; xMin?: number; xMax?: number } = {}
): Dataset {
  const n = opts.n ?? 80;
  const noise = opts.noise ?? 0.15;
  const seed = opts.seed ?? 1;
  const xMin = opts.xMin ?? -1;
  const xMax = opts.xMax ?? 1;
  const rand = mulberry32(seed);
  const truth = TRUTHS[preset];
  const xs = new Array<number>(n);
  const ys = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const x = xMin + (xMax - xMin) * rand();
    xs[i] = x;
    ys[i] = truth(x) + gaussian(rand) * noise;
  }
  // y range with margin
  let yMin = Infinity, yMax = -Infinity;
  for (const y of ys) {
    if (y < yMin) yMin = y;
    if (y > yMax) yMax = y;
  }
  const pad = (yMax - yMin) * 0.15 || 0.2;
  return { xs, ys, truth, xMin, xMax, yMin: yMin - pad, yMax: yMax + pad };
}
