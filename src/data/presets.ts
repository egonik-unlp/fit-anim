import { mulberry32, gaussian } from "../models/rng";

export type Dataset = {
  /** training fold: x coordinates */
  xs: number[];
  /** training fold: y coordinates */
  ys: number[];
  /** optional held-out test fold (undefined when testRatio is 0) */
  test?: { xs: number[]; ys: number[] };
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
  opts: {
    n?: number;
    noise?: number;
    seed?: number;
    xMin?: number;
    xMax?: number;
    /** fraction of points held out as a test fold (0..0.5). default 0. */
    testRatio?: number;
  } = {}
): Dataset {
  const n = opts.n ?? 80;
  const noise = opts.noise ?? 0.15;
  const seed = opts.seed ?? 1;
  const xMin = opts.xMin ?? -1;
  const xMax = opts.xMax ?? 1;
  const testRatio = Math.min(0.5, Math.max(0, opts.testRatio ?? 0));
  const rand = mulberry32(seed);
  const truth = TRUTHS[preset];

  // Generate all points first; bounding box covers both folds.
  const allX = new Array<number>(n);
  const allY = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const x = xMin + (xMax - xMin) * rand();
    allX[i] = x;
    allY[i] = truth(x) + gaussian(rand) * noise;
  }

  // y range with margin, covering all points (train + test).
  let yMin = Infinity, yMax = -Infinity;
  for (const y of allY) {
    if (y < yMin) yMin = y;
    if (y > yMax) yMax = y;
  }
  const pad = (yMax - yMin) * 0.15 || 0.2;
  yMin -= pad;
  yMax += pad;

  // Deterministic shuffle for the train/test partition, seeded from the same seed
  // (so split is stable across rerenders for given (seed, testRatio), but a
  // "new data" click reshuffles both samples and the partition).
  const nTest = Math.floor(n * testRatio);
  if (nTest === 0) {
    return { xs: allX, ys: allY, truth, xMin, xMax, yMin, yMax };
  }

  const splitRand = mulberry32((seed * 2654435761) >>> 0);
  const order = new Array<number>(n);
  for (let i = 0; i < n; i++) order[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(splitRand() * (i + 1));
    const tmp = order[i]!;
    order[i] = order[j]!;
    order[j] = tmp;
  }
  const testIdx = order.slice(n - nTest);
  const trainIdx = order.slice(0, n - nTest);

  const xs = trainIdx.map((i) => allX[i]!);
  const ys = trainIdx.map((i) => allY[i]!);
  const testXs = testIdx.map((i) => allX[i]!);
  const testYs = testIdx.map((i) => allY[i]!);

  return {
    xs,
    ys,
    test: { xs: testXs, ys: testYs },
    truth,
    xMin,
    xMax,
    yMin,
    yMax,
  };
}
