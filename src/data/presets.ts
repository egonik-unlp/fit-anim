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

export type PresetId =
  // Abstract / mathematical
  | "line"
  | "parabola"
  | "cubic"
  | "sine"
  | "step"
  // Casos reales (Argentina)
  | "taxi"
  | "proyectil"
  | "enfriamiento"
  | "crecimiento"
  | "inflacion"
  | "tarifa_horario";

export type PresetCategory = "abstracto" | "real";

export type PresetMeta = {
  id: PresetId;
  label: string;
  category: PresetCategory;
  /** the noise-free truth defined over x ∈ [-1, 1] */
  truth: (x: number) => number;
  /** plain-text formula shown alongside the preset */
  formula?: string;
  /** longer Spanish description explaining the real-world meaning */
  description?: string;
  /** what the normalized x-axis represents in the real-world reading */
  xMeaning?: string;
  /** what the normalized y-axis represents in the real-world reading */
  yMeaning?: string;
};

export const PRESETS: Record<PresetId, PresetMeta> = {
  // --- Abstract families ---
  line: {
    id: "line",
    label: "Línea",
    category: "abstracto",
    truth: (x) => 0.6 * x + 0.2,
    formula: "f(x) = a · x + b",
  },
  parabola: {
    id: "parabola",
    label: "Parábola",
    category: "abstracto",
    truth: (x) => 0.7 * x * x - 0.3,
    formula: "f(x) = a · x² + b",
  },
  cubic: {
    id: "cubic",
    label: "Cúbica",
    category: "abstracto",
    truth: (x) => 0.5 * x * x * x - 0.4 * x,
    formula: "f(x) = a · x³ + b · x",
  },
  sine: {
    id: "sine",
    label: "Sinusoide",
    category: "abstracto",
    truth: (x) => Math.sin(Math.PI * x),
    formula: "f(x) = sen(π · x)",
  },
  step: {
    id: "step",
    label: "Escalón",
    category: "abstracto",
    truth: (x) => (x < 0 ? -0.6 : 0.6),
    formula: "f(x) = a si x < 0, b si x ≥ 0",
  },

  // --- Casos reales (Argentina) ---
  taxi: {
    id: "taxi",
    label: "Viaje en taxi",
    category: "real",
    truth: (x) => 0.3 + 0.5 * x,
    formula: "costo = bajada_de_bandera + costo_por_km · distancia",
    description:
      "El costo de un viaje en taxi crece linealmente con la distancia. La intersección con el eje y es la bajada de bandera; la pendiente es el costo por kilómetro.",
    xMeaning: "distancia recorrida",
    yMeaning: "costo del viaje",
  },
  proyectil: {
    id: "proyectil",
    label: "Pelota lanzada al aire",
    category: "real",
    // Opens downward, peak ~0.6 at x=0, zeros near x = ±0.93
    truth: (x) => 0.6 - 0.7 * x * x,
    formula: "altura(t) = v₀ · t − ½ · g · t²",
    description:
      "Si tirás una pelota hacia arriba, sube, alcanza una altura máxima y vuelve a caer. La trayectoria en función del tiempo es una parábola hacia abajo.",
    xMeaning: "tiempo desde el lanzamiento",
    yMeaning: "altura de la pelota",
  },
  enfriamiento: {
    id: "enfriamiento",
    label: "Enfriamiento del mate",
    category: "real",
    // Exponential decay, monotonic decreasing, asymptote around -0.3
    // At x=-1 (t=0): 0.7·1 + (-0.3) = 0.4
    // At x=1 (t=2): 0.7·exp(-3) + (-0.3) ≈ -0.265
    truth: (x) => 0.7 * Math.exp(-1.5 * (x + 1)) - 0.3,
    formula: "T(t) = T_amb + (T₀ − T_amb) · e^(−k·t)",
    description:
      "El mate se enfría hacia la temperatura ambiente: al principio cae rápido, después cada vez más lento. Es un decaimiento exponencial.",
    xMeaning: "tiempo desde que se sirvió",
    yMeaning: "temperatura del mate",
  },
  crecimiento: {
    id: "crecimiento",
    label: "Crecimiento de una planta",
    category: "real",
    // Sigmoid centred at 0, range about [-0.4, 0.4]
    truth: (x) => 0.8 / (1 + Math.exp(-3 * x)) - 0.4,
    formula: "h(t) = L / (1 + e^(−k(t − t₀)))",
    description:
      "Una planta arranca lento, crece rápido en el medio de su ciclo y se estanca al acercarse a su tamaño máximo. Es una curva sigmoidea (logística).",
    xMeaning: "tiempo desde que brotó",
    yMeaning: "altura de la planta",
  },
  inflacion: {
    id: "inflacion",
    label: "Inflación acumulada",
    category: "real",
    // Map x ∈ [-1,1] to t ∈ [0, 12] months, r = 5% / month compounded.
    // price(t) = 1.05^t ranges in [1, ~1.796]. Center+scale to [-0.8, 0.8].
    truth: (x) => (Math.pow(1.05, (x + 1) * 6) - 1.4) / 0.5,
    formula: "precio(t) = precio₀ · (1 + r)^t",
    description:
      "Con una tasa de inflación mensual constante, el precio acumulado crece exponencialmente (interés compuesto). Una tasa del 5% mensual implica casi un 80% anual.",
    xMeaning: "meses transcurridos",
    yMeaning: "precio acumulado",
  },
  tarifa_horario: {
    id: "tarifa_horario",
    label: "Tarifa día/noche",
    category: "real",
    truth: (x) => (x < 0 ? -0.4 : 0.5),
    formula: "tarifa = T_día si hora < 22, T_noche si hora ≥ 22",
    description:
      "Algunos servicios (estacionamiento, ciertos transportes) cambian de tarifa según el horario. La función pega un salto al pasar de la franja diurna a la nocturna.",
    xMeaning: "hora del día",
    yMeaning: "tarifa aplicada",
  },
};

/** Backwards-compatible label lookup, in the same order as PRESETS. */
export const PRESET_LABELS: Record<PresetId, string> = Object.fromEntries(
  (Object.keys(PRESETS) as PresetId[]).map((k) => [k, PRESETS[k].label])
) as Record<PresetId, string>;

/** Ordered list of preset ids, grouped by category, useful for rendering optgroups. */
export const PRESETS_BY_CATEGORY: Record<PresetCategory, PresetId[]> = {
  abstracto: (Object.keys(PRESETS) as PresetId[]).filter((k) => PRESETS[k].category === "abstracto"),
  real: (Object.keys(PRESETS) as PresetId[]).filter((k) => PRESETS[k].category === "real"),
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
  const truth = PRESETS[preset].truth;

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
