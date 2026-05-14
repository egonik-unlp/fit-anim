import { mulberry32, gaussian } from "../models/rng";

/** Maps internal normalized x/y values ∈ [-1, 1] to real-world units for axis labels.
 * Training stays in normalized space so polynomial / MLP fits don't depend on scale. */
export type DisplayMeta = {
  xToReal: (xNorm: number) => number;
  yToReal: (yNorm: number) => number;
  /** Suffix appended to formatted x-axis tick values (e.g. " km", " min", " °C"). */
  xUnit?: string;
  yUnit?: string;
  /** Prefix prepended to formatted values (e.g. "$"). */
  xPrefix?: string;
  yPrefix?: string;
  xLabel?: string;
  yLabel?: string;
};

export type Dataset = {
  xs: number[];
  ys: number[];
  test?: { xs: number[]; ys: number[] };
  truth: (x: number) => number;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  /** optional display layer for showing real-world units on the axes */
  display?: DisplayMeta;
};

export type PresetId =
  | "line"
  | "parabola"
  | "cubic"
  | "sine"
  | "step"
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
  /** optional display mapping that turns normalized coords into real-world units */
  display?: DisplayMeta;
  /** plain-text formula shown alongside the preset (with real parameter values) */
  formula?: string;
  /** Spanish description explaining the real-world scenario and its parameters */
  description?: string;
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
  // x ∈ [-1, 1] → distancia 0..15 km
  // y ∈ [-1, 1] → costo $1500..$7500
  // costo = $1500 + $400/km · distancia
  taxi: {
    id: "taxi",
    label: "Viaje en taxi",
    category: "real",
    // truth_norm(x) = ((1500 + 400*dist) - 4500) / 3000 = (400*7.5*(x+1) - 3000)/3000 = x
    truth: (x) => x,
    display: {
      xToReal: (x) => 7.5 * (x + 1),
      yToReal: (y) => 4500 + 3000 * y,
      xUnit: " km",
      yPrefix: "$",
      xLabel: "distancia",
      yLabel: "costo del viaje",
    },
    formula: "costo = $1.500 + $400/km · distancia",
    description:
      "Un viaje en taxi tiene una bajada de bandera fija ($1.500) más un costo proporcional a la distancia ($400 por km). Probá polinomial grado 1: la ordenada al origen que aprende es la bajada de bandera, y la pendiente es el costo por km.",
  },

  // x ∈ [-1, 1] → tiempo 0..2 s
  // y ∈ [-1, 1] → altura 0..5 m
  // altura(t) = 10·t − 5·t²  ⇒  truth_norm(x) = 1 − 2x²
  proyectil: {
    id: "proyectil",
    label: "Pelota lanzada al aire",
    category: "real",
    truth: (x) => 1 - 2 * x * x,
    display: {
      xToReal: (x) => x + 1,
      yToReal: (y) => 2.5 + 2.5 * y,
      xUnit: " s",
      yUnit: " m",
      xLabel: "tiempo",
      yLabel: "altura",
    },
    formula: "altura(t) = 10·t − 5·t²   (v₀ = 10 m/s, g ≈ 10 m/s²)",
    description:
      "Tirás una pelota hacia arriba a 10 m/s. Sube hasta 5 m al segundo 1 y vuelve al piso al segundo 2. Probá polinomial grado 2: la parábola se ajusta de forma exacta.",
  },

  // x ∈ [-1, 1] → tiempo 0..15 min
  // y ∈ [-1, 1] → temperatura 20..80 °C
  // T(t) = 20 + 60·e^(−0.15·t)   ⇒  truth_norm(x) = 2·e^(−1.125·(x+1)) − 1
  enfriamiento: {
    id: "enfriamiento",
    label: "Enfriamiento del mate",
    category: "real",
    truth: (x) => 2 * Math.exp(-1.125 * (x + 1)) - 1,
    display: {
      xToReal: (x) => 7.5 * (x + 1),
      yToReal: (y) => 50 + 30 * y,
      xUnit: " min",
      yUnit: " °C",
      xLabel: "tiempo",
      yLabel: "temperatura",
    },
    formula: "T(t) = 20 + 60·e^(−0.15·t)",
    description:
      "Servís un mate a 80 °C en un ambiente a 20 °C. Al principio se enfría rápido, después cada vez más lento; a los 15 min está cerca de los 25 °C. Un polinomio bajo no captura bien la asíntota — probá MLP.",
  },

  // x ∈ [-1, 1] → días 0..60
  // y ∈ [-1, 1] → altura 0..100 cm
  // h(t) = 100 / (1 + e^(−0.15·(t − 30)))   ⇒  truth_norm(x) = tanh(2.25·x)
  crecimiento: {
    id: "crecimiento",
    label: "Crecimiento de una planta",
    category: "real",
    truth: (x) => Math.tanh(2.25 * x),
    display: {
      xToReal: (x) => 30 * (x + 1),
      yToReal: (y) => 50 + 50 * y,
      xUnit: " días",
      yUnit: " cm",
      xLabel: "tiempo desde el brote",
      yLabel: "altura",
    },
    formula: "h(t) = 100 / (1 + e^(−0.15·(t − 30)))",
    description:
      "Una planta crece de 0 a 100 cm a lo largo de 60 días, pero no de manera uniforme: arranca lento, acelera entre los días 20–40, y se estanca al llegar a su tamaño máximo. La curva es una sigmoide. Probá MLP.",
  },

  // x ∈ [-1, 1] → meses 0..12
  // y ∈ [-1, 1] → precio $800..$2000
  // precio(t) = $1000 · (1,05)^t   ⇒  precio en [$1000, ~$1796]
  inflacion: {
    id: "inflacion",
    label: "Inflación acumulada",
    category: "real",
    truth: (x) => (1000 * Math.pow(1.05, 6 * (x + 1)) - 1400) / 600,
    display: {
      xToReal: (x) => 6 * (x + 1),
      yToReal: (y) => 1400 + 600 * y,
      xUnit: " meses",
      yPrefix: "$",
      xLabel: "tiempo",
      yLabel: "precio acumulado",
    },
    formula: "precio(t) = $1.000 · (1,05)^t",
    description:
      "Un producto que vale $1.000 hoy, con una inflación mensual del 5% (interés compuesto), llega a casi $1.800 a los 12 meses — un acumulado de ~80% anual. No es lineal: cada mes el aumento absoluto es más grande. Probá polinomial grado 3 o MLP.",
  },

  // x ∈ [-1, 1] → hora del día 0..24
  // y ∈ [-1, 1] → tarifa $300..$500
  // tarifa = $500 entre 0–6 h, $300 entre 6–24 h
  // step at x = -0.5 (= 6 h)
  tarifa_horario: {
    id: "tarifa_horario",
    label: "Tarifa estacionamiento (día/noche)",
    category: "real",
    truth: (x) => (x < -0.5 ? 1 : -1),
    display: {
      xToReal: (x) => 12 * (x + 1),
      yToReal: (y) => 400 + 100 * y,
      xUnit: " h",
      yPrefix: "$",
      xLabel: "hora del día",
      yLabel: "tarifa por hora",
    },
    formula: "tarifa = $500 entre 0–6 h (nocturna), $300 entre 6–24 h (diurna)",
    description:
      "Un estacionamiento cobra una tarifa nocturna más alta entre la medianoche y las 6 de la mañana ($500/h), y diurna durante el resto del día ($300/h). La función pega un salto a las 6 h. Los polinomios no pueden representar saltos bruscos — probá MLP.",
  },
};

export const PRESET_LABELS: Record<PresetId, string> = Object.fromEntries(
  (Object.keys(PRESETS) as PresetId[]).map((k) => [k, PRESETS[k].label])
) as Record<PresetId, string>;

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
  const meta = PRESETS[preset];
  const truth = meta.truth;

  const allX = new Array<number>(n);
  const allY = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const x = xMin + (xMax - xMin) * rand();
    allX[i] = x;
    allY[i] = truth(x) + gaussian(rand) * noise;
  }

  let yMin = Infinity, yMax = -Infinity;
  for (const y of allY) {
    if (y < yMin) yMin = y;
    if (y > yMax) yMax = y;
  }
  const pad = (yMax - yMin) * 0.15 || 0.2;
  yMin -= pad;
  yMax += pad;

  const baseDs: Omit<Dataset, "xs" | "ys" | "test"> = {
    truth,
    xMin,
    xMax,
    yMin,
    yMax,
    display: meta.display,
  };

  const nTest = Math.floor(n * testRatio);
  if (nTest === 0) {
    return { ...baseDs, xs: allX, ys: allY };
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

  return {
    ...baseDs,
    xs: trainIdx.map((i) => allX[i]!),
    ys: trainIdx.map((i) => allY[i]!),
    test: {
      xs: testIdx.map((i) => allX[i]!),
      ys: testIdx.map((i) => allY[i]!),
    },
  };
}
