import type { Model, Params } from "../models/types";
import type { Dataset } from "../data/presets";

export type StepResult = {
  params: Params;
  loss: number;
  gradNorm: number;
};

/** One full-batch SGD step on MSE loss. Pure: returns new params, doesn't mutate. */
export function step(model: Model, data: Dataset, params: Params, lr: number): StepResult {
  const n = data.xs.length;
  const acc = new Array<number>(params.length).fill(0);
  let loss = 0;
  for (let i = 0; i < n; i++) {
    const x = data.xs[i]!;
    const y = data.ys[i]!;
    const g = model.gradient(x, y, params);
    for (let k = 0; k < params.length; k++) acc[k]! += g[k]!;
    const yhat = model.forward(x, params);
    const e = yhat - y;
    loss += e * e;
  }
  loss /= n;
  let gradNorm2 = 0;
  const newParams = params.slice();
  for (let k = 0; k < params.length; k++) {
    const g = acc[k]! / n;
    gradNorm2 += g * g;
    newParams[k] = params[k]! - lr * g;
  }
  return { params: newParams, loss, gradNorm: Math.sqrt(gradNorm2) };
}

export function loss(model: Model, data: Dataset, params: Params): number {
  const n = data.xs.length;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const e = model.forward(data.xs[i]!, params) - data.ys[i]!;
    s += e * e;
  }
  return s / n;
}

export function residuals(model: Model, data: Dataset, params: Params): number[] {
  const out = new Array<number>(data.xs.length);
  for (let i = 0; i < data.xs.length; i++) {
    out[i] = data.ys[i]! - model.forward(data.xs[i]!, params);
  }
  return out;
}

/** Milestones for auto-run-with-pauses: initial, half-loss, near-converged, final. */
export type Milestone = "initial" | "half" | "near" | "final";

export type AutoRunState = {
  initialLoss: number;
  hitHalf: boolean;
  hitNear: boolean;
  done: boolean;
};

export function newAutoRun(initialLoss: number): AutoRunState {
  return { initialLoss, hitHalf: false, hitNear: false, done: false };
}

export function checkMilestone(
  s: AutoRunState,
  res: StepResult,
  opts: { gradTol?: number; maxStepsReached?: boolean } = {}
): Milestone | null {
  const tol = opts.gradTol ?? 1e-3;
  if (!s.hitHalf && res.loss <= s.initialLoss * 0.5) {
    s.hitHalf = true;
    return "half";
  }
  if (!s.hitNear && res.gradNorm <= tol) {
    s.hitNear = true;
    return "near";
  }
  if (!s.done && (opts.maxStepsReached || (s.hitNear && res.gradNorm <= tol * 0.5))) {
    s.done = true;
    return "final";
  }
  return null;
}
