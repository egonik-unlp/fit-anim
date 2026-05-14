import type { Model, Params } from "../models/types";
import type { Dataset } from "../data/presets";
import { step, loss as computeLoss, newAutoRun, checkMilestone } from "../training/loop";

export type FrameCb = (info: {
  params: Params;
  loss: number;
  step: number;
  milestone: "initial" | "half" | "near" | "final" | null;
}) => void | Promise<void>;

export type SessionOpts = {
  lr: number;
  maxSteps?: number;
  gradTol?: number;
  /** ms to dwell at each milestone, emitting the same frame */
  milestonePauseMs?: number;
  /** how many "dwell" duplicate frames to emit at a milestone */
  milestoneDwellFrames?: number;
  /** target frames per second of the OUTPUT animation; controls step pacing */
  fps?: number;
  /** initial extra frames at the start */
  startDwellFrames?: number;
  /** final extra frames after convergence */
  endDwellFrames?: number;
  /** optional cancellation; the session exits between frames if aborted */
  signal?: AbortSignal;
};

/** Run a full auto-training session, calling onFrame for each frame to record.
 * The session takes over: resets params, steps until converged or maxSteps,
 * pauses for milestoneDwellFrames duplicate emissions on milestones. */
export async function runAutoSession(
  model: Model,
  data: Dataset,
  initParams: Params,
  opts: SessionOpts,
  onFrame: FrameCb
) {
  const maxSteps = opts.maxSteps ?? 800;
  const gradTol = opts.gradTol ?? 1e-3;
  const dwell = opts.milestoneDwellFrames ?? 24;
  const startDwell = opts.startDwellFrames ?? 18;
  const endDwell = opts.endDwellFrames ?? 36;
  const signal = opts.signal;

  let params = initParams.slice();
  let curLoss = computeLoss(model, data, params);
  const auto = newAutoRun(curLoss);

  // initial frame (dwell)
  for (let i = 0; i < startDwell; i++) {
    if (signal?.aborted) return;
    await onFrame({ params, loss: curLoss, step: 0, milestone: i === 0 ? "initial" : null });
  }

  let s = 0;
  while (s < maxSteps && !auto.done) {
    if (signal?.aborted) return;
    const res = step(model, data, params, opts.lr);
    params = res.params;
    curLoss = res.loss;
    s++;
    const ms = checkMilestone(auto, res, { gradTol, maxStepsReached: s >= maxSteps });
    await onFrame({ params, loss: curLoss, step: s, milestone: ms });
    if (ms === "half" || ms === "near") {
      for (let i = 0; i < dwell; i++) {
        if (signal?.aborted) return;
        await onFrame({ params, loss: curLoss, step: s, milestone: null });
      }
    }
  }
  // end dwell
  for (let i = 0; i < endDwell; i++) {
    if (signal?.aborted) return;
    await onFrame({ params, loss: curLoss, step: s, milestone: i === 0 ? "final" : null });
  }
}
