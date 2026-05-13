import { rasteriseSvg, downloadBlob, canvasToBlob } from "./raster";
import { runAutoSession, type SessionOpts } from "./session";
import type { Model, Params } from "../models/types";
import type { Dataset } from "../data/presets";
// @ts-expect-error  no types for gif.js
import GIF from "gif.js";
import { zip } from "fflate";

const GIF_WORKER_URL = "https://cdn.jsdelivr.net/npm/gif.js@0.2.0/dist/gif.worker.js";

type CommonOpts = SessionOpts & {
  onPaint: (params: Params, step: number, loss: number) => Promise<void>;
  /** keep every Nth frame (for size). default 2. */
  keepEvery?: number;
};

export async function recordGif(
  svg: SVGSVGElement,
  canvas: HTMLCanvasElement,
  model: Model,
  data: Dataset,
  initParams: Params,
  opts: CommonOpts,
  setStatus: (s: string) => void
) {
  const fps = opts.fps ?? 20;
  const keep = opts.keepEvery ?? 2;
  setStatus("recording gif…");

  const gif = new GIF({
    workers: 2,
    quality: 10,
    width: canvas.width || 1000,
    height: canvas.height || 600,
    workerScript: GIF_WORKER_URL,
    background: "#ffffff",
  });

  let frameIdx = 0;
  await runAutoSession(model, data, initParams, opts, async ({ params, loss, step }) => {
    await opts.onPaint(params, step, loss);
    await rasteriseSvg(svg, canvas, 1);
    if (frameIdx % keep === 0) {
      gif.addFrame(canvas, { copy: true, delay: Math.round(1000 / (fps / keep)) });
    }
    frameIdx++;
    if (frameIdx % 30 === 0) setStatus(`gif: capturing… frame ${frameIdx}`);
  });

  setStatus("gif: encoding…");
  const blob: Blob = await new Promise((res, rej) => {
    gif.on("finished", (b: Blob) => res(b));
    gif.on("abort", () => rej(new Error("gif aborted")));
    gif.render();
  });
  downloadBlob(blob, `fit-anim-${Date.now()}.gif`);
  setStatus(`gif saved (${(blob.size / 1024).toFixed(0)} kB)`);
}

export async function recordPngSequence(
  svg: SVGSVGElement,
  canvas: HTMLCanvasElement,
  model: Model,
  data: Dataset,
  initParams: Params,
  opts: CommonOpts,
  setStatus: (s: string) => void
) {
  const keep = opts.keepEvery ?? 1;
  setStatus("capturing PNG sequence…");
  const files: Record<string, Uint8Array> = {};
  let frameIdx = 0;
  let saved = 0;
  await runAutoSession(model, data, initParams, opts, async ({ params, loss, step }) => {
    await opts.onPaint(params, step, loss);
    await rasteriseSvg(svg, canvas, 1);
    if (frameIdx % keep === 0) {
      const blob = await canvasToBlob(canvas, "image/png");
      const ab = await blob.arrayBuffer();
      const name = `frame_${String(saved).padStart(5, "0")}.png`;
      files[name] = new Uint8Array(ab);
      saved++;
    }
    frameIdx++;
    if (frameIdx % 20 === 0) setStatus(`capturing… frame ${frameIdx}`);
  });

  setStatus(`zipping ${saved} frames…`);
  const zipped: Uint8Array = await new Promise((res, rej) =>
    zip(files, { level: 6 }, (err, data) => (err ? rej(err) : res(data)))
  );
  downloadBlob(new Blob([zipped.buffer as ArrayBuffer], { type: "application/zip" }), `fit-anim-${Date.now()}.zip`);
  setStatus(`zip saved (${(zipped.byteLength / 1024).toFixed(0)} kB, ${saved} frames)`);
}
