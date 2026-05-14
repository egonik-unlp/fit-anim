import { rasteriseSvgsStacked, downloadBlob } from "./raster";
import { runAutoSession, type SessionOpts } from "./session";
import type { Model, Params } from "../models/types";
import type { Dataset } from "../data/presets";

export async function recordWebm(
  svgs: SVGSVGElement[],
  canvas: HTMLCanvasElement,
  model: Model,
  data: Dataset,
  initParams: Params,
  opts: SessionOpts & { onPaint: (params: Params, step: number, loss: number) => Promise<void> },
  setStatus: (s: string) => void
) {
  const fps = opts.fps ?? 30;
  const stream = canvas.captureStream(fps);
  const mime =
    MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9"
      : "video/webm";
  const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 4_000_000 });
  const chunks: Blob[] = [];
  rec.ondataavailable = (e) => e.data.size > 0 && chunks.push(e.data);
  const done = new Promise<void>((res) => (rec.onstop = () => res()));
  rec.start();

  setStatus("recording webm…");
  let frameIdx = 0;
  await runAutoSession(model, data, initParams, opts, async ({ params, loss, step }) => {
    await opts.onPaint(params, step, loss);
    await rasteriseSvgsStacked(svgs, canvas, 1);
    frameIdx++;
    if (frameIdx % 30 === 0) setStatus(`recording webm… frame ${frameIdx}, step ${step}`);
    // pace to roughly fps
    await new Promise((r) => setTimeout(r, 1000 / fps));
  });

  rec.stop();
  await done;
  const aborted = opts.signal?.aborted === true;
  if (chunks.length === 0) {
    setStatus(aborted ? "stopped, no frames captured" : "no frames captured");
    return;
  }
  const blob = new Blob(chunks, { type: mime });
  downloadBlob(blob, `fit-anim-${Date.now()}.webm`);
  setStatus(`webm saved (${(blob.size / 1024).toFixed(0)} kB${aborted ? ", partial" : ""})`);
}
