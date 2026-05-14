import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Plot, type ResidualView } from "./plot/Plot";
import { LossCurve } from "./plot/LossCurve";
import { makePoly } from "./models/poly";
import { makeMLP } from "./models/mlp";
import type { Model, Params } from "./models/types";
import { makeDataset, PRESET_LABELS, type PresetId } from "./data/presets";
import { step as trainStep, loss as computeLoss } from "./training/loop";
import { rasteriseSvgsStacked, serializeSvg, downloadBlob, canvasToBlob, nextFrame } from "./export/raster";
import { recordWebm } from "./export/webm";
import { recordGif, recordPngSequence } from "./export/gif";

type ModelKind = "poly" | "mlp";
type ExportKind = "webm" | "gif" | "zip";

function makeModel(kind: ModelKind, param: number): Model {
  return kind === "poly" ? makePoly(param) : makeMLP(param);
}

export function App() {
  // --- model & data config ---
  const [modelKind, setModelKind] = useState<ModelKind>("poly");
  const [polyDeg, setPolyDeg] = useState(3);
  const [mlpHidden, setMlpHidden] = useState(16);
  const [preset, setPreset] = useState<PresetId>("cubic");
  const [noise, setNoise] = useState(0.15);
  const [nPoints, setNPoints] = useState(80);
  const [dataSeed, setDataSeed] = useState(1);
  const [paramSeed, setParamSeed] = useState(1);

  const model = useMemo<Model>(
    () => makeModel(modelKind, modelKind === "poly" ? polyDeg : mlpHidden),
    [modelKind, polyDeg, mlpHidden]
  );
  const data = useMemo(
    () => makeDataset(preset, { n: nPoints, noise, seed: dataSeed }),
    [preset, nPoints, noise, dataSeed]
  );

  // --- training state ---
  const [params, setParams] = useState<Params>(() => model.init(paramSeed));
  const [stepN, setStepN] = useState(0);
  const [lr, setLr] = useState(modelKind === "poly" ? 0.05 : 0.05);
  const [speed, setSpeed] = useState(30);
  const [playing, setPlaying] = useState(false);

  // re-init params when model identity changes
  useEffect(() => {
    setParams(model.init(paramSeed));
    setStepN(0);
    setPlaying(false);
  }, [model, paramSeed]);

  // sensible default lr per model
  useEffect(() => {
    if (modelKind === "poly") setLr(polyDeg <= 2 ? 0.1 : 0.05);
    else setLr(0.05);
  }, [modelKind, polyDeg]);

  const curLoss = useMemo(() => computeLoss(model, data, params), [model, data, params]);

  // --- play loop ---
  useEffect(() => {
    if (!playing) return;
    let cancelled = false;
    const tick = () => {
      if (cancelled) return;
      setParams((p) => {
        const res = trainStep(model, data, p, lr);
        return res.params;
      });
      setStepN((s) => s + 1);
    };
    const id = window.setInterval(tick, Math.max(8, 1000 / speed));
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [playing, model, data, lr, speed]);

  const doStep = useCallback(() => {
    setParams((p) => trainStep(model, data, p, lr).params);
    setStepN((s) => s + 1);
  }, [model, data, lr]);

  const doReset = useCallback(() => {
    setParams(model.init(paramSeed));
    setStepN(0);
    setPlaying(false);
  }, [model, paramSeed]);

  const newData = useCallback(() => setDataSeed((s) => s + 1), []);

  // --- view options ---
  const [view, setView] = useState<ResidualView>("segments");
  const [showAxes, setShowAxes] = useState(true);
  const [showLossText, setShowLossText] = useState(true);
  const [showPolyEquation, setShowPolyEquation] = useState(true);
  const [showMlpDiagram, setShowMlpDiagram] = useState(false);
  const [showErrorEquation, setShowErrorEquation] = useState(false);
  const [showLossCurve, setShowLossCurve] = useState(false);
  const [cleanMode, setCleanMode] = useState(false);

  // Loss history for the independent loss-curve plot.
  const [history, setHistory] = useState<Array<{ step: number; loss: number }>>(
    () => [{ step: 0, loss: 0 }]
  );
  useEffect(() => {
    setHistory((h) => {
      const last = h[h.length - 1];
      if (last && last.step === stepN) {
        const copy = h.slice(0, -1);
        copy.push({ step: stepN, loss: curLoss });
        return copy;
      }
      const next = h.concat({ step: stepN, loss: curLoss });
      return next.length > 4000 ? next.slice(-4000) : next;
    });
  }, [stepN, curLoss]);
  // Clear history whenever the training arc resets.
  useEffect(() => {
    setHistory([{ step: 0, loss: curLoss }]);
    // intentionally only on identity-of-arc changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, data, paramSeed]);

  useEffect(() => {
    document.body.classList.toggle("body--clean", cleanMode);
  }, [cleanMode]);

  // Esc exits clean mode.
  useEffect(() => {
    if (!cleanMode) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setCleanMode(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cleanMode]);

  // --- refs ---
  const svgWrapperRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const exportMainRef = useRef<HTMLDivElement>(null);
  const exportLossRef = useRef<HTMLDivElement>(null);
  const recordAbortRef = useRef<AbortController | null>(null);
  const [exportStatus, setExportStatus] = useState("");
  const [exporting, setExporting] = useState(false);
  const [pngScale, setPngScale] = useState(2);

  // Export-time plot selection (independent of on-screen toggles).
  const [exportPlots, setExportPlots] = useState({ main: true, lossCurve: false });
  // When set, the inline picker is shown for that export kind. null = no pending export.
  const [pendingExport, setPendingExport] = useState<ExportKind | null>(null);

  function getMainSvg(): SVGSVGElement | null {
    return svgWrapperRef.current?.querySelector("svg") ?? null;
  }
  function getExportSvgs(): SVGSVGElement[] {
    const out: SVGSVGElement[] = [];
    if (exportPlots.main) {
      const s = exportMainRef.current?.querySelector("svg") as SVGSVGElement | null;
      if (s) out.push(s);
    }
    if (exportPlots.lossCurve) {
      const s = exportLossRef.current?.querySelector("svg") as SVGSVGElement | null;
      if (s) out.push(s);
    }
    return out;
  }

  // --- export handlers ---
  const exportSvg = useCallback(async () => {
    const svg = getMainSvg();
    if (!svg) return;
    const xml = serializeSvg(svg);
    downloadBlob(new Blob([xml], { type: "image/svg+xml" }), `fit-anim-${Date.now()}.svg`);
    setExportStatus("svg saved");
  }, []);

  const exportPng = useCallback(async () => {
    const svg = getMainSvg();
    const canvas = canvasRef.current;
    if (!svg || !canvas) return;
    await rasteriseSvgsStacked([svg], canvas, pngScale);
    const blob = await canvasToBlob(canvas, "image/png");
    downloadBlob(blob, `fit-anim-${Date.now()}.png`);
    setExportStatus(`png saved (${(blob.size / 1024).toFixed(0)} kB, ${canvas.width}×${canvas.height})`);
  }, [pngScale]);

  // Paint helper used by the recorders: drives React state for the given params,
  // then waits for paint so the SVG node reflects them.
  const paint = useCallback(
    async (p: Params, s: number, _l: number) => {
      setParams(p);
      setStepN(s);
      await nextFrame();
    },
    []
  );

  const runRecording = useCallback(
    async (
      kind: ExportKind,
      sessionExtras: { fps?: number; keepEvery?: number } = {}
    ) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      // Ensure the hidden export-mount has flushed at least one frame so its SVGs exist.
      setExporting(true);
      setPlaying(false);
      await nextFrame();
      const svgs = getExportSvgs();
      if (svgs.length === 0) {
        setExportStatus("no plots selected");
        setExporting(false);
        return;
      }
      // reset training so the recording always shows the same arc
      const init = model.init(paramSeed);
      setParams(init);
      setStepN(0);
      await nextFrame();
      // pre-size the canvas based on stacked plot dimensions
      await rasteriseSvgsStacked(svgs, canvas, 1);

      const controller = new AbortController();
      recordAbortRef.current = controller;

      try {
        const common = {
          lr,
          maxSteps: 600,
          gradTol: 1e-3,
          milestoneDwellFrames: Math.round((sessionExtras.fps ?? 30) * 1.0),
          startDwellFrames: Math.round((sessionExtras.fps ?? 30) * 0.6),
          endDwellFrames: Math.round((sessionExtras.fps ?? 30) * 1.2),
          fps: sessionExtras.fps ?? 30,
          keepEvery: sessionExtras.keepEvery,
          onPaint: paint,
          signal: controller.signal,
        };
        if (kind === "webm") {
          await recordWebm(svgs, canvas, model, data, init, common, setExportStatus);
        } else if (kind === "gif") {
          await recordGif(svgs, canvas, model, data, init, common, setExportStatus);
        } else {
          await recordPngSequence(svgs, canvas, model, data, init, common, setExportStatus);
        }
      } catch (e) {
        console.error(e);
        setExportStatus(`error: ${(e as Error).message}`);
      } finally {
        recordAbortRef.current = null;
        setExporting(false);
      }
    },
    [model, data, lr, paramSeed, paint, exportPlots]
  );

  const beginExport = useCallback((kind: ExportKind) => {
    setPendingExport(kind);
  }, []);

  const confirmExport = useCallback(() => {
    const kind = pendingExport;
    if (!kind) return;
    setPendingExport(null);
    const extras =
      kind === "webm" ? { fps: 30 }
      : kind === "gif" ? { fps: 20, keepEvery: 2 }
      : { fps: 30, keepEvery: 1 };
    void runRecording(kind, extras);
  }, [pendingExport, runRecording]);

  const stopRecording = useCallback(() => {
    recordAbortRef.current?.abort();
    setExportStatus((s) => s + " · stopping…");
  }, []);

  // Mount the hidden export plots only when we need them, to avoid the cost of
  // rendering two extra plots during normal interactive use.
  const exportMounted = pendingExport !== null || exporting;

  return (
    <>
      <div className="titlebar">
        <h1>fit-anim</h1>
        <span className="subtitle">iterative function fitting — for slide exports</span>
      </div>
      <div className="app">
        <div className="plot-card" ref={svgWrapperRef}>
          <Plot
            model={model}
            params={params}
            data={data}
            loss={curLoss}
            step={stepN}
            view={view}
            showLossText={showLossText}
            showAxes={showAxes}
            showPolyEquation={showPolyEquation}
            showMlpDiagram={showMlpDiagram}
            showErrorEquation={showErrorEquation}
          />
        </div>
        {showLossCurve && (
          <div className="loss-card">
            <LossCurve history={history} />
          </div>
        )}
        <div className="panel">
          <h3>Model</h3>
          <div className="row">
            <label>Family</label>
            <div className="toggle-group">
              <button className={modelKind === "poly" ? "active" : ""} onClick={() => setModelKind("poly")}>Polynomial</button>
              <button className={modelKind === "mlp" ? "active" : ""} onClick={() => setModelKind("mlp")}>MLP</button>
            </div>
          </div>
          {modelKind === "poly" ? (
            <div className="row">
              <label>Degree <span className="value">{polyDeg}</span></label>
              <input type="range" min={1} max={6} step={1} value={polyDeg} onChange={(e) => setPolyDeg(+e.target.value)} />
            </div>
          ) : (
            <div className="row">
              <label>Hidden units <span className="value">{mlpHidden}</span></label>
              <input type="range" min={4} max={32} step={2} value={mlpHidden} onChange={(e) => setMlpHidden(+e.target.value)} />
            </div>
          )}
          <div className="row">
            <label>Learning rate <span className="value">{lr.toFixed(3)}</span></label>
            <input type="range" min={0.001} max={0.5} step={0.001} value={lr} onChange={(e) => setLr(+e.target.value)} />
          </div>
          <div className="row">
            <label>Param seed <span className="value">{paramSeed}</span></label>
            <div className="btn-row">
              <button className="btn" onClick={() => setParamSeed((s) => s + 1)}>new init</button>
            </div>
          </div>

          <div className="divider" />
          <h3>Data</h3>
          <div className="row">
            <label>Preset</label>
            <select value={preset} onChange={(e) => setPreset(e.target.value as PresetId)}>
              {(Object.keys(PRESET_LABELS) as PresetId[]).map((k) => (
                <option key={k} value={k}>{PRESET_LABELS[k]}</option>
              ))}
            </select>
          </div>
          <div className="row">
            <label>Noise σ <span className="value">{noise.toFixed(2)}</span></label>
            <input type="range" min={0} max={0.5} step={0.01} value={noise} onChange={(e) => setNoise(+e.target.value)} />
          </div>
          <div className="row">
            <label>Points <span className="value">{nPoints}</span></label>
            <input type="range" min={20} max={300} step={10} value={nPoints} onChange={(e) => setNPoints(+e.target.value)} />
          </div>
          <div className="btn-row">
            <button className="btn" onClick={newData}>new data</button>
          </div>

          <div className="divider" />
          <h3>Pacing</h3>
          <div className="btn-row">
            <button className="btn primary" onClick={() => setPlaying((p) => !p)}>
              {playing ? "pause" : "play"}
            </button>
            <button className="btn" onClick={doStep} disabled={playing}>step</button>
            <button className="btn" onClick={doReset}>reset</button>
          </div>
          <div className="row">
            <label>Speed (steps/sec) <span className="value">{speed}</span></label>
            <input type="range" min={1} max={60} step={1} value={speed} onChange={(e) => setSpeed(+e.target.value)} />
          </div>
          <div className="row">
            <label className="value">step {stepN}  ·  loss {curLoss.toExponential(2)}</label>
          </div>

          <div className="divider" />
          <h3>View</h3>
          <div className="row">
            <label>Residual view</label>
            <div className="toggle-group">
              <button className={view === "segments" ? "active" : ""} onClick={() => setView("segments")}>segments</button>
              <button className={view === "histogram" ? "active" : ""} onClick={() => setView("histogram")}>histogram</button>
              <button className={view === "strips" ? "active" : ""} onClick={() => setView("strips")}>strips</button>
            </div>
          </div>
          <label className="checkbox-row"><input type="checkbox" checked={showAxes} onChange={(e) => setShowAxes(e.target.checked)} /> axes</label>
          <label className="checkbox-row"><input type="checkbox" checked={showLossText} onChange={(e) => setShowLossText(e.target.checked)} /> step / loss label</label>
          <label className="checkbox-row"><input type="checkbox" checked={showPolyEquation} onChange={(e) => setShowPolyEquation(e.target.checked)} /> polynomial equation</label>
          <label className="checkbox-row"><input type="checkbox" checked={showMlpDiagram} onChange={(e) => setShowMlpDiagram(e.target.checked)} /> MLP diagram</label>
          <label className="checkbox-row"><input type="checkbox" checked={showErrorEquation} onChange={(e) => setShowErrorEquation(e.target.checked)} /> error function + value</label>
          <label className="checkbox-row"><input type="checkbox" checked={showLossCurve} onChange={(e) => setShowLossCurve(e.target.checked)} /> loss curve (independent plot)</label>
          <label className="checkbox-row"><input type="checkbox" checked={cleanMode} onChange={(e) => setCleanMode(e.target.checked)} /> clean mode (hide UI)</label>

          <div className="divider" />
          <h3>Export</h3>
          <div className="row">
            <label>PNG scale <span className="value">{pngScale}×</span></label>
            <input type="range" min={1} max={4} step={1} value={pngScale} onChange={(e) => setPngScale(+e.target.value)} />
          </div>
          <div className="btn-row">
            <button className="btn" onClick={exportPng} disabled={exporting}>PNG</button>
            <button className="btn" onClick={exportSvg} disabled={exporting}>SVG</button>
          </div>

          {exporting ? (
            <div className="btn-row">
              <button className="btn" onClick={stopRecording}>Stop</button>
            </div>
          ) : pendingExport ? (
            <div className="export-picker">
              <span className="title">include in {pendingExport}</span>
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={exportPlots.main}
                  onChange={(e) => setExportPlots((s) => ({ ...s, main: e.target.checked }))}
                />
                main plot
              </label>
              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={exportPlots.lossCurve}
                  onChange={(e) => setExportPlots((s) => ({ ...s, lossCurve: e.target.checked }))}
                />
                loss curve
              </label>
              <div className="btn-row">
                <button
                  className="btn primary"
                  onClick={confirmExport}
                  disabled={!exportPlots.main && !exportPlots.lossCurve}
                >
                  Record
                </button>
                <button className="btn" onClick={() => setPendingExport(null)}>Cancel</button>
              </div>
            </div>
          ) : (
            <div className="btn-row">
              <button className="btn" onClick={() => beginExport("webm")}>WebM</button>
              <button className="btn" onClick={() => beginExport("gif")}>GIF</button>
              <button className="btn" onClick={() => beginExport("zip")}>PNG zip</button>
            </div>
          )}
          <div className="export-status">{exportStatus}</div>
        </div>
      </div>
      <canvas id="hidden-canvas" ref={canvasRef} width={1000} height={600} />

      {/* Hidden mount used as the canonical source for export rasterisation.
          Mounted only while a recording is being configured or running. */}
      {exportMounted && (
        <div className="export-mount" aria-hidden>
          <div ref={exportMainRef} style={{ width: 1000 }}>
            <Plot
              model={model}
              params={params}
              data={data}
              loss={curLoss}
              step={stepN}
              view={view}
              showLossText={showLossText}
              showAxes={showAxes}
              showPolyEquation={showPolyEquation}
              showMlpDiagram={showMlpDiagram}
              showErrorEquation={showErrorEquation}
            />
          </div>
          <div ref={exportLossRef} style={{ width: 1000 }}>
            <LossCurve history={history} />
          </div>
        </div>
      )}

      {cleanMode && (
        <button
          className="zen-exit"
          onClick={() => setCleanMode(false)}
          title="Exit clean mode (Esc)"
          aria-label="Exit clean mode"
        >
          exit clean mode  ·  esc
        </button>
      )}
    </>
  );
}
