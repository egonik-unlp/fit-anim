/** Rasterise the given SVG element into a canvas at the given scale. */
export function rasteriseSvg(
  svg: SVGSVGElement,
  canvas: HTMLCanvasElement,
  scale = 1
): Promise<void> {
  return new Promise((resolve, reject) => {
    const vb = svg.viewBox.baseVal;
    const w = (vb.width || svg.clientWidth) * scale;
    const h = (vb.height || svg.clientHeight) * scale;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, w, h);

    const clone = svg.cloneNode(true) as SVGSVGElement;
    if (!clone.getAttribute("xmlns")) clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    if (!clone.getAttribute("xmlns:xlink")) clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
    const xml = new XMLSerializer().serializeToString(clone);
    const svgBlob = new Blob([xml], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(svgBlob);
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0, w, h);
      URL.revokeObjectURL(url);
      resolve();
    };
    img.onerror = (e) => {
      URL.revokeObjectURL(url);
      reject(e);
    };
    img.src = url;
  });
}

/** Rasterise multiple SVGs vertically stacked onto a single canvas.
 * Canvas width = max(viewBox.width_i) * scale, height = sum(viewBox.height_i) * scale.
 * Each SVG is painted at its intrinsic viewBox aspect ratio, top-aligned, centred horizontally. */
export function rasteriseSvgsStacked(
  svgs: SVGSVGElement[],
  canvas: HTMLCanvasElement,
  scale = 1
): Promise<void> {
  if (svgs.length === 0) return Promise.resolve();

  const sizes = svgs.map((svg) => {
    const vb = svg.viewBox.baseVal;
    return {
      w: vb.width || svg.clientWidth || 1000,
      h: vb.height || svg.clientHeight || 600,
    };
  });
  const totalW = Math.max(...sizes.map((s) => s.w)) * scale;
  const totalH = sizes.reduce((a, s) => a + s.h, 0) * scale;
  canvas.width = totalW;
  canvas.height = totalH;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, totalW, totalH);

  let yOffset = 0;
  const tasks = svgs.map((svg, i) => {
    const w = sizes[i]!.w * scale;
    const h = sizes[i]!.h * scale;
    const xOffset = (totalW - w) / 2;
    const y = yOffset;
    yOffset += h;

    const clone = svg.cloneNode(true) as SVGSVGElement;
    if (!clone.getAttribute("xmlns")) clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    if (!clone.getAttribute("xmlns:xlink")) clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
    const xml = new XMLSerializer().serializeToString(clone);
    const svgBlob = new Blob([xml], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(svgBlob);

    return new Promise<void>((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, xOffset, y, w, h);
        URL.revokeObjectURL(url);
        resolve();
      };
      img.onerror = (e) => {
        URL.revokeObjectURL(url);
        reject(e);
      };
      img.src = url;
    });
  });

  return Promise.all(tasks).then(() => undefined);
}

export function serializeSvg(svg: SVGSVGElement): string {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  if (!clone.getAttribute("xmlns")) clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  if (!clone.getAttribute("xmlns:xlink")) clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
  return '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(clone);
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function nextFrame(): Promise<void> {
  return new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())));
}

export function canvasToBlob(canvas: HTMLCanvasElement, type = "image/png"): Promise<Blob> {
  return new Promise((res, rej) =>
    canvas.toBlob((b) => (b ? res(b) : rej(new Error("toBlob failed"))), type)
  );
}
