// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * Pure canvas drawing primitives for Studio simulation/analysis plots.
 *
 * Kept free of React and store dependencies so tick scaling and line/axis
 * rendering can be unit-tested without mounting SimulationPlot.
 */

export const PLOT_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373", "#ce93d8", "#90a4ae"] as const;
export const PLOT_BG = "#0d1117";
export const PLOT_PANEL_BG = "#0a0e14";
export const PLOT_GRID = "#1a1f2a";
export const PLOT_AXIS = "#484f58";
export const PLOT_BORDER = "#21262d";

/**
 * Choose a "nice" major tick step for a numeric range.
 *
 * @param range - Span of the axis (max − min).
 * @param ticks - Approximate number of major ticks desired.
 * @returns Positive step size on a 1–2–5×10ⁿ ladder, or 1 for non-positive range.
 */
export function niceStep(range: number, ticks: number): number {
  if (range <= 0 || !isFinite(range)) return 1;
  const rough = range / ticks;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const n = rough / mag;
  return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
}

/**
 * Paint a dark plot panel with grid lines and axis tick labels.
 */
export function drawAxes(
  ctx: CanvasRenderingContext2D,
  x0: number,
  y0: number,
  pw: number,
  ph: number,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
  xLabel?: string,
): void {
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  ctx.fillStyle = PLOT_PANEL_BG;
  ctx.fillRect(x0, y0, pw, ph);
  ctx.strokeStyle = PLOT_BORDER;
  ctx.lineWidth = 1;
  ctx.strokeRect(x0, y0, pw, ph);

  ctx.strokeStyle = PLOT_GRID;
  ctx.lineWidth = 0.5;
  ctx.font = "10px monospace";
  ctx.fillStyle = PLOT_AXIS;
  ctx.textAlign = "right";
  const ys = niceStep(yRange, 4);
  for (let v = Math.ceil(yMin / ys) * ys; v <= yMax; v += ys) {
    const y = y0 + ph - ((v - yMin) / yRange) * ph;
    if (y < y0 + 2 || y > y0 + ph - 2) continue;
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x0 + pw, y);
    ctx.stroke();
    const lbl = Math.abs(v) >= 100 ? v.toFixed(0) : v.toPrecision(3);
    ctx.fillText(lbl, x0 - 4, y + 3);
  }

  ctx.textAlign = "center";
  const xs = niceStep(xRange, 6);
  for (let v = Math.ceil(xMin / xs) * xs; v <= xMax; v += xs) {
    const x = x0 + ((v - xMin) / xRange) * pw;
    ctx.beginPath();
    ctx.strokeStyle = PLOT_GRID;
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y0 + ph);
    ctx.stroke();
    ctx.fillStyle = PLOT_AXIS;
    ctx.fillText(v.toFixed(xs < 1 ? 2 : 0), x, y0 + ph + 12);
  }
  if (xLabel) {
    ctx.textAlign = "right";
    ctx.fillText(xLabel, x0 + pw, y0 + ph + 12);
  }
}

/**
 * Stroke a polyline of ``(xData[i], yData[i])`` samples into a plot panel.
 */
export function drawLine(
  ctx: CanvasRenderingContext2D,
  x0: number,
  y0: number,
  pw: number,
  ph: number,
  xData: number[],
  yData: number[],
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
  color: string,
  lineWidth = 1.2,
): void {
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  for (let i = 0; i < xData.length; i++) {
    const x = x0 + ((xData[i] - xMin) / xRange) * pw;
    const y = y0 + ph - ((yData[i] - yMin) / yRange) * ph;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}
