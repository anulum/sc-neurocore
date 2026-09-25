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

import { at } from "./arrayAt";

/** The series colours, in the order traces are assigned them. */
export const PLOT_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373", "#ce93d8", "#90a4ae"] as const;
/** The ground behind a whole plot. */
export const PLOT_BG = "#0d1117";
/** The ground inside a plot panel, one step darker than the plot's. */
export const PLOT_PANEL_BG = "#0a0e14";
/**
 * The gridlines. Deliberately below the 3:1 contrast floor: a gridline is not
 * information a reader needs to identify, and one at 3:1 competes with the
 * trace drawn over it.
 */
export const PLOT_GRID = "#1a1f2a";
/**
 * Axis rule and tick-label colour for every Studio plot canvas.
 *
 * Canvas text is pixels, so no DOM audit can measure it. The value is the
 * lowest luminance of the original blue-grey that still reaches 4.5:1 against
 * both plot grounds (4.62:1 on PLOT_PANEL_BG, 4.53:1 on PLOT_BG); the same
 * colour draws the axis rules, which need only 3:1. Guarded by
 * paletteContrast.test.ts.
 */
export const PLOT_AXIS = "#727d8b";
/** The rule around a plot panel. */
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
 * Paint a plot panel: its ground, its border, its grid and its tick labels.
 *
 * Tick labels are placed on a 1-2-5 ladder rather than at fixed intervals, so
 * an axis reads in round numbers whatever range it spans. A label that would
 * land within two pixels of the panel edge is skipped: a half-clipped number
 * is worse than an unlabelled gridline.
 *
 * @param ctx - The canvas to draw on.
 * @param x0 - The panel's left edge, in canvas pixels.
 * @param y0 - The panel's top edge, in canvas pixels.
 * @param pw - The panel's width.
 * @param ph - The panel's height.
 * @param xMin - The lowest value on the horizontal axis.
 * @param xMax - The highest value on the horizontal axis.
 * @param yMin - The lowest value on the vertical axis.
 * @param yMax - The highest value on the vertical axis.
 * @param xLabel - A unit or name to print at the right of the horizontal axis.
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
 * Choose which samples of a series to stroke across `columns` pixel columns.
 *
 * A series with more samples than a panel has pixels is drawn as, per column,
 * its first, lowest, highest and last finite sample, in sample order, plus the
 * nearest sample on each side of the visible window so the line enters and
 * leaves the panel where it should. Every extreme a reader could see survives
 * and the result data are untouched; only the path handed to the canvas is
 * shorter. A series whose horizontal values are not non-decreasing (a phase
 * portrait) or that is short enough is drawn in full.
 *
 * @param xData - The samples' horizontal values.
 * @param yData - The samples' vertical values, paired with `xData`.
 * @param xMin - The lowest visible horizontal value.
 * @param xMax - The highest visible horizontal value.
 * @param columns - The panel's width in pixel columns.
 * @returns The indices to stroke, ascending.
 */
export function displayIndices(
  xData: readonly number[],
  yData: readonly number[],
  xMin: number,
  xMax: number,
  columns: number,
): number[] {
  const paired = Math.min(xData.length, yData.length);
  const width = Math.max(1, Math.floor(columns));
  const all = (): number[] => Array.from({ length: paired }, (_, index) => index);
  if (paired <= 4 * width) return all();
  for (let i = 1; i < paired; i++) {
    if (!(at(xData, i) >= at(xData, i - 1))) return all();
  }
  let first = 0;
  while (first < paired - 1 && at(xData, first + 1) < xMin) first++;
  let last = paired - 1;
  while (last > first && at(xData, last - 1) > xMax) last--;
  const range = xMax - xMin || 1;
  const kept = new Set<number>([first, last]);
  let column = -1;
  let low = -1;
  let high = -1;
  let end = -1;
  const flush = () => {
    for (const index of [low, high, end]) if (index >= 0) kept.add(index);
  };
  for (let i = first; i <= last; i++) {
    const y = at(yData, i);
    if (!Number.isFinite(y)) continue;
    const here = Math.min(width - 1, Math.max(0, Math.floor(((at(xData, i) - xMin) / range) * width)));
    if (here !== column) {
      flush();
      column = here;
      kept.add(i);
      low = high = end = i;
      continue;
    }
    if (y < at(yData, low)) low = i;
    if (y > at(yData, high)) high = i;
    end = i;
  }
  flush();
  return [...kept].sort((a, b) => a - b);
}

/**
 * Stroke a polyline of ``(xData[i], yData[i])`` samples into a plot panel.
 *
 * The two lists are paired samples of one trace and callers pass them from the
 * same response, so a length mismatch means the response itself is malformed.
 * This draws the paired prefix rather than throwing: a render path that fails
 * takes the whole plot down for one bad trace, and the previous behaviour —
 * reading past the end and stroking `NaN` coordinates — was worse than either.
 * Surfacing a malformed response to the reader belongs to whoever validates
 * responses, not to a drawing primitive. A series with more samples than the
 * panel has pixels is stroked through {@link displayIndices}.
 *
 * @param ctx - The canvas to draw on.
 * @param x0 - The panel's left edge, in canvas pixels.
 * @param y0 - The panel's top edge, in canvas pixels.
 * @param pw - The panel's width.
 * @param ph - The panel's height.
 * @param xData - The samples' horizontal values.
 * @param yData - The samples' vertical values, paired with `xData`.
 * @param xMin - The lowest value on the horizontal axis.
 * @param xMax - The highest value on the horizontal axis.
 * @param yMin - The lowest value on the vertical axis.
 * @param yMax - The highest value on the vertical axis.
 * @param color - The stroke colour.
 * @param lineWidth - The stroke width.
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
  displayIndices(xData, yData, xMin, xMax, pw).forEach((i, order) => {
    const x = x0 + ((at(xData, i) - xMin) / xRange) * pw;
    const y = y0 + ph - ((at(yData, i) - yMin) / yRange) * ph;
    if (order === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}
