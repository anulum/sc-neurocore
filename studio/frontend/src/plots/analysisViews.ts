// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * The single-series analysis views, as functions rather than branches.
 *
 * Each takes a context, a frame and the one result it draws, and returns
 * nothing. That shape is the point: these were branches inside an 836-line
 * component's `draw` callback, reachable only by mounting the component,
 * choosing a tab and giving the store a result. As functions they are called
 * directly, which is why they now have cases.
 */

import { at } from "../arrayAt";
import type {
  BifurcationResponse,
  FICurveResponse,
  FreqResponse,
  HeatmapResponse,
  SensitivityResponse,
  SimulateResponse,
} from "../api/client";
import {
  drawAxes,
  drawLine,
  PLOT_AXIS as AXIS,
} from "../simulationPlotCanvas";
import type { PlotFrame } from "./plotFrame";

/** A spike-triggered average, as the store holds it. */
export interface SpikeTriggeredAverage {
  /** Times relative to the spike, in milliseconds. */
  time_ms: number[];
  /** The averaged trace at those times. */
  average: number[];
  /** How many spikes were averaged. */
  n_spikes: number;
}

/**
 * Draw firing rate against injected current.
 *
 * @param ctx - The context to draw into.
 * @param frame - Where the view may draw.
 * @param fiResult - The measured curve.
 */
export function drawFICurveView(ctx: CanvasRenderingContext2D, frame: PlotFrame, fiResult: FICurveResponse): void {
  const ph = frame.height - frame.top - frame.bottom;
  const xMin = fiResult.currents[0] ?? 0;
  const xMax = fiResult.currents[fiResult.currents.length - 1] ?? xMin + 1;
  const yMax = Math.max(...fiResult.rates, 1);
  drawAxes(ctx, frame.left, frame.top, frame.plotWidth, ph, xMin, xMax, 0, yMax * 1.1, "I (nA)");
  drawLine(ctx, frame.left, frame.top, frame.plotWidth, ph, fiResult.currents, fiResult.rates, xMin, xMax, 0, yMax * 1.1, "#4fc3f7", 2);
  ctx.fillStyle = AXIS;
  ctx.font = "10px monospace";
  ctx.textAlign = "left";
  ctx.fillText("f (Hz)", frame.left + 4, frame.top + 12);
}

/**
 * Draw the inter-spike-interval histogram.
 *
 * Does nothing when the run carries no histogram, which is the case for a run
 * with too few spikes to form one.
 *
 * @param ctx - The context to draw into.
 * @param frame - Where the view may draw.
 * @param result - The run whose histogram to draw.
 */
export function drawIsiHistogramView(ctx: CanvasRenderingContext2D, frame: PlotFrame, result: SimulateResponse): void {
  if (!result.stats.isi_histogram) return;
  const ph = frame.height - frame.top - frame.bottom;
  const hist = result.stats.isi_histogram;
  const maxCount = Math.max(...hist.counts, 1);
  const xMin = hist.edges[0] ?? 0;
  const xMax = hist.edges[hist.edges.length - 1] ?? xMin + 1;

  drawAxes(ctx, frame.left, frame.top, frame.plotWidth, ph, xMin, xMax, 0, maxCount * 1.1, "ISI (ms)");
  ctx.fillStyle = "rgba(79, 195, 247, 0.6)";
  const xRange = xMax - xMin || 1;
  for (const [i, count] of hist.counts.entries()) {
    const edge = at(hist.edges, i);
    const bx = frame.left + ((edge - xMin) / xRange) * frame.plotWidth;
    // The last bin has no right edge when the server sent one edge per
    // count instead of one more; falling back to the bin's own left edge
    // draws it with zero width rather than a NaN rectangle.
    const bw = (((hist.edges[i + 1] ?? edge) - edge) / xRange) * frame.plotWidth;
    const bh = (count / (maxCount * 1.1)) * ph;
    ctx.fillRect(bx, frame.top + ph - bh, Math.max(bw - 1, 1), bh);
  }
  ctx.fillStyle = AXIS; ctx.font = "10px monospace"; ctx.textAlign = "left";
  ctx.fillText("count", frame.left + 4, frame.top + 12);
}

/**
 * Draw the attractor sweep.
 *
 * @param ctx - The context to draw into.
 * @param frame - Where the view may draw.
 * @param bifResult - The sweep to draw.
 */
export function drawBifurcationView(ctx: CanvasRenderingContext2D, frame: PlotFrame, bifResult: BifurcationResponse): void {
  const ph = frame.height - frame.top - frame.bottom;
  const { param_values, attractors } = bifResult;
  const xMin = param_values[0] ?? 0;
  const xMax = param_values[param_values.length - 1] ?? xMin + 1;
  let yMin = Infinity, yMax = -Infinity;
  for (const a of attractors) for (const v of a) { if (v < yMin) yMin = v; if (v > yMax) yMax = v; }
  if (!isFinite(yMin)) { yMin = -80; yMax = 40; }
  const yPad = (yMax - yMin) * 0.05 || 1;
  yMin -= yPad; yMax += yPad;
  drawAxes(ctx, frame.left, frame.top, frame.plotWidth, ph, xMin, xMax, yMin, yMax, bifResult.param_name);
  ctx.fillStyle = "rgba(79,195,247,0.5)";
  for (const [i, paramValue] of param_values.entries()) {
    const x = frame.left + ((paramValue - xMin) / (xMax - xMin || 1)) * frame.plotWidth;
    for (const v of attractors[i] ?? []) {
      const y = frame.top + ph - ((v - yMin) / (yMax - yMin)) * ph;
      ctx.fillRect(x - 1, y - 1, 2, 2);
    }
  }
  ctx.fillStyle = AXIS; ctx.font = "10px monospace"; ctx.textAlign = "left";
  ctx.fillText("V attractor", frame.left + 4, frame.top + 12);
}

/**
 * Draw the two-parameter rate heatmap.
 *
 * @param ctx - The context to draw into.
 * @param frame - Where the view may draw.
 * @param heatmapResult - The grid to draw.
 */
export function drawHeatmapView(ctx: CanvasRenderingContext2D, frame: PlotFrame, heatmapResult: HeatmapResponse): void {
  const ph = frame.height - frame.top - frame.bottom - 16;
  const { x_values, y_values, rates, rate_min, rate_max } = heatmapResult;
  const xMin = x_values[0] ?? 0;
  const xMax = x_values[x_values.length - 1] ?? xMin + 1;
  const yMin = y_values[0] ?? 0;
  const yMax = y_values[y_values.length - 1] ?? yMin + 1;
  const rRange = rate_max - rate_min || 1;

  drawAxes(ctx, frame.left, frame.top, frame.plotWidth, ph, xMin, xMax, yMin, yMax, heatmapResult.param_x);
  const cellW = frame.plotWidth / x_values.length;
  const cellH = ph / y_values.length;
  for (let j = 0; j < y_values.length; j++) {
    const row = rates[j] ?? [];
    for (let i = 0; i < x_values.length; i++) {
      const norm = ((row[i] ?? rate_min) - rate_min) / rRange;
      const r = Math.floor(norm * 200 + 20);
      const g = Math.floor(norm * 50);
      const b = Math.floor((1 - norm) * 200 + 55);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      const cx = frame.left + (i / x_values.length) * frame.plotWidth;
      const cy = frame.top + ph - ((j + 1) / y_values.length) * ph;
      ctx.fillRect(cx, cy, cellW + 1, cellH + 1);
    }
  }
  ctx.fillStyle = AXIS; ctx.font = "10px monospace"; ctx.textAlign = "left";
  ctx.fillText(`${heatmapResult.param_y} vs ${heatmapResult.param_x}  (${rate_min.toFixed(0)}–${rate_max.toFixed(0)} Hz)`, frame.left + 4, frame.top + 12);
}

/**
 * Draw parameter elasticities as bars, at most fifteen of them.
 *
 * A parameter whose elasticity is undefined gets its reason written where its
 * bar would be, rather than a zero-length bar that would read as insensitive.
 *
 * @param ctx - The context to draw into.
 * @param frame - Where the view may draw.
 * @param sensResult - The elasticities to draw.
 */
export function drawSensitivityView(ctx: CanvasRenderingContext2D, frame: PlotFrame, sensResult: SensitivityResponse): void {
  const ph = frame.height - frame.top - frame.bottom - 20;
  const sens = sensResult.sensitivities.slice(0, 15);
  if (sens.length === 0) return;
  const defined = sens.map((s) => s.sensitivity).filter((v): v is number => v !== null);
  const maxS = Math.max(...defined, 0.01);
  const barH = Math.min(20, ph / sens.length - 2);
  ctx.font = "10px monospace";
  sens.forEach((s, i) => {
    const y = frame.top + i * (barH + 2);
    ctx.fillStyle = AXIS; ctx.textAlign = "right";
    ctx.fillText(s.param, frame.left + 65, y + barH - 4);
    ctx.textAlign = "left";
    if (s.sensitivity === null) {
      // Undefined elasticity (zero base rate or zero parameter): no bar, the reason instead of a zero.
      ctx.fillStyle = "#ffb74d";
      ctx.fillText(`undefined: ${s.reason ?? "no reason given"}`, frame.left + 75, y + barH - 4);
      return;
    }
    const bw = (s.sensitivity / maxS) * (frame.plotWidth - 80);
    ctx.fillStyle = "rgba(79,195,247,0.6)";
    ctx.fillRect(frame.left + 70, y, bw, barH);
    ctx.fillStyle = AXIS;
    ctx.fillText(s.sensitivity.toFixed(3), frame.left + 75 + bw, y + barH - 4);
  });
  ctx.fillStyle = AXIS; ctx.textAlign = "left";
  ctx.fillText(`base rate: ${sensResult.base_rate} Hz (elasticity |Δrate/Δp|·|p|/rate)`, frame.left + 4, frame.height - 8);
}

/**
 * Draw firing rate against input frequency.
 *
 * @param ctx - The context to draw into.
 * @param frame - Where the view may draw.
 * @param freqResult - The sweep to draw.
 */
export function drawFrequencyResponseView(ctx: CanvasRenderingContext2D, frame: PlotFrame, freqResult: FreqResponse): void {
  const ph = frame.height - frame.top - frame.bottom;
  const xMin = freqResult.frequencies_hz[0] ?? 0;
  const xMax = freqResult.frequencies_hz[freqResult.frequencies_hz.length - 1] ?? xMin + 1;
  const yMax = Math.max(...freqResult.rates, 1);
  drawAxes(ctx, frame.left, frame.top, frame.plotWidth, ph, xMin, xMax, 0, yMax * 1.1, "freq (Hz)");
  drawLine(ctx, frame.left, frame.top, frame.plotWidth, ph, freqResult.frequencies_hz, freqResult.rates,
    xMin, xMax, 0, yMax * 1.1, "#4fc3f7", 2);
  ctx.fillStyle = AXIS; ctx.font = "10px monospace"; ctx.textAlign = "left";
  ctx.fillText(`rate (Hz) @ amplitude=${freqResult.amplitude}`, frame.left + 4, frame.top + 12);
}

/**
 * Draw the spike-triggered average, with a rule at the spike itself.
 *
 * Does nothing for an empty average, which is what an analysis of too few
 * spikes produces.
 *
 * @param ctx - The context to draw into.
 * @param frame - Where the view may draw.
 * @param staResult - The average to draw.
 */
export function drawSpikeTriggeredAverageView(ctx: CanvasRenderingContext2D, frame: PlotFrame, staResult: SpikeTriggeredAverage): void {
  if (staResult.time_ms.length === 0) return;
  const ph = frame.height - frame.top - frame.bottom;
  const xMin = at(staResult.time_ms, 0);
  const xMax = at(staResult.time_ms, staResult.time_ms.length - 1);
  let yMin = Math.min(...staResult.average), yMax = Math.max(...staResult.average);
  const yPad = (yMax - yMin) * 0.05 || 1;
  yMin -= yPad; yMax += yPad;
  drawAxes(ctx, frame.left, frame.top, frame.plotWidth, ph, xMin, xMax, yMin, yMax, "ms (relative to spike)");
  drawLine(ctx, frame.left, frame.top, frame.plotWidth, ph, staResult.time_ms, staResult.average, xMin, xMax, yMin, yMax, "#4fc3f7", 2);
  // Vertical line at t=0
  const x0 = frame.left + ((0 - xMin) / (xMax - xMin)) * frame.plotWidth;
  ctx.strokeStyle = "#ff5252"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
  ctx.beginPath(); ctx.moveTo(x0, frame.top); ctx.lineTo(x0, frame.top + ph); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = AXIS; ctx.font = "10px monospace"; ctx.textAlign = "left";
  ctx.fillText(`STA (n=${staResult.n_spikes} spikes)`, frame.left + 4, frame.top + 12);
}
