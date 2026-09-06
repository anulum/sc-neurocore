// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * The views that draw state space, a model's character, or a whole network.
 *
 * These three share nothing with each other beyond being the views that are
 * not a series against time. Each was a branch of one component's `draw`
 * callback and is a function here.
 */

import { at } from "../arrayAt";
import type {
  CharacterizeResponse,
  NetworkResult,
  NullclineResponse,
  SimulateResponse,
} from "../api/client";
import {
  drawAxes,
  drawLine,
  PLOT_AXIS as AXIS,
  PLOT_BORDER as BORDER,
  PLOT_PANEL_BG as PANEL_BG,
} from "../simulationPlotCanvas";
import type { PlotFrame } from "./plotFrame";

/**
 * Draw the trajectory in state space, with the nullclines over it if there are any.
 *
 * The trajectory fades from its start to its end, so a closed orbit can be
 * told from a spiral. Nullcline points are drawn only where the server said
 * the component could be evaluated: a curve drawn through the rest would run
 * confidently through regions where nothing was computed.
 *
 * @param ctx - The context to draw into.
 * @param frame - Where the view may draw.
 * @param result - The run whose state to draw.
 * @param nullclineResult - The nullclines to overlay, or `null` for none.
 */
export function drawPhasePortraitView(ctx: CanvasRenderingContext2D,
  frame: PlotFrame,
  result: SimulateResponse,
  nullclineResult: NullclineResponse | null,): void {
  const vars = Object.keys(result.states);
  if (vars.length < 2) return;
  const ph = frame.height - frame.top - frame.bottom;
  const xData = result.states[at(vars, 0)] ?? [];
  const yData = result.states[at(vars, 1)] ?? [];
  let xMin = Math.min(...xData), xMax = Math.max(...xData);
  let yMin = Math.min(...yData), yMax = Math.max(...yData);
  const xPad = (xMax - xMin) * 0.05 || 1;
  const yPad = (yMax - yMin) * 0.05 || 1;
  xMin -= xPad; xMax += xPad; yMin -= yPad; yMax += yPad;

  drawAxes(ctx, frame.left, frame.top, frame.plotWidth, ph, xMin, xMax, yMin, yMax, vars[0]);
  // Draw trajectory with fading colour
  for (let i = 1; i < xData.length; i++) {
    const alpha = 0.15 + 0.85 * (i / xData.length);
    ctx.strokeStyle = `rgba(79, 195, 247, ${alpha})`;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(
      frame.left + ((at(xData, i - 1) - xMin) / (xMax - xMin)) * frame.plotWidth,
      frame.top + ph - ((at(yData, i - 1) - yMin) / (yMax - yMin)) * ph
    );
    ctx.lineTo(
      frame.left + ((at(xData, i) - xMin) / (xMax - xMin)) * frame.plotWidth,
      frame.top + ph - ((at(yData, i) - yMin) / (yMax - yMin)) * ph
    );
    ctx.stroke();
  }
  // Start and end markers
  const sx = frame.left + ((at(xData, 0) - xMin) / (xMax - xMin)) * frame.plotWidth;
  const sy = frame.top + ph - ((at(yData, 0) - yMin) / (yMax - yMin)) * ph;
  ctx.fillStyle = "#81c784";
  ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = AXIS; ctx.font = "10px monospace"; ctx.textAlign = "left";
  ctx.fillText(at(vars, 1), frame.left + 4, frame.top + 12);

  // Nullcline overlay
  if (nullclineResult) {
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    for (const [nc, color] of [
      [nullclineResult.nullcline_0, "#ff5252"],
      [nullclineResult.nullcline_1, "#81c784"],
    ] as const) {
      ctx.fillStyle = color;
      for (const point of nc.points) {
        const px = at(point, 0);
        const py = at(point, 1);
        const cx = frame.left + ((px - xMin) / xRange) * frame.plotWidth;
        const cy = frame.top + ph - ((py - yMin) / yRange) * ph;
        if (cx >= frame.left && cx <= frame.left + frame.plotWidth && cy >= frame.top && cy <= frame.top + ph) {
          ctx.fillRect(cx - 1, cy - 1, 2, 2);
        }
      }
    }
    ctx.font = "9px monospace"; ctx.textAlign = "right";
    ctx.fillStyle = "#ff5252"; ctx.fillText(`d${vars[0]}/dt=0`, frame.left + frame.plotWidth - 4, frame.top + ph - 16);
    ctx.fillStyle = "#81c784"; ctx.fillText(`d${vars[1]}/dt=0`, frame.left + frame.plotWidth - 4, frame.top + ph - 4);
    // Invalid part of the field (domain errors, overflow, non-finite values): not zero.
    const domain = nullclineResult.domain;
    if (domain && domain.status !== "complete") {
      const fractions = Object.entries(domain.invalid_fraction)
        .map(([name, fraction]) => `${name} ${(fraction * 100).toFixed(0)}%`)
        .join(", ");
      ctx.fillStyle = "#ffb74d"; ctx.textAlign = "left";
      ctx.fillText(
        domain.status === "empty"
          ? `field undefined on the whole grid (${fractions} invalid)`
          : `partial domain: ${fractions} of samples invalid, no contour there`,
        frame.left + 4, frame.top + 24,
      );
    }
  }
}

/**
 * Draw a model's character: its state ranges beside its f-I curve.
 *
 * @param ctx - The context to draw into.
 * @param frame - Where the view may draw.
 * @param charResult - The characterisation to draw.
 */
export function drawCharacterizeView(ctx: CanvasRenderingContext2D, frame: PlotFrame, charResult: CharacterizeResponse): void {
  ctx.fillStyle = "#e6edf3"; ctx.font = "12px sans-serif"; ctx.textAlign = "left";
  let y = frame.top + 16;
  const lineH = 18;
  const col1 = frame.left, col2 = frame.left + frame.plotWidth / 2;

  ctx.fillStyle = "#4fc3f7"; ctx.font = "bold 13px sans-serif";
  ctx.fillText("Model Characterisation", col1, y); y += lineH + 4;

  ctx.font = "11px monospace"; ctx.fillStyle = "#e6edf3";
  ctx.fillText(`Pattern: ${charResult.pattern.description}`, col1, y); y += lineH;
  ctx.fillText(`Threshold current: ${charResult.threshold_current ?? "N/A"} nA`, col1, y); y += lineH;
  ctx.fillText(`Max firing rate: ${charResult.max_rate} Hz`, col1, y); y += lineH;
  ctx.fillText(`Spikes: ${charResult.spike_count}`, col1, y); y += lineH;
  if (charResult.stats.isi_mean_ms) {
    ctx.fillText(`ISI: ${charResult.stats.isi_mean_ms} ms (CV=${charResult.stats.isi_cv})`, col1, y); y += lineH;
  }

  y += 8;
  ctx.fillStyle = "#4fc3f7"; ctx.font = "bold 11px sans-serif";
  ctx.fillText("State Variable Ranges", col1, y); y += lineH;
  ctx.font = "10px monospace"; ctx.fillStyle = "#8b949e";
  for (const [v, r] of Object.entries(charResult.state_ranges)) {
    ctx.fillText(`${v}: [${r.min}, ${r.max}] mean=${r.mean}`, col1, y); y += lineH - 2;
  }

  y += 8;
  ctx.fillStyle = "#4fc3f7"; ctx.font = "bold 11px sans-serif";
  ctx.fillText("Top Sensitive Parameters", col1, y); y += lineH;
  ctx.font = "10px monospace"; ctx.fillStyle = "#8b949e";
  for (const s of charResult.top_sensitivities) {
    ctx.fillText(`${s.param}: ±${s.rate_change} Hz`, col1, y); y += lineH - 2;
  }

  // f-I curve in right half
  const fiX = col2, fiY = frame.top + 20, fiW = frame.plotWidth / 2 - 20, fiH = frame.height - frame.top - frame.bottom - 40;
  const curs = charResult.fi_curve.currents;
  const rts = charResult.fi_curve.rates;
  const rMax = Math.max(...rts, 1);
  const curMin = curs[0] ?? 0;
  const curMax = curs[curs.length - 1] ?? curMin + 1;
  drawAxes(ctx, fiX, fiY, fiW, fiH, curMin, curMax, 0, rMax * 1.1, "I (nA)");
  drawLine(ctx, fiX, fiY, fiW, fiH, curs, rts, curMin, curMax, 0, rMax * 1.1, "#4fc3f7", 2);
  ctx.fillStyle = "#4fc3f7"; ctx.font = "10px monospace"; ctx.textAlign = "left";
  ctx.fillText("f-I curve", fiX + 4, fiY + 12);
}

/**
 * Draw a network run: the spike raster above, the population rates below.
 *
 * Excitatory and inhibitory spikes take different colours, and the split is by
 * neuron index against the excitatory count rather than by anything stored per
 * spike, because the raster carries one array of indices and not two.
 *
 * @param ctx - The context to draw into.
 * @param frame - Where the view may draw.
 * @param networkResult - The run to draw.
 */
export function drawNetworkView(ctx: CanvasRenderingContext2D, frame: PlotFrame, networkResult: NetworkResult): void {
  const rasterH = Math.floor((frame.height - frame.top - frame.bottom) * 0.6);
  const rateH = frame.height - frame.top - frame.bottom - rasterH - 10;

  // Raster plot
  ctx.fillStyle = PANEL_BG; ctx.fillRect(frame.left, frame.top, frame.plotWidth, rasterH);
  ctx.strokeStyle = BORDER; ctx.strokeRect(frame.left, frame.top, frame.plotWidth, rasterH);
  const dur = networkResult.duration;
  for (const [i, t] of networkResult.spike_times.entries()) {
    const n = at(networkResult.spike_neurons, i);
    const x = frame.left + (t / dur) * frame.plotWidth;
    const y = frame.top + (n / networkResult.n_total) * rasterH;
    ctx.fillStyle = n < networkResult.n_exc ? "#4fc3f7" : "#ff5252";
    ctx.fillRect(x, y, 1.5, 1.5);
  }
  ctx.fillStyle = "#4fc3f7"; ctx.font = "9px monospace"; ctx.textAlign = "left";
  ctx.fillText(`E (${networkResult.n_exc})`, frame.left + 4, frame.top + 10);
  ctx.fillStyle = "#ff5252";
  ctx.fillText(`I (${networkResult.n_inh})`, frame.left + 60, frame.top + 10);
  ctx.fillStyle = AXIS;
  ctx.fillText(`${networkResult.n_spikes} spikes`, frame.left + 120, frame.top + 10);

  // Population rates
  const rateY = frame.top + rasterH + 10;
  const rt = networkResult.rate_time;
  if (rt.length > 1) {
    const rMax = Math.max(...networkResult.exc_rates, ...networkResult.inh_rates, 1);
    const rtMin = at(rt, 0);
    const rtMax = at(rt, rt.length - 1);
    drawAxes(ctx, frame.left, rateY, frame.plotWidth, rateH, rtMin, rtMax, 0, rMax * 1.1, "ms");
    drawLine(ctx, frame.left, rateY, frame.plotWidth, rateH, rt, networkResult.exc_rates, rtMin, rtMax, 0, rMax * 1.1, "#4fc3f7", 1.5);
    drawLine(ctx, frame.left, rateY, frame.plotWidth, rateH, rt, networkResult.inh_rates, rtMin, rtMax, 0, rMax * 1.1, "#ff5252", 1.5);
    ctx.fillStyle = AXIS; ctx.font = "9px monospace"; ctx.textAlign = "left";
    ctx.fillText(`E: ${networkResult.mean_exc_rate}Hz  I: ${networkResult.mean_inh_rate}Hz`, frame.left + 4, rateY + 10);
  }
}
