// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * The views that draw more than one run at once, or one run against itself.
 *
 * Precision compares a float64 reference with a fixed-point candidate and the
 * error between them; compare puts two runs above one another; multi overlays
 * as many as were asked for. Each was a branch of one component's `draw`
 * callback and is a function here, so its geometry can be checked without a
 * browser.
 */

import { at } from "../arrayAt";
import type { CompareResponse, PrecisionResponse, SimulateResponse } from "../api/client";
import {
  drawAxes,
  drawLine,
  PLOT_AXIS as AXIS,
  PLOT_COLORS as COLORS,
} from "../simulationPlotCanvas";
import type { PlotFrame } from "./plotFrame";

/**
 * Draw the float64 reference, the fixed-point candidate, and their error.
 *
 * Two panels: the traces above, the absolute error below. Each result is drawn
 * on its own display sample times, because the two runs' projections are
 * chosen independently and plotting one on the other's clock would shift it.
 *
 * @param ctx - The context to draw into.
 * @param frame - Where the view may draw.
 * @param precResult - The comparison to draw.
 */
export function drawPrecisionView(ctx: CanvasRenderingContext2D, frame: PlotFrame, precResult: PrecisionResponse): void {
  const ph = (frame.height - frame.top - frame.bottom - 30) / 2;
  const variable = precResult.error.variable;
  const float_v = precResult.float_result.states[variable] ?? [];
  const fixed_v = precResult.fixed_result.states[variable] ?? [];
  const time_f = precResult.float_result.time;
  const time_x = precResult.fixed_result.time;
  const tMin = time_f[0] ?? 0;
  const tMax = time_f[time_f.length - 1] ?? tMin + 1;
  let vMin = Math.min(...float_v, ...fixed_v);
  let vMax = Math.max(...float_v, ...fixed_v);
  const vPad = (vMax - vMin) * 0.05 || 1;
  vMin -= vPad; vMax += vPad;
  const qLabel = precResult.arithmetic?.q_format ?? precResult.encoding?.q_format ?? "fixed-point";
  const arithmeticLabel = precResult.arithmetic
    ? `bit-true ${qLabel} kernel (${precResult.arithmetic.overflow}, ${precResult.arithmetic.rounding})`
    : qLabel;

  drawAxes(ctx, frame.left, frame.top, frame.plotWidth, ph, tMin, tMax, vMin, vMax);
  drawLine(ctx, frame.left, frame.top, frame.plotWidth, ph, time_f, float_v, tMin, tMax, vMin, vMax, "#4fc3f7", 1.2);
  // Each result is drawn on its own display sample times: the projections
  // of the two runs are chosen independently.
  drawLine(ctx, frame.left, frame.top, frame.plotWidth, ph, time_x, fixed_v, tMin, tMax, vMin, vMax, "#ff5252", 1.2);
  ctx.font = "10px monospace"; ctx.textAlign = "left";
  ctx.fillStyle = "#4fc3f7"; ctx.fillText("float64", frame.left + 6, frame.top + 12);
  ctx.fillStyle = "#ff5252"; ctx.fillText(arithmeticLabel, frame.left + 60, frame.top + 12);

  // Error trace at the float result's display samples (the raw error stays in error.trace).
  const errorSeries = precResult.error.display ?? precResult.error.trace;
  const errY = frame.top + ph + 16;
  const errH = ph - 8;
  const paramError = precResult.comparison?.parameter_quantisation.variables[variable];
  const errMax = Math.max(...errorSeries, paramError?.max_abs_error ?? 0, 0.001);
  drawAxes(ctx, frame.left, errY, frame.plotWidth, errH, tMin, tMax, 0, errMax * 1.1, "ms");
  if (paramError?.display.length === time_f.length) {
    drawLine(ctx, frame.left, errY, frame.plotWidth, errH, time_f, paramError.display, tMin, tMax, 0, errMax * 1.1, "#b39ddb", 1.0);
  }
  if (errorSeries.length === time_f.length) {
    drawLine(ctx, frame.left, errY, frame.plotWidth, errH, time_f, errorSeries, tMin, tMax, 0, errMax * 1.1, "#ffb74d", 1.5);
  }
  ctx.fillStyle = "#ffb74d"; ctx.font = "10px monospace"; ctx.textAlign = "left";
  const divergence = precResult.error.first_divergence_step;
  ctx.fillText(
    `|float64 − bit-true| (max=${precResult.error.max_error.toFixed(4)}, rms=${precResult.error.rms_error.toFixed(4)}`
      + (divergence === null || divergence === undefined ? ", never beyond ½ LSB)" : `, diverges at step ${divergence})`),
    frame.left + 6, errY + 12,
  );
  if (paramError) {
    ctx.fillStyle = "#b39ddb";
    ctx.fillText(
      `|float64 − quantised-parameter float64| (max=${paramError.max_abs_error.toFixed(4)}, rms=${paramError.rms_error.toFixed(4)})`,
      frame.left + 6, errY + 24,
    );
  }
  const events = precResult.comparison?.bit_true.events;
  if (events) {
    ctx.fillStyle = AXIS;
    ctx.fillText(
      events.identical
        ? `spikes identical (${events.reference_count})`
        : `spikes differ: float64 ${events.reference_count}, bit-true ${events.candidate_count}, first at #${events.first_divergence?.index ?? "?"}`,
      frame.left + 6, errY + 36,
    );
  }
}

/**
 * Draw two runs in stacked panels, each on its own scale.
 *
 * Each panel is scaled to its own run rather than to a shared range, because
 * the comparison being made is of shape, and a shared scale would flatten
 * whichever run has the smaller excursion.
 *
 * @param ctx - The context to draw into.
 * @param frame - Where the view may draw.
 * @param compareResult - The two runs to draw.
 */
export function drawCompareView(ctx: CanvasRenderingContext2D, frame: PlotFrame, compareResult: CompareResponse): void {
  const ph = (frame.height - frame.top - frame.bottom - 10) / 2;
  for (const [idx, label, res] of [[0, "A", compareResult.a], [1, "B", compareResult.b]] as const) {
    const yOff = frame.top + idx * (ph + 10);
    const v0 = Object.keys(res.states)[0];
    const data = v0 === undefined ? [] : res.states[v0] ?? [];
    const tm = res.time;
    let yMin = Math.min(...data), yMax = Math.max(...data);
    const yPad = (yMax - yMin) * 0.05 || 1;
    yMin -= yPad; yMax += yPad;
    const tStart = tm[0] ?? 0;
    const tEnd = tm[tm.length - 1] ?? tStart + 1;
    const colour = at(COLORS as readonly string[], idx);
    drawAxes(ctx, frame.left, yOff, frame.plotWidth, ph, tStart, tEnd, yMin, yMax);
    drawLine(ctx, frame.left, yOff, frame.plotWidth, ph, tm, data, tStart, tEnd, yMin, yMax, colour, 1.2);
    ctx.fillStyle = colour; ctx.font = "10px monospace"; ctx.textAlign = "left";
    // `||` and not `??`: a run with an empty model name is a custom system,
    // and `??` would label it with the empty string it actually carries.
    // eslint-disable-next-line @typescript-eslint/prefer-nullish-coalescing
    const name = res.model_name || "custom";
    ctx.fillText(`${label}: ${name} (${res.stats.rate_hz} Hz)`, frame.left + 6, yOff + 12);
  }
}

/**
 * Overlay several runs on one shared scale, with a legend.
 *
 * Here the scale *is* shared, and deliberately: the point of the overlay is to
 * compare magnitudes across models, which per-run scaling would destroy.
 *
 * @param ctx - The context to draw into.
 * @param frame - Where the view may draw.
 * @param multiResults - The runs to overlay, in the order to draw them.
 */
export function drawMultiModelView(ctx: CanvasRenderingContext2D, frame: PlotFrame, multiResults: SimulateResponse[]): void {
  if (multiResults.length === 0) return;
  const ph = frame.height - frame.top - frame.bottom;
  let tMin = Infinity, tMax = -Infinity, vMin = Infinity, vMax = -Infinity;
  for (const r of multiResults) {
    const start = r.time[0];
    const end = r.time[r.time.length - 1];
    if (start !== undefined && start < tMin) tMin = start;
    if (end !== undefined && end > tMax) tMax = end;
    const v0 = Object.keys(r.states)[0];
    for (const v of (v0 === undefined ? [] : r.states[v0] ?? [])) {
      if (isFinite(v)) { if (v < vMin) vMin = v; if (v > vMax) vMax = v; }
    }
  }
  const vPad = (vMax - vMin) * 0.06 || 1;
  vMin -= vPad; vMax += vPad;
  drawAxes(ctx, frame.left, frame.top, frame.plotWidth, ph, tMin, tMax, vMin, vMax, "ms");
  multiResults.forEach((r, i) => {
    const v0 = Object.keys(r.states)[0];
    const trace = v0 === undefined ? [] : r.states[v0] ?? [];
    const colour = at(COLORS as readonly string[], i % COLORS.length);
    drawLine(ctx, frame.left, frame.top, frame.plotWidth, ph, r.time, trace, tMin, tMax, vMin, vMax, colour, 1.5);
  });
  ctx.font = "10px monospace";
  multiResults.forEach((r, i) => {
    // `||` and not `??`, for the same reason as in the compare view: an
    // empty name is not a name, and numbering it is the point.
    // eslint-disable-next-line @typescript-eslint/prefer-nullish-coalescing
    const name = r.model_name || `Model ${String(i + 1)}`;
    const colour = at(COLORS as readonly string[], i % COLORS.length);
    ctx.fillStyle = colour;
    ctx.fillRect(frame.left + 6 + i * 120, frame.top + 4, 8, 2);
    ctx.textAlign = "left";
    ctx.fillText(`${name} (${r.stats.rate_hz}Hz)`, frame.left + 17 + i * 120, frame.top + 9);
  });
}
