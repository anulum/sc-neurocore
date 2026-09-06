// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * The default view: voltage, drive and raster stacked on one time axis.
 *
 * This is the one a reader spends most of their time in, and the only view
 * that carries interaction state — a zoom window and a crosshair. Both are
 * passed in rather than read from a ref, which is what makes the drawing
 * checkable: a zoomed frame is a different argument, not a different session.
 */

import { at } from "../arrayAt";
import type { ImportedTrace, SimulateResponse } from "../api/client";
import {
  drawAxes,
  drawLine,
  niceStep,
  PLOT_AXIS as AXIS,
  PLOT_BORDER as BORDER,
  PLOT_COLORS as COLORS,
  PLOT_PANEL_BG as PANEL_BG,
} from "../simulationPlotCanvas";
import type { PlotFrame } from "./plotFrame";

/** The time window the trace view is showing. */
export interface TraceZoom {
  /** Left edge in milliseconds, or `NaN` for the whole run. */
  xMin: number;
  /** Right edge in milliseconds, or `NaN` for the whole run. */
  xMax: number;
}

/** Everything the trace view needs beyond the run itself. */
export interface TraceViewOptions {
  /** The time window to show. */
  zoom: TraceZoom;
  /** Where to rule the crosshair, in CSS pixels from the left, or `null`. */
  crosshair: number | null;
  /** A recorded trace to overlay, or `null`. */
  importedTrace: ImportedTrace | null;
}

/**
 * Draw the trace view.
 *
 * @param ctx - The context to draw into.
 * @param frame - Where the view may draw.
 * @param result - The run to draw.
 * @param options - Zoom, crosshair and any overlaid trace.
 */
export function drawTraceView(
  ctx: CanvasRenderingContext2D,
  frame: PlotFrame,
  result: SimulateResponse,
  options: TraceViewOptions,
): void {
  const { zoom, crosshair, importedTrace } = options;
  const time = result.time;
  const vars = Object.keys(result.states);
  const tMin = time[0] ?? 0;
  const tMax = time[time.length - 1] ?? tMin + 1;
  const hasSpikes = result.spikes.length > 0;
  // Default: Trace view (with nullcline overlay on phase + imported trace overlay)
  // Apply zoom viewport if set
  const zTMin = isNaN(zoom.xMin) ? tMin : zoom.xMin;
  const zTMax = isNaN(zoom.xMax) ? tMax : zoom.xMax;

  // Layout: voltage 65%, current 15%, raster 8%, x-labels
  const gap = 4;
  const rasterH = hasSpikes ? 22 : 0;
  const currentH = 40;
  const xLabelH = 16;
  const voltH = frame.height - frame.top - currentH - rasterH - gap * 2 - xLabelH;
  if (voltH < 30) return;

  // Compute Y range
  let vMin = Infinity, vMax = -Infinity;
  for (const v of vars) {
    for (const val of result.states[v] ?? []) {
      if (isFinite(val)) { if (val < vMin) vMin = val; if (val > vMax) vMax = val; }
    }
  }
  const vPad = (vMax - vMin) * 0.06 || 1;
  vMin -= vPad; vMax += vPad;

  // Voltage plot
  drawAxes(ctx, frame.left, frame.top, frame.plotWidth, voltH, zTMin, zTMax, vMin, vMax);
  vars.forEach((v, i) => {
    const trace = result.states[v] ?? [];
    drawLine(ctx, frame.left, frame.top, frame.plotWidth, voltH, time, trace, zTMin, zTMax, vMin, vMax,
      at(COLORS as readonly string[], i % COLORS.length));
  });
  // Y-axis label
  ctx.save();
  ctx.translate(10, frame.top + voltH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = AXIS; ctx.font = "9px monospace"; ctx.textAlign = "center";
  ctx.fillText("mV", 0, 0);
  ctx.restore();
  // Spike markers
  if (hasSpikes) {
    ctx.strokeStyle = "rgba(255,82,82,0.2)"; ctx.lineWidth = 1;
    for (const idx of result.spikes) {
      const x = frame.left + (((idx + 1) * result.dt - zTMin) / (zTMax - zTMin || 1)) * frame.plotWidth;
      ctx.beginPath(); ctx.moveTo(x, frame.top); ctx.lineTo(x, frame.top + voltH); ctx.stroke();
    }
  }
  // Legend
  ctx.font = "10px monospace";
  vars.forEach((v, i) => {
    const colour = at(COLORS as readonly string[], i % COLORS.length);
    ctx.fillStyle = colour;
    ctx.fillRect(frame.left + 6 + i * 52, frame.top + 4, 8, 2);
    ctx.textAlign = "left"; ctx.fillText(v, frame.left + 17 + i * 52, frame.top + 9);
  });

  // Imported trace overlay
  if (importedTrace) {
    ctx.setLineDash([4, 3]);
    drawLine(ctx, frame.left, frame.top, frame.plotWidth, voltH, importedTrace.time, importedTrace.voltage,
      zTMin, zTMax, vMin, vMax, "#ff9800", 1.5);
    ctx.setLineDash([]);
    ctx.fillStyle = "#ff9800"; ctx.font = "9px monospace"; ctx.textAlign = "left";
    ctx.fillText("imported", frame.left + 6 + vars.length * 52, frame.top + 9);
  }

  // Current plot
  const curY = frame.top + voltH + gap;
  const I = result.current_trace;
  let iMin = Math.min(...I), iMax = Math.max(...I);
  if (iMin === iMax) { iMin -= 1; iMax += 1; }
  drawAxes(ctx, frame.left, curY, frame.plotWidth, currentH, zTMin, zTMax, iMin, iMax * 1.1);
  drawLine(ctx, frame.left, curY, frame.plotWidth, currentH, time, I, zTMin, zTMax, iMin, iMax * 1.1, "#ffb74d", 1.5);
  ctx.fillStyle = "#ffb74d"; ctx.font = "10px monospace"; ctx.textAlign = "left";
  ctx.fillText("I", frame.left + 4, curY + 10);
  // Y-axis label for current
  ctx.save();
  ctx.translate(10, curY + currentH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = AXIS; ctx.font = "9px monospace"; ctx.textAlign = "center";
  ctx.fillText("nA", 0, 0);
  ctx.restore();

  // Spike raster
  if (hasSpikes) {
    const rasY = curY + currentH + gap;
    ctx.fillStyle = PANEL_BG; ctx.fillRect(frame.left, rasY, frame.plotWidth, rasterH);
    ctx.strokeStyle = BORDER; ctx.lineWidth = 1; ctx.strokeRect(frame.left, rasY, frame.plotWidth, rasterH);
    ctx.strokeStyle = "#ff5252"; ctx.lineWidth = 1.5;
    for (const idx of result.spikes) {
      const x = frame.left + (((idx + 1) * result.dt - zTMin) / (zTMax - zTMin || 1)) * frame.plotWidth;
      ctx.beginPath(); ctx.moveTo(x, rasY + 2); ctx.lineTo(x, rasY + rasterH - 2); ctx.stroke();
    }
  }

  // X-axis labels
  ctx.fillStyle = AXIS; ctx.font = "10px monospace"; ctx.textAlign = "center";
  const xs = niceStep(zTMax - zTMin, 6);
  for (let v = Math.ceil(zTMin / xs) * xs; v <= zTMax; v += xs) {
    const x = frame.left + ((v - zTMin) / (zTMax - zTMin || 1)) * frame.plotWidth;
    ctx.fillText(v.toFixed(0), x, frame.height - 2);
  }
  ctx.textAlign = "right"; ctx.fillText("ms", frame.left + frame.plotWidth, frame.height - 2);

  // Crosshair
  if (crosshair !== null) {
    const cx = crosshair;
    ctx.strokeStyle = "rgba(79,195,247,0.3)"; ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath(); ctx.moveTo(cx, frame.top); ctx.lineTo(cx, frame.height - 10); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Zoom indicator
  if (!isNaN(zoom.xMin)) {
    ctx.fillStyle = "#4fc3f7"; ctx.font = "9px monospace"; ctx.textAlign = "right";
    ctx.fillText(`zoom: ${zTMin.toFixed(1)}–${zTMax.toFixed(1)} ms (dbl-click to reset)`, frame.left + frame.plotWidth - 2, frame.height - 2);
  }
}
