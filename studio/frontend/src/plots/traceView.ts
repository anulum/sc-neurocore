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

/** What the drive panel of the trace is labelled with. */
export const CURRENT_PANEL_LABEL = "I (model's own units)";

/** The time window the trace view is showing. */
export interface TraceZoom {
  /** Left edge in milliseconds, or `NaN` for the whole run. */
  xMin: number;
  /** Right edge in milliseconds, or `NaN` for the whole run. */
  xMax: number;
}

/** How many times wider or narrower a state may swing and still share the first state's axis. */
export const SHARED_AXIS_SPAN_RATIO = 10;

/** The states drawn on each of the trace's axes. */
export interface TraceAxisGroups {
  /** The first state and those on its scale. */
  primary: string[];
  /** States on a scale of their own, drawn in a second panel. */
  secondary: string[];
}

/**
 * Split a run's states between the first state's axis and a second one.
 *
 * Every state shared one axis, labelled mV: a conductance model's gating
 * variables (0 to 1) lay flat along its zero line under a voltage swinging
 * over 100 mV, and could not be read at all. A state whose range is more than
 * {@link SHARED_AXIS_SPAN_RATIO} times wider or narrower than the first
 * state's gets the second panel.
 *
 * @param states - The run's states, the first one first.
 * @returns The two groups; `secondary` is empty when every state fits.
 */
export function traceAxisGroups(states: Record<string, readonly number[]>): TraceAxisGroups {
  const names = Object.keys(states);
  const span = (name: string): number => {
    let lo = Infinity, hi = -Infinity;
    for (const value of states[name] ?? []) {
      if (Number.isFinite(value)) { if (value < lo) lo = value; if (value > hi) hi = value; }
    }
    return hi > lo ? hi - lo : 0;
  };
  const [first, ...rest] = names;
  if (first === undefined) return { primary: [], secondary: [] };
  const reference = span(first);
  const primary = [first];
  const secondary: string[] = [];
  for (const name of rest) {
    const own = span(name);
    const apart = reference > 0 && own > 0
      ? Math.max(own / reference, reference / own) > SHARED_AXIS_SPAN_RATIO
      : reference > 0 !== own > 0;
    (apart ? secondary : primary).push(name);
  }
  return { primary, secondary };
}

/**
 * Label an axis with its states' declared unit, or with their names.
 *
 * The axis said mV for every model; no catalogue model declares its state
 * units to the Studio yet, and a map model's state has none.
 *
 * @param names - The states on the axis.
 * @param units - Each state's declared unit, empty when undeclared.
 * @returns The label.
 */
export function traceAxisLabel(names: readonly string[], units: Readonly<Record<string, string>>): string {
  const declared = new Set(names.map((name) => units[name] ?? ""));
  const [only] = [...declared];
  return declared.size === 1 && only !== undefined && only !== "" ? only : names.join(", ");
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

  const groups = traceAxisGroups(result.states);
  const units: Record<string, string> = {};
  for (const variable of result.state_layout?.variables ?? []) units[variable.name] = variable.unit;
  const secondH = groups.secondary.length > 0 ? Math.round(voltH * 0.35) : 0;
  const firstH = groups.secondary.length > 0 ? voltH - secondH - gap : voltH;

  /**
   * Draw one panel of states on its own axis.
   *
   * @param names - The states.
   * @param top - The panel's top edge.
   * @param height - The panel's height.
   */
  const drawStatePanel = (names: readonly string[], top: number, height: number): void => {
    let lo = Infinity, hi = -Infinity;
    for (const name of names) {
      for (const val of result.states[name] ?? []) {
        if (isFinite(val)) { if (val < lo) lo = val; if (val > hi) hi = val; }
      }
    }
    if (!(lo <= hi)) { lo = -1; hi = 1; }
    const pad = (hi - lo) * 0.06 || 1;
    lo -= pad; hi += pad;
    drawAxes(ctx, frame.left, top, frame.plotWidth, height, zTMin, zTMax, lo, hi, undefined, false);
    for (const name of names) {
      drawLine(ctx, frame.left, top, frame.plotWidth, height, time, result.states[name] ?? [],
        zTMin, zTMax, lo, hi, at(COLORS as readonly string[], vars.indexOf(name) % COLORS.length));
    }
    ctx.save();
    ctx.translate(10, top + height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = AXIS; ctx.font = "11px monospace"; ctx.textAlign = "center";
    ctx.fillText(traceAxisLabel(names, units), 0, 0);
    ctx.restore();
  };
  drawStatePanel(groups.primary, frame.top, firstH);
  if (groups.secondary.length > 0) drawStatePanel(groups.secondary, frame.top + firstH + gap, secondH);
  // The imported trace is a voltage; it is compared on the first state's axis.
  let vMin = Infinity, vMax = -Infinity;
  for (const name of groups.primary) {
    for (const val of result.states[name] ?? []) {
      if (isFinite(val)) { if (val < vMin) vMin = val; if (val > vMax) vMax = val; }
    }
  }
  const vPad = (vMax - vMin) * 0.06 || 1;
  vMin -= vPad; vMax += vPad;

  // Spike markers
  if (hasSpikes) {
    ctx.strokeStyle = "rgba(255,82,82,0.2)"; ctx.lineWidth = 1;
    for (const idx of result.spikes) {
      const x = frame.left + (((idx + 1) * result.dt - zTMin) / (zTMax - zTMin || 1)) * frame.plotWidth;
      ctx.beginPath(); ctx.moveTo(x, frame.top); ctx.lineTo(x, frame.top + voltH); ctx.stroke();
    }
  }
  // Legend
  ctx.font = "11px monospace";
  vars.forEach((v, i) => {
    const colour = at(COLORS as readonly string[], i % COLORS.length);
    ctx.fillStyle = colour;
    ctx.fillRect(frame.left + 6 + i * 52, frame.top + 4, 8, 2);
    ctx.textAlign = "left"; ctx.fillText(v, frame.left + 17 + i * 52, frame.top + 9);
  });

  // Imported trace overlay
  if (importedTrace) {
    ctx.setLineDash([4, 3]);
    drawLine(ctx, frame.left, frame.top, frame.plotWidth, firstH, importedTrace.time, importedTrace.voltage,
      zTMin, zTMax, vMin, vMax, "#ff9800", 1.5);
    ctx.setLineDash([]);
    ctx.fillStyle = "#ff9800"; ctx.font = "11px monospace"; ctx.textAlign = "left";
    ctx.fillText("imported", frame.left + 6 + vars.length * 52, frame.top + 9);
  }

  // Current plot
  const curY = frame.top + voltH + gap;
  const I = result.current_trace;
  let iMin = Math.min(...I), iMax = Math.max(...I);
  if (iMin === iMax) { iMin -= 1; iMax += 1; }
  drawAxes(ctx, frame.left, curY, frame.plotWidth, currentH, zTMin, zTMax, iMin, iMax * 1.1, undefined, false);
  drawLine(ctx, frame.left, curY, frame.plotWidth, currentH, time, I, zTMin, zTMax, iMin, iMax * 1.1, "#ffb74d", 1.5);
  ctx.fillStyle = "#ffb74d"; ctx.font = "11px monospace"; ctx.textAlign = "left";
  // No catalogue model declares the unit of its drive (a conductance model
  // reads µA/cm², an integrate-and-fire model pA or a dimensionless input),
  // so the panel names the quantity and says whose unit it is; the axis was
  // labelled nA for every model.
  ctx.fillText(CURRENT_PANEL_LABEL, frame.left + 4, curY + 10);
  ctx.save();
  ctx.translate(10, curY + currentH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = AXIS; ctx.font = "11px monospace"; ctx.textAlign = "center";
  ctx.fillText("drive", 0, 0);
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
  ctx.fillStyle = AXIS; ctx.font = "11px monospace"; ctx.textAlign = "center";
  const xs = niceStep(zTMax - zTMin, 6);
  for (let v = Math.ceil(zTMin / xs) * xs; v <= zTMax; v += xs) {
    const x = frame.left + ((v - zTMin) / (zTMax - zTMin || 1)) * frame.plotWidth;
    ctx.fillText(v.toFixed(0), x, frame.height - 2);
  }
  // The unit sits at the left end of the axis: at the right end the view's
  // "Data table" button covered it.
  ctx.textAlign = "right"; ctx.fillText("ms", frame.left - 4, frame.height - 2);

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
    ctx.fillStyle = "#4fc3f7"; ctx.font = "11px monospace"; ctx.textAlign = "right";
    ctx.fillText(`zoom: ${zTMin.toFixed(1)}–${zTMax.toFixed(1)} ms (dbl-click to reset)`, frame.left + frame.plotWidth - 2, frame.height - 2);
  }
}
