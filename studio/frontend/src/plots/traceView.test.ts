// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * Cases for the default trace view.
 *
 * Zoom and crosshair are arguments here, not refs, so the interaction states
 * that used to need a mouse are ordinary inputs.
 */

import { describe, expect, it } from "vitest";

import type { ImportedTrace, SimulateResponse } from "../api/client";
import { drewNonFinite, mockPlotContext } from "./mockPlotContext";
import { plotFrame } from "./plotFrame";
import { drawTraceView, type TraceViewOptions } from "./traceView";

const FRAME = plotFrame(400, 300);

/** No zoom, no crosshair, no overlay. */
const PLAIN: TraceViewOptions = {
  crosshair: null,
  importedTrace: null,
  zoom: { xMax: NaN, xMin: NaN },
};

/**
 * A run with one variable and the spikes given.
 *
 * @param spikes - Raw step indices at which it spiked.
 * @returns A run with only what the trace view reads.
 */
function run(spikes: number[] = []): SimulateResponse {
  return {
    current_trace: [0, 1, 1, 0],
    dt: 0.1,
    spikes,
    states: { v: [-65, -60, -20, -64] },
    time: [0.1, 0.2, 0.3, 0.4],
  } as unknown as SimulateResponse;
}

describe("drawTraceView", () => {
  it("labels both axes and draws the trace", () => {
    const recording = mockPlotContext();

    drawTraceView(recording.ctx, FRAME, run(), PLAIN);

    const labels = recording.texts.map((t) => t.text);
    expect(labels).toContain("mV");
    expect(labels).toContain("ms");
    expect(labels).toContain("v");
    expect(drewNonFinite(recording)).toBe(false);
  });

  it("draws no raster band for a run that never spiked", () => {
    const withSpikes = mockPlotContext();
    const without = mockPlotContext();

    drawTraceView(withSpikes.ctx, FRAME, run([1, 2]), PLAIN);
    drawTraceView(without.ctx, FRAME, run([]), PLAIN);

    expect(withSpikes.strokes).toContain("#ff5252");
    expect(without.strokes).not.toContain("#ff5252");
  });

  it("says what it is zoomed to, and only when it is zoomed", () => {
    const zoomed = mockPlotContext();
    const whole = mockPlotContext();

    drawTraceView(zoomed.ctx, FRAME, run(), { ...PLAIN, zoom: { xMax: 0.3, xMin: 0.1 } });
    drawTraceView(whole.ctx, FRAME, run(), PLAIN);

    expect(zoomed.texts.map((t) => t.text)).toContain(
      "zoom: 0.1–0.3 ms (dbl-click to reset)",
    );
    expect(whole.texts.some((t) => t.text.startsWith("zoom:"))).toBe(false);
  });

  it("rules the crosshair where it is asked to and nowhere otherwise", () => {
    const withCross = mockPlotContext();
    const without = mockPlotContext();

    drawTraceView(withCross.ctx, FRAME, run(), { ...PLAIN, crosshair: 123 });
    drawTraceView(without.ctx, FRAME, run(), PLAIN);

    expect(withCross.path).toContain("M123,8");
    expect(without.path.some((p) => p.startsWith("M123,"))).toBe(false);
  });

  it("overlays an imported trace when there is one", () => {
    const recording = mockPlotContext();
    const imported = {
      dt: 0.1,
      spikes: [],
      time: [0.1, 0.2, 0.3, 0.4],
      voltage: [-66, -61, -25, -63],
    } as unknown as ImportedTrace;

    drawTraceView(recording.ctx, FRAME, run(), { ...PLAIN, importedTrace: imported });

    expect(recording.strokes).toContain("#ff9800");
    expect(drewNonFinite(recording)).toBe(false);
  });

  it("draws finite geometry for a run with no samples at all", () => {
    const recording = mockPlotContext();
    const empty = {
      current_trace: [],
      dt: 0.1,
      spikes: [],
      states: {},
      time: [],
    } as unknown as SimulateResponse;

    drawTraceView(recording.ctx, FRAME, empty, PLAIN);

    expect(drewNonFinite(recording)).toBe(false);
  });

  it("draws nothing when the frame is too short for the stacked panels", () => {
    const recording = mockPlotContext();

    drawTraceView(recording.ctx, plotFrame(400, 101), run(), PLAIN);

    expect(recording.texts).toEqual([]);
    expect(recording.path).toEqual([]);
  });
});
