// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/** Cases for the plot geometry and the canvas preparation. */

import { describe, expect, it, vi } from "vitest";

import { PLOT_BG } from "../simulationPlotCanvas";
import {
  PLOT_BOTTOM,
  PLOT_LEFT,
  PLOT_MIN_SIDE,
  PLOT_RIGHT,
  PLOT_TOP,
  plotFrame,
  preparePlotCanvas,
} from "./plotFrame";
import { mockPlotContext } from "./mockPlotContext";

/**
 * The arguments every `scale` call was made with.
 *
 * Read through a helper rather than asserted on the method itself, because
 * naming `ctx.scale` unbound is a real hazard the strict profile refuses.
 *
 * @param ctx - The recording context.
 * @returns One entry per call.
 */
function scaleCalls(ctx: CanvasRenderingContext2D): [number, number][] {
  return (ctx.scale as unknown as { mock: { calls: [number, number][] } }).mock.calls;
}

/**
 * A canvas element whose context is the recording one.
 *
 * @param ctx - The context `getContext` should hand back, or `null`.
 * @returns The stand-in element.
 */
function canvasWith(ctx: CanvasRenderingContext2D | null): HTMLCanvasElement {
  return {
    width: 0,
    height: 0,
    style: { width: "", height: "" },
    getContext: vi.fn(() => ctx),
  } as unknown as HTMLCanvasElement;
}

describe("plotFrame", () => {
  it("insets the plot area from both sides", () => {
    const frame = plotFrame(400, 300);

    expect(frame.left).toBe(PLOT_LEFT);
    expect(frame.top).toBe(PLOT_TOP);
    expect(frame.bottom).toBe(PLOT_BOTTOM);
    expect(frame.plotWidth).toBe(400 - PLOT_LEFT - PLOT_RIGHT);
    expect(frame.width).toBe(400);
    expect(frame.height).toBe(300);
  });

  it("reports a negative plot width rather than clamping a canvas too narrow to draw in", () => {
    // Clamping here would hand the views a frame that looks usable; the
    // caller refuses the canvas instead, and this records which of the two
    // does the refusing.
    expect(plotFrame(10, 300).plotWidth).toBeLessThan(0);
  });
});

describe("preparePlotCanvas", () => {
  it("sizes the canvas in device pixels and scales the context back", () => {
    const recording = mockPlotContext();
    const canvas = canvasWith(recording.ctx);

    const prepared = preparePlotCanvas(canvas, 400, 300, 2);

    expect(prepared).not.toBeNull();
    expect(canvas.width).toBe(800);
    expect(canvas.height).toBe(600);
    expect(canvas.style.width).toBe("400px");
    expect(canvas.style.height).toBe("300px");
    expect(scaleCalls(recording.ctx)).toEqual([[2, 2]]);
  });

  it("clears the canvas to the plot background before returning it", () => {
    const recording = mockPlotContext();

    preparePlotCanvas(canvasWith(recording.ctx), 400, 300, 1);

    expect(recording.rects).toEqual([
      { fill: PLOT_BG, height: 300, width: 400, x: 0, y: 0 },
    ]);
  });

  it("treats a nonsensical device pixel ratio as one", () => {
    const recording = mockPlotContext();
    const canvas = canvasWith(recording.ctx);

    preparePlotCanvas(canvas, 400, 300, 0);

    expect(canvas.width).toBe(400);
    expect(scaleCalls(recording.ctx)).toEqual([[1, 1]]);
  });

  it("refuses a canvas too small to draw in, on either side", () => {
    const recording = mockPlotContext();

    expect(preparePlotCanvas(canvasWith(recording.ctx), PLOT_MIN_SIDE - 1, 300, 1)).toBeNull();
    expect(preparePlotCanvas(canvasWith(recording.ctx), 400, PLOT_MIN_SIDE - 1, 1)).toBeNull();
    expect(recording.rects).toEqual([]);
  });

  it("refuses a canvas whose context the browser will not give", () => {
    expect(preparePlotCanvas(canvasWith(null), 400, 300, 1)).toBeNull();
  });
});
