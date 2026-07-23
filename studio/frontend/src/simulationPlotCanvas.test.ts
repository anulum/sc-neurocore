// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { describe, expect, it, vi } from "vitest";

import {
  drawAxes,
  drawLine,
  niceStep,
  PLOT_AXIS,
  PLOT_BORDER,
  PLOT_PANEL_BG,
} from "./simulationPlotCanvas";

function mockContext(): CanvasRenderingContext2D {
  const path: string[] = [];
  return {
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 0,
    font: "",
    textAlign: "start",
    fillRect: vi.fn(),
    strokeRect: vi.fn(),
    beginPath: vi.fn(() => {
      path.push("begin");
    }),
    moveTo: vi.fn((x: number, y: number) => {
      path.push(`M${x},${y}`);
    }),
    lineTo: vi.fn((x: number, y: number) => {
      path.push(`L${x},${y}`);
    }),
    stroke: vi.fn(),
    fillText: vi.fn(),
    __path: path,
  } as unknown as CanvasRenderingContext2D & { __path: string[] };
}

describe("niceStep", () => {
  it("returns 1 for non-positive or non-finite ranges", () => {
    expect(niceStep(0, 5)).toBe(1);
    expect(niceStep(-10, 5)).toBe(1);
    expect(niceStep(Number.NaN, 5)).toBe(1);
    expect(niceStep(Number.POSITIVE_INFINITY, 5)).toBe(1);
  });

  it("snaps rough steps onto the 1–2–5 decade ladder", () => {
    expect(niceStep(100, 5)).toBe(20);
    expect(niceStep(10, 5)).toBe(2);
    expect(niceStep(1, 5)).toBe(0.2);
  });
});

describe("drawAxes", () => {
  it("paints panel background and border then emits grid strokes", () => {
    const ctx = mockContext();
    drawAxes(ctx, 10, 20, 200, 100, 0, 10, 0, 5, "t (ms)");
    expect(ctx.fillStyle).toBe(PLOT_AXIS);
    expect(ctx.fillRect).toHaveBeenCalledWith(10, 20, 200, 100);
    expect(ctx.strokeRect).toHaveBeenCalledWith(10, 20, 200, 100);
    // last strokeStyle before label work uses axis/grid palette
    expect([PLOT_BORDER, PLOT_AXIS, PLOT_PANEL_BG, "#1a1f2a"]).toContain(
      // strokeStyle ends as grid during X ticks
      (ctx as { strokeStyle: string }).strokeStyle,
    );
    expect(ctx.fillText).toHaveBeenCalled();
    expect(ctx.beginPath).toHaveBeenCalled();
  });
});

describe("drawLine", () => {
  it("strokes a polyline through scaled data points", () => {
    const ctx = mockContext() as CanvasRenderingContext2D & { __path: string[] };
    drawLine(ctx, 0, 0, 100, 50, [0, 1], [0, 10], 0, 1, 0, 10, "#4fc3f7", 2);
    expect(ctx.strokeStyle).toBe("#4fc3f7");
    expect(ctx.lineWidth).toBe(2);
    expect(ctx.moveTo).toHaveBeenCalledWith(0, 50);
    expect(ctx.lineTo).toHaveBeenCalledWith(100, 0);
    expect(ctx.stroke).toHaveBeenCalled();
  });
});
