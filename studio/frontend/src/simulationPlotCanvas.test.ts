// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { describe, expect, it } from "vitest";

import { mockPlotContext, drewNonFinite } from "./plots/mockPlotContext";

import {
  drawAxes,
  drawLine,
  niceStep,
  PLOT_AXIS,
  PLOT_BORDER,
  PLOT_PANEL_BG,
} from "./simulationPlotCanvas";

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
  it("paints the panel, then draws every tick label in the axis colour", () => {
    const recording = mockPlotContext();
    drawAxes(recording.ctx, 10, 20, 200, 100, 0, 10, 0, 5, "t (ms)");

    expect(recording.rects).toContainEqual({
      fill: PLOT_PANEL_BG, height: 100, width: 200, x: 10, y: 20,
    });
    expect(recording.strokes).toContain(PLOT_BORDER);
    expect(recording.texts.length).toBeGreaterThan(0);
    expect(recording.texts.every((text) => text.fill === PLOT_AXIS)).toBe(true);
    expect(drewNonFinite(recording)).toBe(false);
  });

  it("labels the horizontal axis with the unit it was given", () => {
    const recording = mockPlotContext();
    drawAxes(recording.ctx, 10, 20, 200, 100, 0, 10, 0, 5, "t (ms)");

    expect(recording.texts.map((text) => text.text)).toContain("t (ms)");
  });
});

describe("drawLine", () => {
  it("strokes a polyline through scaled data points", () => {
    const recording = mockPlotContext();
    drawLine(recording.ctx, 0, 0, 100, 50, [0, 1], [0, 10], 0, 1, 0, 10, "#4fc3f7", 2);

    expect(recording.strokes).toContain("#4fc3f7");
    expect(recording.path).toEqual(["M0,50", "L100,0"]);
    expect(drewNonFinite(recording)).toBe(false);
  });

  it("draws the paired prefix when the two lists disagree in length", () => {
    const recording = mockPlotContext();
    drawLine(recording.ctx, 0, 0, 100, 50, [0, 0.5, 1], [0, 10], 0, 1, 0, 10, "#4fc3f7", 2);

    expect(recording.path).toEqual(["M0,50", "L50,0"]);
    expect(drewNonFinite(recording)).toBe(false);
  });

  it("draws nothing at all rather than a NaN path when one list is empty", () => {
    const recording = mockPlotContext();
    drawLine(recording.ctx, 0, 0, 100, 50, [0, 1], [], 0, 1, 0, 10, "#4fc3f7", 2);

    expect(recording.path).toEqual([]);
  });
});
