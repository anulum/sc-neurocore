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
  displayIndices,
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

/**
 * A long membrane-like trace: a slow ramp with narrow spikes.
 *
 * @param samples - How many samples.
 * @returns Time and voltage, time strictly increasing.
 */
function longTrace(samples: number): { time: number[]; voltage: number[] } {
  const time = Array.from({ length: samples }, (_, i) => i * 0.1);
  const voltage = time.map((t, i) => (i % 997 === 0 ? 30 : -70 + 10 * Math.sin(t / 50)));
  return { time, voltage };
}

describe("displayIndices", () => {
  it("keeps every sample of a series the panel can hold", () => {
    expect(displayIndices([0, 1, 2], [5, 6, 7], 0, 2, 100)).toEqual([0, 1, 2]);
    expect(displayIndices([0, 1], [5], 0, 1, 0)).toEqual([0]);
  });

  it("keeps every sample when the horizontal values do not increase", () => {
    const x = Array.from({ length: 1000 }, (_, i) => Math.sin(i));
    const y = Array.from({ length: 1000 }, (_, i) => Math.cos(i));
    expect(displayIndices(x, y, -1, 1, 10)).toHaveLength(1000);
  });

  it("strokes at most four samples per column and keeps every visible extreme", () => {
    const { time, voltage } = longTrace(200_000);
    const kept = displayIndices(time, voltage, 0, time.at(-1) ?? 0, 800);

    expect(kept.length).toBeLessThanOrEqual(4 * 800 + 2);
    const keptValues = kept.map((i) => voltage[i] ?? Number.NaN);
    const highest = (values: number[]) => values.reduce((a, b) => Math.max(a, b), -Infinity);
    const lowest = (values: number[]) => values.reduce((a, b) => Math.min(a, b), Infinity);
    expect(highest(keptValues)).toBe(highest(voltage));
    expect(lowest(keptValues)).toBe(lowest(voltage));
    // Every spike sample survives: a spike is a column maximum.
    const spikes = voltage.flatMap((v, i) => (v === 30 ? [i] : []));
    expect(spikes.every((i) => kept.includes(i))).toBe(true);
    expect(kept[0]).toBe(0);
    expect(kept.at(-1)).toBe(voltage.length - 1);
    expect([...kept].sort((a, b) => a - b)).toEqual(kept);
  });

  it("strokes the visible window and one sample either side of it", () => {
    const { time, voltage } = longTrace(100_000);
    const kept = displayIndices(time, voltage, 1000, 2000, 200);

    expect(kept[0]).toBe(9999);
    expect(kept.at(-1)).toBe(20001);
    expect(kept.every((i) => i >= 9999 && i <= 20001)).toBe(true);
  });

  it("leaves non-finite samples out of the extremes", () => {
    const time = Array.from({ length: 5000 }, (_, i) => i);
    const voltage = time.map((i) => (i % 2 === 0 ? Number.NaN : i));
    const kept = displayIndices(time, voltage, 0, 4999, 10);

    expect(kept.filter((i) => Number.isNaN(voltage[i])).every((i) => i === 0)).toBe(true);
    expect(kept).toContain(4999);
  });

  it("cuts the path drawLine hands the canvas without moving what it shows", () => {
    const { time, voltage } = longTrace(100_000);
    const recording = mockPlotContext();
    drawLine(recording.ctx, 0, 0, 800, 200, time, voltage, 0, time.at(-1) ?? 0, -80, 40, "#4fc3f7");

    expect(recording.path.length).toBeLessThanOrEqual(4 * 800 + 2);
    // The first sample is a spike at 30 mV on a -80..40 mV axis 200 px tall.
    expect(recording.path[0]).toBe(`M0,${200 - ((30 + 80) / 120) * 200}`);
    expect(drewNonFinite(recording)).toBe(false);
  });
});
