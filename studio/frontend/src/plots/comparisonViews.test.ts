// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/** Cases for the views that draw more than one run. */

import { describe, expect, it } from "vitest";

import type { CompareResponse, PrecisionResponse, SimulateResponse } from "../api/client";
import { PLOT_COLORS } from "../simulationPlotCanvas";
import {
  drawCompareView,
  drawMultiModelView,
  drawPrecisionView,
} from "./comparisonViews";
import { drewNonFinite, mockPlotContext } from "./mockPlotContext";
import { plotFrame } from "./plotFrame";

const FRAME = plotFrame(400, 300);

/**
 * A run carrying one variable, enough for the views that only plot the first.
 *
 * @param name - The variable's name.
 * @param values - Its samples.
 * @param modelName - The name to label it with.
 * @returns A run with only what these views read.
 */
function run(name: string, values: number[], modelName = "LIF"): SimulateResponse {
  return {
    dt: 0.1,
    model_name: modelName,
    states: { [name]: values },
    stats: { rate_hz: 12 },
    time: values.map((_, index) => index * 0.1),
  } as unknown as SimulateResponse;
}

describe("drawCompareView", () => {
  it("labels both runs with their own model and rate", () => {
    const recording = mockPlotContext();

    drawCompareView(recording.ctx, FRAME, {
      a: run("v", [-65, -60, -20], "AdEx"),
      b: run("v", [-70, -68, -30], "Izhikevich"),
    } as CompareResponse);

    const labels = recording.texts.map((t) => t.text);
    expect(labels).toContain("A: AdEx (12 Hz)");
    expect(labels).toContain("B: Izhikevich (12 Hz)");
    expect(drewNonFinite(recording)).toBe(false);
  });

  it("labels a run with no model name as custom", () => {
    const recording = mockPlotContext();

    drawCompareView(recording.ctx, FRAME, {
      a: run("v", [-65, -60], ""),
      b: run("v", [-70, -68], ""),
    } as CompareResponse);

    expect(recording.texts.map((t) => t.text)).toContain("A: custom (12 Hz)");
  });

  it("draws finite geometry for a run with no states at all", () => {
    const recording = mockPlotContext();
    const empty = { dt: 0.1, states: {}, stats: { rate_hz: 0 }, time: [] } as unknown as SimulateResponse;

    drawCompareView(recording.ctx, FRAME, { a: empty, b: empty } as CompareResponse);

    expect(drewNonFinite(recording)).toBe(false);
  });
});

describe("drawMultiModelView", () => {
  it("draws one trace and one legend entry per run", () => {
    const recording = mockPlotContext();

    drawMultiModelView(recording.ctx, FRAME, [
      run("v", [-65, -60, -20], "AdEx"),
      run("v", [-70, -68, -30], "Izhikevich"),
    ]);

    const swatches = recording.rects.filter((r) => r.width === 8 && r.height === 2);
    expect(swatches).toHaveLength(2);
    expect(swatches[0]?.fill).toBe(PLOT_COLORS[0]);
    expect(swatches[1]?.fill).toBe(PLOT_COLORS[1]);
    expect(drewNonFinite(recording)).toBe(false);
  });

  it("names a run with no model name by its position", () => {
    const recording = mockPlotContext();

    drawMultiModelView(recording.ctx, FRAME, [run("v", [-65, -60], "")]);

    expect(recording.texts.map((t) => t.text)).toContain("Model 1 (12Hz)");
  });

  it("wraps the palette rather than running off the end of it", () => {
    const recording = mockPlotContext();
    const many = Array.from({ length: PLOT_COLORS.length + 2 }, () => run("v", [-65, -60]));

    drawMultiModelView(recording.ctx, FRAME, many);

    const swatches = recording.rects.filter((r) => r.width === 8 && r.height === 2);
    expect(swatches).toHaveLength(many.length);
    expect(swatches[PLOT_COLORS.length]?.fill).toBe(PLOT_COLORS[0]);
  });

  it("draws nothing for no runs", () => {
    const recording = mockPlotContext();

    drawMultiModelView(recording.ctx, FRAME, []);

    expect(recording.texts).toEqual([]);
    expect(recording.path).toEqual([]);
  });
});

describe("drawPrecisionView", () => {
  /** A precision comparison with only the fields the view reads. */
  const comparison = {
    arithmetic: { overflow: "wrap", q_format: "Q8.8", rounding: "truncate" },
    error: {
      display: [0, 0.1, 0.2],
      max_error: 0.2,
      mean_error: 0.1,
      rms_error: 0.12,
      trace: [0, 0.1, 0.2],
      variable: "v",
    },
    fixed_result: run("v", [-65, -60, -21]),
    float_result: run("v", [-65, -60, -20]),
  } as unknown as PrecisionResponse;

  it("names both arithmetics it is comparing", () => {
    const recording = mockPlotContext();

    drawPrecisionView(recording.ctx, FRAME, comparison);

    const labels = recording.texts.map((t) => t.text);
    expect(labels).toContain("float64");
    expect(labels.some((l) => l.includes("Q8.8"))).toBe(true);
    expect(drewNonFinite(recording)).toBe(false);
  });

  it("falls back to a plain format name when there is no bit-true arithmetic", () => {
    const recording = mockPlotContext();
    const withoutArithmetic = {
      ...comparison,
      arithmetic: undefined,
      encoding: { q_format: "Q4.12" },
    } as unknown as PrecisionResponse;

    drawPrecisionView(recording.ctx, FRAME, withoutArithmetic);

    expect(recording.texts.map((t) => t.text)).toContain("Q4.12");
  });

  it("says fixed-point when neither block names a format", () => {
    const recording = mockPlotContext();
    const bare = {
      ...comparison,
      arithmetic: undefined,
      encoding: undefined,
    } as unknown as PrecisionResponse;

    drawPrecisionView(recording.ctx, FRAME, bare);

    expect(recording.texts.map((t) => t.text)).toContain("fixed-point");
  });
});
