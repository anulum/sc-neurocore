// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * Cases for the single-series analysis views.
 *
 * None of these could be reached before without mounting the component and
 * driving the store. The cases that matter are the degenerate inputs — an
 * empty sweep, a histogram whose edges are one short, an elasticity that is
 * undefined — because those are what produced silent `NaN` geometry.
 */

import { describe, expect, it } from "vitest";

import type { AnalysisResultMetadata, SimulateResponse } from "../api/client";
import { PLOT_AXIS } from "../simulationPlotCanvas";
import {
  drawBifurcationView,
  drawFICurveView,
  drawFrequencyResponseView,
  drawHeatmapView,
  drawIsiHistogramView,
  drawSensitivityView,
  drawSpikeTriggeredAverageView,
} from "./analysisViews";
import { drewNonFinite, mockPlotContext } from "./mockPlotContext";
import { plotFrame } from "./plotFrame";

const FRAME = plotFrame(400, 300);

/** Analysis metadata is required by the types and read by none of these views. */
const METADATA: AnalysisResultMetadata = {
  analysis_type: "test",
  evidence_classification: "analysis",
  input_sha256: "a".repeat(64),
  output_keys: [],
  result_sha256: "b".repeat(64),
  schema_version: "studio.analysis-result.v1",
  source: "model",
  status: "completed",
};

describe("drawFICurveView", () => {
  it("labels the axis and draws the curve", () => {
    const recording = mockPlotContext();

    drawFICurveView(recording.ctx, FRAME, {
      analysis_metadata: METADATA,
      currents: [0, 1, 2],
      rates: [0, 10, 20],
    });

    expect(recording.texts.map((t) => t.text)).toContain("f (Hz)");
    expect(recording.path.length).toBeGreaterThan(0);
    expect(drewNonFinite(recording)).toBe(false);
  });

  it("draws finite geometry for a sweep with a single point", () => {
    // One current gives a zero-width axis; the fallback puts the right edge
    // one unit past the left rather than dividing by zero.
    const recording = mockPlotContext();

    drawFICurveView(recording.ctx, FRAME, {
      analysis_metadata: METADATA,
      currents: [1],
      rates: [5],
    });

    expect(drewNonFinite(recording)).toBe(false);
  });

  it("draws finite geometry for an empty sweep", () => {
    const recording = mockPlotContext();

    drawFICurveView(recording.ctx, FRAME, {
      analysis_metadata: METADATA,
      currents: [],
      rates: [],
    });

    expect(drewNonFinite(recording)).toBe(false);
  });
});

describe("drawIsiHistogramView", () => {
  /**
   * A run carrying the histogram given.
   *
   * @param histogram - The histogram to attach, or `null` for none.
   * @returns A run with only what this view reads.
   */
  function runWith(histogram: { counts: number[]; edges: number[] } | null): SimulateResponse {
    return { stats: { isi_histogram: histogram } } as SimulateResponse;
  }

  it("draws one bar per count", () => {
    const recording = mockPlotContext();

    drawIsiHistogramView(recording.ctx, FRAME, runWith({
      counts: [2, 5, 1],
      edges: [0, 10, 20, 30],
    }));

    expect(recording.rects.filter((r) => r.fill === "rgba(79, 195, 247, 0.6)")).toHaveLength(3);
    expect(drewNonFinite(recording)).toBe(false);
  });

  it("draws nothing for a run with no histogram", () => {
    const recording = mockPlotContext();

    drawIsiHistogramView(recording.ctx, FRAME, runWith(null));

    expect(recording.rects).toEqual([]);
    expect(recording.texts).toEqual([]);
  });

  it("gives the last bar zero width when the edges are one short", () => {
    // A server sending one edge per count leaves the final bar without a
    // right-hand edge. It used to be drawn with a NaN width, which draws
    // nothing and reports nothing.
    const recording = mockPlotContext();

    drawIsiHistogramView(recording.ctx, FRAME, runWith({
      counts: [2, 5, 1],
      edges: [0, 10, 20],
    }));

    expect(drewNonFinite(recording)).toBe(false);
    const bars = recording.rects.filter((r) => r.fill === "rgba(79, 195, 247, 0.6)");
    expect(bars).toHaveLength(3);
    expect(bars[2]?.width).toBe(1);
  });
});

describe("drawBifurcationView", () => {
  it("marks every attractor value it was given", () => {
    const recording = mockPlotContext();

    drawBifurcationView(recording.ctx, FRAME, {
      analysis_metadata: METADATA,
      attractors: [[-60, -50], [-55]],
      param_name: "I",
      param_values: [0, 1],
    });

    expect(recording.rects.filter((r) => r.width === 2 && r.height === 2)).toHaveLength(3);
    expect(drewNonFinite(recording)).toBe(false);
  });

  it("survives a sweep point with no attractors recorded for it", () => {
    const recording = mockPlotContext();

    drawBifurcationView(recording.ctx, FRAME, {
      analysis_metadata: METADATA,
      attractors: [[-60]],
      param_name: "I",
      param_values: [0, 1, 2],
    });

    expect(drewNonFinite(recording)).toBe(false);
  });
});

describe("drawHeatmapView", () => {
  it("fills one cell per grid position", () => {
    const recording = mockPlotContext();

    drawHeatmapView(recording.ctx, FRAME, {
      analysis_metadata: METADATA,
      param_x: "a",
      param_y: "b",
      rate_max: 20,
      rate_min: 0,
      rates: [[0, 10], [15, 20]],
      x_values: [0, 1],
      y_values: [0, 1],
    });

    const cells = recording.rects.filter((r) => r.fill.startsWith("rgb("));
    expect(cells).toHaveLength(4);
    expect(drewNonFinite(recording)).toBe(false);
  });

  it("draws a row the server did not send as the floor rate rather than NaN", () => {
    const recording = mockPlotContext();

    drawHeatmapView(recording.ctx, FRAME, {
      analysis_metadata: METADATA,
      param_x: "a",
      param_y: "b",
      rate_max: 20,
      rate_min: 0,
      rates: [[0, 10]],
      x_values: [0, 1],
      y_values: [0, 1],
    });

    expect(drewNonFinite(recording)).toBe(false);
    expect(recording.rects.filter((r) => r.fill.startsWith("rgb("))).toHaveLength(4);
  });
});

describe("drawSensitivityView", () => {
  it("draws a bar for a defined elasticity", () => {
    const recording = mockPlotContext();

    drawSensitivityView(recording.ctx, FRAME, {
      analysis_metadata: METADATA,
      base_rate: 12,
      sensitivities: [{ param: "tau", sensitivity: 0.5 }],
    });

    expect(recording.rects.filter((r) => r.fill === "rgba(79,195,247,0.6)")).toHaveLength(1);
    expect(recording.texts.map((t) => t.text)).toContain("0.500");
  });

  it("writes the reason instead of a bar when the elasticity is undefined", () => {
    // A zero-length bar would read as an insensitive parameter, which is a
    // different statement from one whose elasticity does not exist.
    const recording = mockPlotContext();

    drawSensitivityView(recording.ctx, FRAME, {
      analysis_metadata: METADATA,
      base_rate: 0,
      sensitivities: [{ param: "tau", reason: "zero base rate", sensitivity: null }],
    });

    expect(recording.rects.filter((r) => r.fill === "rgba(79,195,247,0.6)")).toEqual([]);
    expect(recording.texts.map((t) => t.text)).toContain("undefined: zero base rate");
  });

  it("says so when an undefined elasticity carries no reason", () => {
    const recording = mockPlotContext();

    drawSensitivityView(recording.ctx, FRAME, {
      analysis_metadata: METADATA,
      base_rate: 0,
      sensitivities: [{ param: "tau", sensitivity: null }],
    });

    expect(recording.texts.map((t) => t.text)).toContain("undefined: no reason given");
  });

  it("draws nothing at all for an empty result", () => {
    const recording = mockPlotContext();

    drawSensitivityView(recording.ctx, FRAME, {
      analysis_metadata: METADATA,
      base_rate: 1,
      sensitivities: [],
    });

    expect(recording.texts).toEqual([]);
  });

  it("draws at most fifteen parameters", () => {
    const recording = mockPlotContext();

    drawSensitivityView(recording.ctx, FRAME, {
      analysis_metadata: METADATA,
      base_rate: 1,
      sensitivities: Array.from({ length: 40 }, (_, index) => ({
        param: `p${String(index)}`,
        sensitivity: 1,
      })),
    });

    expect(recording.rects.filter((r) => r.fill === "rgba(79,195,247,0.6)")).toHaveLength(15);
  });
});

describe("drawFrequencyResponseView", () => {
  it("names the amplitude the sweep was driven at", () => {
    const recording = mockPlotContext();

    drawFrequencyResponseView(recording.ctx, FRAME, {
      amplitude: 2.5,
      analysis_metadata: METADATA,
      frequencies_hz: [1, 10, 100],
      rates: [1, 5, 9],
    });

    expect(recording.texts.map((t) => t.text)).toContain("rate (Hz) @ amplitude=2.5");
    expect(drewNonFinite(recording)).toBe(false);
  });
});

describe("drawSpikeTriggeredAverageView", () => {
  it("rules the spike itself and names the sample size", () => {
    const recording = mockPlotContext();

    drawSpikeTriggeredAverageView(recording.ctx, FRAME, {
      average: [-65, -60, -20, -60],
      n_spikes: 7,
      time_ms: [-10, -5, 0, 5],
    });

    expect(recording.texts.map((t) => t.text)).toContain("STA (n=7 spikes)");
    expect(recording.strokes).toContain("#ff5252");
    expect(drewNonFinite(recording)).toBe(false);
  });

  it("draws nothing for an empty average", () => {
    const recording = mockPlotContext();

    drawSpikeTriggeredAverageView(recording.ctx, FRAME, {
      average: [],
      n_spikes: 0,
      time_ms: [],
    });

    expect(recording.texts).toEqual([]);
    expect(recording.path).toEqual([]);
  });
});

describe("every analysis view", () => {
  it("uses the axis colour for its own labels", () => {
    const recording = mockPlotContext();

    drawFICurveView(recording.ctx, FRAME, {
      analysis_metadata: METADATA,
      currents: [0, 1],
      rates: [0, 5],
    });

    expect(recording.texts.find((t) => t.text === "f (Hz)")?.fill).toBe(PLOT_AXIS);
  });
});
