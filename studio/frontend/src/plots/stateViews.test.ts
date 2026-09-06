// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/** Cases for the state-space, characterisation and network views. */

import { describe, expect, it } from "vitest";

import type {
  CharacterizeResponse,
  NetworkResult,
  NullclineResponse,
  SimulateResponse,
} from "../api/client";
import { drewNonFinite, mockPlotContext } from "./mockPlotContext";
import { plotFrame } from "./plotFrame";
import {
  drawCharacterizeView,
  drawNetworkView,
  drawPhasePortraitView,
} from "./stateViews";

const FRAME = plotFrame(400, 300);

/**
 * A run with the state variables given.
 *
 * @param states - Variable name to samples.
 * @returns A run with only what these views read.
 */
function run(states: Record<string, number[]>): SimulateResponse {
  return { dt: 0.1, states, time: [0, 0.1, 0.2] } as unknown as SimulateResponse;
}

describe("drawPhasePortraitView", () => {
  it("draws the trajectory and names the second variable", () => {
    const recording = mockPlotContext();

    drawPhasePortraitView(recording.ctx, FRAME, run({ v: [-65, -60, -20], w: [0, 1, 2] }), null);

    expect(recording.texts.map((t) => t.text)).toContain("w");
    expect(recording.path.length).toBeGreaterThan(0);
    expect(drewNonFinite(recording)).toBe(false);
  });

  it("draws nothing for a system with one variable", () => {
    // The component falls through to the trace view in this case; the view
    // refuses rather than dividing by a zero-width axis.
    const recording = mockPlotContext();

    drawPhasePortraitView(recording.ctx, FRAME, run({ v: [-65, -60] }), null);

    expect(recording.path).toEqual([]);
    expect(recording.texts).toEqual([]);
  });

  it("plots a nullcline point only where the field was valid", () => {
    const recording = mockPlotContext();
    const nullclines = {
      grid: { size: 2, x: [-70, -20], y: [0, 2] },
      nullcline_0: { points: [[-65, 0], [-60, 1]], variable: "v" },
      nullcline_1: { points: [[-64, 0]], variable: "w" },
      validity_0: [[1, 0], [0, 0]],
      validity_1: [[1, 0], [0, 0]],
      var_names: ["v", "w"],
    } as unknown as NullclineResponse;

    drawPhasePortraitView(recording.ctx, FRAME, run({ v: [-65, -60, -20], w: [0, 1, 2] }), nullclines);

    expect(drewNonFinite(recording)).toBe(false);
    expect(recording.rects.some((r) => r.width === 2 && r.height === 2)).toBe(true);
  });
});

describe("drawCharacterizeView", () => {
  it("names the f-I curve panel and every state range", () => {
    const recording = mockPlotContext();

    drawCharacterizeView(recording.ctx, FRAME, {
      fi_curve: { currents: [0, 1, 2], rates: [0, 5, 12] },
      max_rate: 12,
      pattern: { description: "regular firing", pattern: "tonic" },
      spike_count: 9,
      state_ranges: { v: { max: -20, mean: -55, min: -70 } },
      stats: { rate_hz: 12 },
      threshold_current: 0.5,
      top_sensitivities: [],
    } as unknown as CharacterizeResponse);

    const labels = recording.texts.map((t) => t.text);
    expect(labels).toContain("f-I curve");
    // The panel writes the pattern's sentence, not its slug: the reader is
    // being told what the model does, not which enum it matched.
    expect(labels).toContain("Pattern: regular firing");
    expect(labels).toContain("v: [-70, -20] mean=-55");
    expect(drewNonFinite(recording)).toBe(false);
  });

  it("draws finite geometry when no current elicited a spike", () => {
    const recording = mockPlotContext();

    drawCharacterizeView(recording.ctx, FRAME, {
      fi_curve: { currents: [], rates: [] },
      max_rate: 0,
      pattern: { description: "no spikes", pattern: "silent" },
      spike_count: 0,
      state_ranges: {},
      stats: { rate_hz: 0 },
      threshold_current: null,
      top_sensitivities: [],
    } as unknown as CharacterizeResponse);

    expect(drewNonFinite(recording)).toBe(false);
  });
});

describe("drawNetworkView", () => {
  /** A network run with the spikes given. */
  const network = {
    duration: 100,
    dt: 0.1,
    exc_rates: [10, 12],
    inh_rates: [20, 22],
    inh_rates_time: [],
    mean_exc_rate: 11,
    mean_inh_rate: 21,
    n_exc: 8,
    n_inh: 2,
    n_spikes: 3,
    n_total: 10,
    rate_time: [0, 50],
    spike_neurons: [0, 5, 9],
    spike_times: [1, 20, 60],
  } as unknown as NetworkResult;

  it("colours excitatory and inhibitory spikes differently", () => {
    const recording = mockPlotContext();

    drawNetworkView(recording.ctx, FRAME, network);

    const dots = recording.rects.filter((r) => r.width === 1.5);
    expect(dots).toHaveLength(3);
    expect(dots.filter((d) => d.fill === "#4fc3f7")).toHaveLength(2);
    expect(dots.filter((d) => d.fill === "#ff5252")).toHaveLength(1);
    expect(drewNonFinite(recording)).toBe(false);
  });

  it("names both population counts", () => {
    const recording = mockPlotContext();

    drawNetworkView(recording.ctx, FRAME, network);

    const labels = recording.texts.map((t) => t.text);
    expect(labels).toContain("E (8)");
    expect(labels).toContain("I (2)");
    expect(labels).toContain("3 spikes");
  });

  it("skips the rate panel when there is only one rate sample", () => {
    const recording = mockPlotContext();

    drawNetworkView(recording.ctx, FRAME, { ...network, rate_time: [0] });

    expect(drewNonFinite(recording)).toBe(false);
    expect(recording.texts.map((t) => t.text)).not.toContain("E: 11Hz  I: 21Hz");
  });
});
