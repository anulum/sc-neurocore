// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — every analysis view in words and as a table

import { describe, expect, it } from "vitest";

import type {
  BifurcationResponse,
  CharacterizeResponse,
  CompareResponse,
  FICurveResponse,
  FreqResponse,
  HeatmapResponse,
  NetworkResult,
  NullclineResponse,
  PrecisionResponse,
  SensitivityResponse,
  SimulateResponse,
} from "./api/types";
import {
  describeBifurcation,
  describeCharacterization,
  describeCompare,
  describeFICurve,
  describeFrequency,
  describeHeatmap,
  describeIsi,
  describeNetwork,
  describePhase,
  describePrecision,
  describeSensitivity,
  describeSta,
  describeView,
  VIEW_TABLE_ROW_LIMIT,
  type ViewResults,
} from "./viewDescriptions";

/**
 * A run with the fields the descriptions read.
 *
 * @param overrides - Fields to change.
 * @returns The run.
 */
function run(overrides: Partial<SimulateResponse> = {}): SimulateResponse {
  return {
    time: [1, 2, 3, 4],
    states: { v: [-70, -50, 20, -68], w: [0, 1, 2, 1.5] },
    current_trace: [],
    spikes: [2],
    spike_count: 1,
    stats: { rate_hz: 250, isi_mean_ms: 4, isi_cv: 0.25, isi_histogram: { counts: [1, 5, 2], edges: [2, 4, 6, 8] } },
    dt: 1,
    n_steps: 4,
    model_name: "AdExNeuron",
    ...overrides,
  } as SimulateResponse;
}

describe("describeFICurve", () => {
  it("names the first measured firing point and the peak, not an interpolated threshold", () => {
    const fi = { currents: [0, 5, 10, 15], rates: [0, 0, 40, 90] } as FICurveResponse;
    const d = describeFICurve(fi);
    expect(d.sentence).toBe(
      "f-I curve over I from 0 to 15 in 4 points: silent up to I = 5; first firing measured at I = 10 (40 Hz); " +
        "highest rate 90 Hz at I = 15. Currents are in the model's own units.",
    );
    expect(d.table.rows).toEqual([["0", "0"], ["5", "0"], ["10", "40"], ["15", "90"]]);
  });

  it("says so when nothing fired", () => {
    expect(describeFICurve({ currents: [0, 1], rates: [0, 0] } as FICurveResponse).sentence)
      .toContain("no spikes at any current in the range");
  });

  it("caps a long table and counts what it left out", () => {
    const n = VIEW_TABLE_ROW_LIMIT + 5;
    const fi = { currents: Array.from({ length: n }, (_, i) => i), rates: Array<number>(n).fill(1) } as FICurveResponse;
    const rows = describeFICurve(fi).table.rows;
    expect(rows).toHaveLength(VIEW_TABLE_ROW_LIMIT + 1);
    expect(rows.at(-1)?.[0]).toBe("5 more rows in the JSON export of this analysis");
  });
});

describe("describeBifurcation", () => {
  it("tallies the server's own classification and says it is not a continuation", () => {
    const bif = {
      param_name: "g_na", param_values: [10, 20, 30], attractors: [[-65], [-70, 20], []],
      attractor_kinds: ["fixed_point", "extrema", "insufficient_samples"], variable: "v", protocol: "constant",
    } as unknown as BifurcationResponse;
    const d = describeBifurcation(bif);
    expect(d.sentence).toBe(
      "Extrema of v swept over g_na from 10 to 30 in 3 points under a constant drive: 1 point settles to a fixed " +
        "point; 1 point shows oscillation extrema; 1 point had too few samples to tell. This is a sampled sweep, " +
        "not a numerical continuation.",
    );
    expect(d.table.rows[1]).toEqual(["20", "extrema", "2", "-70", "20"]);
  });

  it("does not guess a kind the server did not give", () => {
    const bif = { param_name: "a", param_values: [1], attractors: [[1]] } as unknown as BifurcationResponse;
    const d = describeBifurcation(bif);
    expect(d.table.rows[0]?.[1]).toBe("not classified");
    expect(d.sentence).toContain("no point was classified");
  });
});

describe("describeHeatmap", () => {
  it("states the range, where the peak is and how many points were silent", () => {
    const map = {
      param_x: "g_na", x_values: [20, 40], param_y: "g_k", y_values: [5, 10],
      rates: [[0, 140], [30, 0]], rate_min: 0, rate_max: 140,
    } as unknown as HeatmapResponse;
    const d = describeHeatmap(map);
    expect(d.sentence).toBe(
      "Firing rate over g_na × g_k, 2 × 2 points: from 0 to 140 Hz, highest at g_na = 40, g_k = 5; 2 points of 4 silent.",
    );
    expect(d.table.columns).toEqual(["g_k \\ g_na", "20", "40"]);
    expect(d.table.rows).toEqual([["5", "0", "140"], ["10", "30", "0"]]);
  });
});

describe("describeSensitivity", () => {
  it("ranks the defined elasticities by size and counts the undefined ones", () => {
    const sens = {
      base_rate: 50,
      sensitivities: [
        { param: "a", sensitivity: 0.2, rate_minus: 48, rate_plus: 52 },
        { param: "b", sensitivity: -1.5 },
        { param: "c", sensitivity: null, reason: "parameter is zero" },
      ],
    } as unknown as SensitivityResponse;
    const d = describeSensitivity(sens);
    expect(d.sentence).toBe("Rate elasticity of 3 parameters around a base rate of 50 Hz: largest b (-1.5), a (0.2); 1 undefined.");
    expect(d.table.rows[2]).toEqual(["c", "none", "none", "none", "parameter is zero"]);
  });
});

describe("describeSta", () => {
  it("says which state it averages and where it peaks", () => {
    const d = describeSta({ time_ms: [-2, -1, 0, 1], average: [-60, -55, 10, -65], n_spikes: 3 }, "v");
    expect(d.sentence).toBe("Average of v around 3 spikes, from -2 to 1 ms relative to the spike: from -65 to 10, highest at 0 ms.");
  });
});

describe("describeFrequency", () => {
  it("names the drive amplitude and the frequencies of the highest and lowest rate", () => {
    const freq = { frequencies_hz: [1, 10, 100], rates: [20, 60, 5], amplitude: 8 } as unknown as FreqResponse;
    expect(describeFrequency(freq).sentence).toBe(
      "Firing rate under a sine drive of amplitude 8 from 1 to 100 Hz in 3 points: highest 60 Hz at 10 Hz, lowest 5 Hz at 100 Hz.",
    );
  });
});

describe("describeCharacterization", () => {
  it("states the pattern, the threshold, the peak rate and the most sensitive parameter", () => {
    const char = {
      pattern: { pattern: "tonic", description: "Regular tonic firing (CV=0.05)" },
      fi_curve: { currents: [], rates: [] }, threshold_current: 3.5, max_rate: 120,
      state_ranges: { v: { min: -70, max: 30, mean: -55 } },
      top_sensitivities: [{ param: "g_na", rate_change: 12 }], spike_count: 9,
      stats: { rate_hz: 90, isi_mean_ms: 11, isi_cv: 0.05, isi_histogram: null },
    } as CharacterizeResponse;
    const d = describeCharacterization(char);
    expect(d.sentence).toBe(
      "Regular tonic firing (CV=0.05); firing threshold near I = 3.5; highest rate 120 Hz; 9 spikes in the base run; most rate-sensitive parameter g_na.",
    );
    expect(d.table.rows).toContainEqual(["v range", "-70 to 30 (mean -55)"]);
  });

  it("says when no threshold lies in the swept range", () => {
    const char = {
      pattern: { pattern: "silent", description: "No spikes detected" }, fi_curve: { currents: [], rates: [] },
      threshold_current: null, max_rate: 0, state_ranges: {}, top_sensitivities: [], spike_count: 0,
      stats: { rate_hz: 0, isi_mean_ms: null, isi_cv: null, isi_histogram: null },
    } as CharacterizeResponse;
    expect(describeCharacterization(char).sentence).toContain("no firing threshold within the swept currents");
  });
});

describe("describeCompare", () => {
  it("states both runs' models, spikes, rates and ranges", () => {
    const cmp = { a: run(), b: run({ model_name: "WangBuzsakiNeuron", spike_count: 29, stats: { ...run().stats, rate_hz: 290 } }) } as CompareResponse;
    expect(describeCompare(cmp).sentence).toBe(
      "A/B comparison on the same drive. A: AdExNeuron, 1 spike (250 Hz), v from -70 to 20; " +
        "B: WangBuzsakiNeuron, 29 spikes (290 Hz), v from -70 to 20.",
    );
  });
});

describe("describeNetwork", () => {
  const net = (exc: number): NetworkResult => ({
    spike_times: [], spike_neurons: [], n_exc: 80, n_inh: 20, n_total: 100, n_spikes: 300,
    rate_time: [], exc_rates: [], inh_rates: [], duration: 100, dt: 0.5, mean_exc_rate: exc, mean_inh_rate: 30,
  });

  it("states the populations and their mean rates", () => {
    expect(describeNetwork(net(29)).sentence).toBe(
      "E-I network of 80 excitatory and 20 inhibitory neurons over 100 ms at dt 0.5 ms: 300 spikes; mean rate 29 Hz excitatory, 30 Hz inhibitory.",
    );
  });

  it("adds the saturation warning when a population is set by the time step", () => {
    expect(describeNetwork(net(1500)).sentence).toContain("the time step sets this rate, not the model");
  });
});

describe("describeIsi", () => {
  it("names the busiest bin and the interval statistics", () => {
    expect(describeIsi(run())?.sentence).toBe(
      "Interspike-interval histogram of 8 intervals: mean 4 ms, CV 0.25; most intervals (5) between 4 and 6 ms.",
    );
    expect(describeIsi(run({ stats: { rate_hz: 0, isi_mean_ms: null, isi_cv: null, isi_histogram: null } }))).toBeNull();
  });
});

describe("describePhase", () => {
  it("states both axes, where the orbit ended and the nullclines", () => {
    const nullclines = {
      var_names: ["v", "w"], nullcline_0: { variable: "v", points: [[0, 0], [1, 1]] },
      nullcline_1: { variable: "w", points: [[0, 0]] },
    } as unknown as NullclineResponse;
    expect(describePhase(run(), nullclines)?.sentence).toBe(
      "Phase portrait of w against v: v from -70 to 20, w from 0 to 2, ending at (-68, 1.5); v-nullcline 2 points, w-nullcline 1 point.",
    );
    expect(describePhase(run(), null)?.sentence).toContain("nullclines not computed");
    expect(describePhase(run({ states: { v: [1] } }), null)).toBeNull();
  });
});

describe("describePrecision", () => {
  it("states the format, the errors and where the spike trains diverge", () => {
    const prec = {
      float_result: run({ spike_count: 5 }), fixed_result: run({ spike_count: 4 }),
      arithmetic: { q_format: "Q8.8", overflow: "saturate", rounding: "nearest" },
      error: { variable: "v", max_error: 0.5, mean_error: 0.1, rms_error: 0.2, first_divergence_step: 120, trace: [] },
    } as unknown as PrecisionResponse;
    expect(describePrecision(prec).sentence).toBe(
      "Float64 reference against the Q8.8 candidate for v: largest error 0.5, mean 0.1, RMS 0.2; the spike trains " +
        "first diverge at step 120; 5 spikes in float, 4 spikes in Q8.8.",
    );
  });
});

describe("describeView", () => {
  const none: ViewResults = {
    result: null, fiResult: null, bifResult: null, heatmapResult: null, sensResult: null, staResult: null,
    freqResult: null, charResult: null, compareResult: null, networkResult: null, nullclineResult: null,
    precResult: null, multiResults: null,
  };

  it("describes the view that is drawn, and the trace only when the trace is drawn", () => {
    const fiResult = { currents: [0, 1], rates: [0, 5] } as FICurveResponse;
    expect(describeView("fi-curve", { ...none, result: run(), fiResult }, false)?.table.caption)
      .toBe("f-I curve: firing rate at each constant current");
    expect(describeView("fi-curve", { ...none, result: run(), fiResult }, true)?.sentence).toMatch(/^Trace of v, w/);
  });

  it("has nothing to describe for a view without its result", () => {
    expect(describeView("bifurcation", none, false)).toBeNull();
    expect(describeView("sta", { ...none, staResult: { time_ms: [0], average: [1], n_spikes: 3 } }, false)).toBeNull();
  });
});
