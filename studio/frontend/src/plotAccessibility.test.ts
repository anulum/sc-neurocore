// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — The plot in words and as a table

import { describe, expect, it } from "vitest";

import type { SimulateResponse } from "./api/types";
import {
  formatReading,
  multiModelDescription,
  multiModelDriveNote,
  plotDescription,
  traceDataRows,
  traceDescription,
} from "./plotAccessibility";

/**
 * A run with the fields the description reads.
 *
 * @param overrides - Fields to change.
 * @returns The run.
 */
function run(overrides: Partial<SimulateResponse> = {}): SimulateResponse {
  return {
    time: [0.1, 0.2, 0.3, 0.4],
    states: { v: [-70, -55.123456, 20, -68], w: [0, Number.NaN, 1.5, 1] },
    current_trace: [],
    spikes: [2],
    spike_count: 1,
    dt: 0.1,
    n_steps: 4,
    ...overrides,
  } as SimulateResponse;
}

describe("traceDataRows", () => {
  it("gives each variable its range and final value, leaving non-finite samples out", () => {
    expect(traceDataRows(run())).toEqual([
      { variable: "v", samples: 4, minimum: -70, maximum: 20, final: -68 },
      { variable: "w", samples: 4, minimum: 0, maximum: 1.5, final: 1 },
    ]);
  });

  it("has no range for a variable with no finite sample", () => {
    expect(traceDataRows(run({ states: { v: [Number.NaN] } }))).toEqual([
      { variable: "v", samples: 1, minimum: null, maximum: null, final: null },
    ]);
  });
});

describe("formatReading", () => {
  it("reads four significant digits, or none", () => {
    expect(formatReading(-55.123456)).toBe("-55.12");
    expect(formatReading(0.1)).toBe("0.1");
    expect(formatReading(null)).toBe("none");
  });
});

describe("traceDescription", () => {
  it("states the variables, the length, the spikes and each range in one sentence", () => {
    expect(traceDescription(run())).toBe(
      "Trace of v, w over 0.4 ms (4 steps of 0.1 ms): 1 spike. " +
        "v from -70 to 20, ending at -68; w from 0 to 1.5, ending at 1.",
    );
    expect(traceDescription(run({ spike_count: 3 }))).toContain(": 3 spikes.");
  });
});

describe("plotDescription", () => {
  it("describes the trace in numbers and any other view by what it is", () => {
    expect(plotDescription("Trace", run(), true)).toBe(traceDescription(run()));
    expect(plotDescription("Trace", null, false)).toBe("Trace plot: nothing has run yet.");
  });

  it("does not send the reader to exports that hold only the trace", () => {
    // The CSV and JSON exports write the trace run; the sentence used to say
    // they held every view's values.
    const sentence = plotDescription("Bifurcation", run(), false);
    expect(sentence).toBe(
      "Bifurcation plot. This view is not put into words; the data table and the CSV and " +
        "JSON exports hold the trace run, not this view.",
    );
    expect(sentence).not.toContain("Its values are in");
  });

  it("uses the view's own sentence when it has one", () => {
    expect(plotDescription("Multi-model", run(), false, "Multi-model overlay of 2 runs.")).toBe(
      "Multi-model overlay of 2 runs.",
    );
  });
});

describe("multiModelDriveNote", () => {
  /**
   * A run with a recorded drive.
   *
   * @param name - The model.
   * @param current - The drive's current.
   * @returns The run.
   */
  function driven(name: string, current: number): SimulateResponse {
    return run({
      model_name: name,
      experiment: { protocol: { kind: "constant", current, frequency_hz: null } },
    } as unknown as Partial<SimulateResponse>);
  }

  it("says the one drive every model had, in each model's own units", () => {
    expect(multiModelDriveNote([driven("AdExNeuron", 10), driven("LIF", 10)])).toBe(
      "Every model had the same drive (constant, I = 10), read in each model's own current units.",
    );
  });

  it("names each model's drive when they differ", () => {
    expect(multiModelDriveNote([driven("AdExNeuron", 10), driven("LIF", 2)])).toBe(
      "The drives differ (AdExNeuron: constant, I = 10; LIF: constant, I = 2), " +
        "read in each model's own current units.",
    );
  });

  it("says nothing it cannot read from the runs", () => {
    expect(multiModelDriveNote([run()])).toBeNull();
    expect(multiModelDriveNote([])).toBeNull();
  });
});

describe("multiModelDescription", () => {
  it("states each model's first state, spikes and rate, and the time steps", () => {
    const a = run({ model_name: "AdExNeuron", stats: { rate_hz: 12 } } as Partial<SimulateResponse>);
    const b = run({ model_name: "", spike_count: 3, stats: { rate_hz: 30 } } as Partial<SimulateResponse>);
    expect(multiModelDescription([a, b])).toBe(
      "Multi-model overlay of 2 runs, each model's first state on one shared axis. " +
        "AdExNeuron: v from -70 to 20, 1 spike (12 Hz); Model 2: v from -70 to 20, 3 spikes (30 Hz). " +
        "Time steps: AdExNeuron 0.1 ms, Model 2 0.1 ms.",
    );
  });
});
