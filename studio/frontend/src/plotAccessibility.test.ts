// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — The plot in words and as a table

import { describe, expect, it } from "vitest";

import type { SimulateResponse } from "./api/types";
import { formatReading, plotDescription, traceDataRows, traceDescription } from "./plotAccessibility";

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
    expect(plotDescription("Bifurcation", run(), false)).toBe(
      "Bifurcation plot. Its values are in the CSV and JSON exports; the data table describes the trace.",
    );
    expect(plotDescription("Trace", null, false)).toBe("Trace plot: nothing has run yet.");
  });
});
