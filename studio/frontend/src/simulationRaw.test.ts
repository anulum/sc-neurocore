// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio raw-trace accessor tests

import { describe, expect, it } from "vitest";

import type { SimulateResponse } from "./api/client";
import {
  displayPositionAtTime,
  fullDriveTrace,
  fullSampleTimes,
  fullStateNames,
  fullStateTrace,
  hasRawTraces,
  rawStepAtTime,
} from "./simulationRaw";

const rawV = [0, 1, 2, 3, 4, 5, 6, 7];
const rawDrive = [1, 1, 1, 1, 2, 2, 2, 2];

const custodyResult: SimulateResponse = {
  time: [0.1, 0.4, 0.8],
  states: { v: [0, 3, 7] },
  current_trace: [1, 1, 2],
  spikes: [3],
  spike_count: 1,
  stats: { isi_cv: null, isi_histogram: null, isi_mean_ms: null, rate_hz: 0 },
  dt: 0.1,
  n_steps: 8,
  run_metadata: {
    dt: 0.1,
    evidence_classification: "simulation",
    input_sha256: "1".repeat(64),
    n_steps: 8,
    result_sha256: "2".repeat(64),
    sample_count: 3,
    schema_version: "studio.simulation-run.v2",
    source: "model",
    spike_count: 1,
    status: "completed",
    state_variables: ["v"],
  },
  raw: {
    schema_version: "studio.raw-trace.v1",
    included: true,
    element_count: 16,
    element_budget: 2_000_000,
    dt: 0.1,
    n_steps: 8,
    sample_time_ms: "(index + 1) * dt",
    drive_interval_ms: "[index * dt, (index + 1) * dt)",
    spike_indices: [3],
    spike_times_ms: [0.4],
    vector_snapshots_only: [],
    states: { v: rawV },
    drive: rawDrive,
  },
  display: {
    schema_version: "studio.display-projection.v1",
    method: "bucket-extrema",
    max_points: 3,
    bucket_count: 1,
    point_count: 3,
    sample_index: [0, 3, 7],
    first_sample_included: true,
    final_sample_included: true,
    spikes_are_raw_steps: true,
  },
};

const legacyResult: SimulateResponse = {
  ...custodyResult,
  raw: undefined,
  display: undefined,
  time: [0.1, 0.2, 0.3],
  states: { v: [0, 1, 2] },
  current_trace: [1, 1, 1],
  n_steps: 3,
};

describe("simulation raw-trace accessors", () => {
  it("reads full-resolution traces from the raw block", () => {
    expect(hasRawTraces(custodyResult)).toBe(true);
    expect(fullStateNames(custodyResult)).toEqual(["v"]);
    expect(fullStateTrace(custodyResult, "v")).toEqual(rawV);
    expect(fullDriveTrace(custodyResult)).toEqual(rawDrive);
    expect(fullSampleTimes(custodyResult).map((t) => Number(t.toFixed(6)))).toEqual([
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
    ]);
  });

  it("falls back to the carried arrays for a legacy result", () => {
    expect(hasRawTraces(legacyResult)).toBe(false);
    expect(fullStateTrace(legacyResult, "v")).toEqual([0, 1, 2]);
    expect(fullDriveTrace(legacyResult)).toEqual([1, 1, 1]);
    expect(fullSampleTimes(legacyResult)).toEqual([0.1, 0.2, 0.3]);
  });

  it("maps a cursor time to the nearest display position and its raw step", () => {
    expect(displayPositionAtTime(custodyResult, 0.05)).toBe(0);
    expect(displayPositionAtTime(custodyResult, 0.45)).toBe(1);
    expect(displayPositionAtTime(custodyResult, 0.79)).toBe(2);
    expect(rawStepAtTime(custodyResult, 0.45)).toBe(3);
    expect(rawStepAtTime(custodyResult, 5)).toBe(7);
    expect(rawStepAtTime(legacyResult, 0.2)).toBe(1);
    expect(rawStepAtTime(legacyResult, 0.0)).toBe(0);
  });
});
