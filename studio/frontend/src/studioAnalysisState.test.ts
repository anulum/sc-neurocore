// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio analysis store state helper tests
import { describe, expect, it } from "vitest";

import type {
  FICurveResponse,
  ImportedTrace,
  NetworkResult,
  SimulateResponse,
} from "./api/client";
import {
  studioAnalysisErrorState,
  studioAnalysisFailureState,
  studioAnalysisIdleState,
  studioAnalysisStartState,
  studioCodegenResultState,
  studioCodegenStartState,
  studioFICurveResultState,
  studioImportedTraceState,
  studioMultiResultsState,
  studioNetworkResultState,
  studioSimulationResultState,
  studioSTAResultState,
} from "./studioAnalysisState";

function simulationResult(overrides: Partial<SimulateResponse> = {}): SimulateResponse {
  return {
    current_trace: overrides.current_trace ?? [10, 10, 10],
    dt: overrides.dt ?? 1,
    model_name: overrides.model_name ?? "lif",
    n_steps: overrides.n_steps ?? 3,
    run_metadata: overrides.run_metadata ?? {
      dt: 1,
      evidence_classification: "simulation",
      input_sha256: "a".repeat(64),
      n_steps: 3,
      result_sha256: "b".repeat(64),
      sample_count: 3,
      schema_version: "studio.simulation-run.v1",
      source: "model",
      spike_count: 0,
      state_variables: ["v"],
    },
    spike_count: overrides.spike_count ?? 0,
    spikes: overrides.spikes ?? [],
    states: overrides.states ?? { v: [-65, -64, -63] },
    stats: overrides.stats ?? {
      isi_cv: null,
      isi_histogram: null,
      isi_mean_ms: null,
      rate_hz: 0,
    },
    time: overrides.time ?? [0, 1, 2],
  };
}

describe("studio analysis state helpers", () => {
  it("builds shared start, failure, and idle patches", () => {
    expect(studioAnalysisStartState()).toEqual({ error: null, isSimulating: true });
    expect(studioAnalysisStartState("fi-curve")).toEqual({
      activeTab: "fi-curve",
      error: null,
      isSimulating: true,
    });
    expect(studioAnalysisFailureState(new Error("solver offline"))).toEqual({
      error: "solver offline",
      isSimulating: false,
    });
    expect(studioAnalysisErrorState("ODE mode required")).toEqual({ error: "ODE mode required" });
    expect(studioAnalysisIdleState()).toEqual({ isSimulating: false });
  });

  it("builds simulation and analysis result patches", () => {
    const result = simulationResult();
    const fiResult: FICurveResponse = {
      analysis_metadata: {
        analysis_type: "fi_curve",
        evidence_classification: "analysis",
        input_sha256: "a".repeat(64),
        output_keys: ["rates"],
        result_sha256: "b".repeat(64),
        schema_version: "studio.analysis-result.v1",
        source: "model",
      },
      currents: [0, 10],
      rates: [0, 20],
    };
    const networkResult: NetworkResult = {
      dt: 0.1,
      duration: 100,
      exc_rates: [5],
      inh_rates: [3],
      mean_exc_rate: 5,
      mean_inh_rate: 3,
      n_exc: 1,
      n_inh: 1,
      n_spikes: 1,
      n_total: 2,
      rate_time: [0],
      spike_neurons: [0],
      spike_times: [1],
    };
    const importedTrace: ImportedTrace = {
      dt: 0.1,
      spike_count: 0,
      spikes: [],
      stats: {
        max: -65,
        mean: -65,
        min: -65,
        std: 0,
        threshold_estimate: -55,
      },
      time: [0],
      voltage: [-65],
    };

    expect(studioSimulationResultState(result)).toEqual({
      isSimulating: false,
      result,
    });
    expect(studioFICurveResultState(fiResult)).toEqual({
      fiResult,
      isSimulating: false,
    });
    expect(studioMultiResultsState([result])).toEqual({
      isSimulating: false,
      multiResults: [result],
    });
    expect(studioNetworkResultState(networkResult)).toEqual({
      isSimulating: false,
      networkResult,
    });
    expect(studioImportedTraceState(importedTrace)).toEqual({
      activeTab: "trace",
      importedTrace,
    });
  });

  it("builds code generation patches", () => {
    expect(studioCodegenStartState()).toEqual({ activeTab: "code" });
    expect(studioCodegenResultState("print('run')", "sc-neurocore run")).toEqual({
      codeOneliner: "sc-neurocore run",
      codeScript: "print('run')",
    });
  });

  it("computes spike-triggered average state from a valid trace", () => {
    const values = Array.from({ length: 30 }, (_, index) => index);
    const result = simulationResult({
      dt: 5,
      spikes: [3, 10, 20],
      states: { v: values },
      time: values,
    });

    expect(studioSTAResultState(result)).toEqual({
      activeTab: "sta",
      staResult: {
        average: [9, 10, 11, 12],
        n_spikes: 3,
        time_ms: [-10, -5, 0, 5],
      },
    });
  });

  it("returns null when STA cannot be computed", () => {
    expect(studioSTAResultState(simulationResult({ spikes: [1, 2] }))).toBeNull();
    expect(studioSTAResultState(simulationResult({ spikes: [3, 10, 20], states: {} }))).toBeNull();
    expect(studioSTAResultState(simulationResult({
      dt: 0.1,
      spikes: [3, 10, 20],
      states: { v: [0, 1] },
    }))).toBeNull();
  });
});
