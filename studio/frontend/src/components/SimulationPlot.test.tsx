import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { FICurveResponse, SimulateResponse } from "../api/client";
import SimulationPlot from "./SimulationPlot";

const mockStore = vi.hoisted(() => ({
  state: {} as Record<string, unknown>,
}));

vi.mock("../stores/studio", () => ({
  useStudioStore: () => mockStore.state,
}));

const simulationResult: SimulateResponse = {
  current_trace: [1, 1],
  dt: 0.1,
  model_name: "custom",
  n_steps: 2,
  run_metadata: {
    dt: 0.1,
    evidence_classification: "simulation",
    input_sha256: "1".repeat(64),
    n_steps: 2,
    result_sha256: "2".repeat(64),
    sample_count: 2,
    schema_version: "studio.simulation-run.v1",
    source: "ode",
    spike_count: 0,
    state_variables: ["v"],
  },
  spike_count: 0,
  spikes: [],
  states: { v: [0, 0.1] },
  stats: {
    isi_cv: null,
    isi_histogram: null,
    isi_mean_ms: null,
    rate_hz: 0,
  },
  time: [0, 0.1],
};

const fiResult: FICurveResponse = {
  analysis_metadata: {
    analysis_type: "fi_curve",
    evidence_classification: "analysis",
    input_sha256: "3".repeat(64),
    output_keys: ["currents", "rates"],
    result_sha256: "4".repeat(64),
    schema_version: "studio.analysis-result.v1",
    source: "ode",
  },
  currents: [0, 1],
  rates: [0, 10],
};

function resetStore(overrides: Record<string, unknown>): void {
  mockStore.state = {
    activeTab: "trace",
    bifResult: null,
    charResult: null,
    compareResult: null,
    fiResult: null,
    freqResult: null,
    heatmapResult: null,
    importedTrace: null,
    multiResults: null,
    networkResult: null,
    nullclineResult: null,
    precResult: null,
    result: simulationResult,
    runSimulation: async () => undefined,
    sensResult: null,
    ...overrides,
  };
}

describe("SimulationPlot", () => {
  beforeEach(() => resetStore({}));

  it("renders simulation evidence metadata on the trace view", () => {
    const html = renderToStaticMarkup(<SimulationPlot />);

    expect(html).toContain("class simulation");
    expect(html).toContain("ode");
    expect(html).toContain("in 1111111111");
    expect(html).toContain("out 2222222222");
  });

  it("renders analysis evidence metadata on analysis views", () => {
    resetStore({ activeTab: "fi-curve", fiResult });

    const html = renderToStaticMarkup(<SimulationPlot />);

    expect(html).toContain("fi_curve");
    expect(html).toContain("class analysis");
    expect(html).toContain("in 3333333333");
    expect(html).toContain("out 4444444444");
  });
});
