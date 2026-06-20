import { describe, expect, it } from "vitest";

import type { AnalysisResultMetadata, SimulationRunMetadata } from "./api/client";
import { buildAnalysisEvidenceItems, buildSimulationEvidenceItems } from "./plotEvidence";

const simulationMetadata: SimulationRunMetadata = {
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
};

const analysisMetadata: AnalysisResultMetadata = {
  analysis_type: "fi_curve",
  evidence_classification: "analysis",
  input_sha256: "3".repeat(64),
  output_keys: ["currents", "rates"],
  result_sha256: "4".repeat(64),
  schema_version: "studio.analysis-result.v1",
  source: "model",
};

describe("plot evidence models", () => {
  it("summarizes simulation run metadata without exposing paths", () => {
    expect(buildSimulationEvidenceItems(simulationMetadata)).toEqual([
      { label: "class", value: "simulation" },
      { label: "source", value: "ode" },
      { label: "in", value: "1111111111" },
      { label: "out", value: "2222222222" },
    ]);
  });

  it("summarizes analysis result metadata without exposing paths", () => {
    expect(buildAnalysisEvidenceItems(analysisMetadata)).toEqual([
      { label: "type", value: "fi_curve" },
      { label: "class", value: "analysis" },
      { label: "source", value: "model" },
      { label: "in", value: "3333333333" },
      { label: "out", value: "4444444444" },
    ]);
  });
});
