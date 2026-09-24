// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { describe, expect, it } from "vitest";

import { buildPipelineEvidenceModel, pipelineCosimSummary, pipelineFailureReason } from "./pipelineEvidence";

describe("pipeline evidence model", () => {
  it("describes completed worker-backed pipeline evidence", () => {
    expect(buildPipelineEvidenceModel({
      pipeline: "graph -> simulate -> compile -> synthesise",
      success: true,
      target: "ice40",
    })).toEqual({
      actionKind: "studio.pipeline.run",
      classification: "compile",
      evidenceArtifact: "pipeline/evidence.json",
      pipeline: "graph -> simulate -> compile -> synthesise",
      replayRoute: "POST /api/pipeline/run",
      resultArtifact: "pipeline/result.json",
      status: "completed",
      step: "complete",
      target: "ICE40",
    });
  });

  it("describes failed pipeline evidence at the failing step", () => {
    expect(buildPipelineEvidenceModel({
      error: "Compilation failed",
      step: "compile",
      success: false,
      target: "gowin",
    })).toMatchObject({
      status: "failed",
      step: "compile",
      target: "GOWIN",
    });
  });
});

describe("pipeline stop reasons", () => {
  it("joins every reason a stop carries", () => {
    expect(pipelineFailureReason({
      success: false,
      target: "ice40",
      error: "the graph cannot be lowered to hardware exactly",
      reasons: ["population a: model AdExNeuron has no hardware lowering", "population b: a Poisson drive has no hardware source"],
    })).toBe(
      "the graph cannot be lowered to hardware exactly; population a: model AdExNeuron has no hardware lowering; "
        + "population b: a Poisson drive has no hardware source",
    );
    expect(pipelineFailureReason({ success: false, target: "ice40", errors: ["no populations"] })).toBe("no populations");
    expect(pipelineFailureReason({ success: false, target: "ice40", error: "" })).toBe("unknown");
  });
});

describe("pipeline co-simulation summary", () => {
  const run = (cosimulate: unknown) => ({ success: true, target: "ice40", steps: { cosimulate } });

  it("says nothing when no co-simulation ran", () => {
    expect(pipelineCosimSummary({ success: false, target: "ice40", steps: { validate: {} } })).toBeNull();
    expect(pipelineCosimSummary({ success: false, target: "ice40" })).toBeNull();
  });

  it("states agreement with the model and with the Studio run", () => {
    expect(pipelineCosimSummary(run({
      steps: 30,
      rtl_matches_bit_true_model: true,
      studio_agreement: { identical: true, first_divergent_step: null },
    }))).toBe(
      "Co-simulation: the RTL reproduces its bit-true model on all 30 steps and spikes as the Studio's run does on every step.",
    );
  });

  it("names the first step the fixed-point hardware differs from the Studio run", () => {
    expect(pipelineCosimSummary(run({
      steps: 40,
      rtl_matches_bit_true_model: true,
      studio_agreement: { identical: false, first_divergent_step: 10 },
    }))).toContain("first differs from the Studio's run at step 10, where the fixed-point values round");
  });

  it("says plainly when the RTL is not its model", () => {
    expect(pipelineCosimSummary(run({
      steps: 40,
      rtl_matches_bit_true_model: false,
      studio_agreement: { identical: false, first_divergent_step: 0 },
    }))).toBe("Co-simulation: the RTL does not reproduce its bit-true model.");
  });
});
