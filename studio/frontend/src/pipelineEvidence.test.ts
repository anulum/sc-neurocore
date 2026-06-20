import { describe, expect, it } from "vitest";

import { buildPipelineEvidenceModel } from "./pipelineEvidence";

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
