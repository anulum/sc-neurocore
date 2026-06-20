import type { PipelineResult } from "./api/client";

export interface PipelineEvidenceModel {
  actionKind: string;
  classification: string;
  evidenceArtifact: string;
  pipeline: string;
  replayRoute: string;
  resultArtifact: string;
  status: "completed" | "failed";
  step: string;
  target: string;
}

export function buildPipelineEvidenceModel(result: PipelineResult): PipelineEvidenceModel {
  return {
    actionKind: "studio.pipeline.run",
    classification: "compile",
    evidenceArtifact: "pipeline/evidence.json",
    pipeline: result.pipeline ?? "graph to simulate to compile to synthesise",
    replayRoute: "POST /api/pipeline/run",
    resultArtifact: "pipeline/result.json",
    status: result.success ? "completed" : "failed",
    step: result.success ? "complete" : result.step ?? "unknown",
    target: result.target.toUpperCase(),
  };
}
