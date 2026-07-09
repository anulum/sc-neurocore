// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

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
