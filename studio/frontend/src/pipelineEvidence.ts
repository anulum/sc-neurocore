// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import type { PipelineCosimStep, PipelineResult } from "./api/client";

/**
 * What the evidence header states about one pipeline run: what was run, on
 * what, how it ended, and where the artefacts recording it are.
 */
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

/**
 * Describe a pipeline run for its evidence header.
 *
 * A failed run still has a step: the one it stopped at. A run that
 * reports no step at all is shown as `unknown` rather than as blank,
 * so the gap is visible to the reader.
 *
 * @param result - The run, as the server reported it.
 * @returns What the header should state.
 */
export function buildPipelineEvidenceModel(result: PipelineResult): PipelineEvidenceModel {
  return {
    actionKind: "studio.pipeline.run",
    classification: "compile",
    evidenceArtifact: "pipeline/evidence.json",
    pipeline: result.pipeline ?? "graph → simulate → lower → co-simulate → synthesise",
    replayRoute: "POST /api/pipeline/run",
    resultArtifact: "pipeline/result.json",
    status: result.success ? "completed" : "failed",
    step: result.success ? "complete" : result.step ?? "unknown",
    target: result.target.toUpperCase(),
  };
}

/**
 * Say why a pipeline run stopped, with every reason it carries.
 *
 * @param result - The run, as the server reported it.
 * @returns The reasons, joined, or `unknown` when the run named none.
 */
export function pipelineFailureReason(result: PipelineResult): string {
  const parts = [...(result.errors ?? []), result.error, ...(result.reasons ?? [])];
  const stated = parts.filter((part): part is string => typeof part === "string" && part !== "");
  return stated.length > 0 ? stated.join("; ") : "unknown";
}

/**
 * Say what the co-simulation established, when the run got that far.
 *
 * When the RTL reproduces its bit-true model, a difference from the Studio's
 * run comes from the fixed-point values alone, and the first differing step
 * is named rather than left out.
 *
 * @param result - The run, as the server reported it.
 * @returns The sentence, or `null` when no co-simulation ran.
 */
export function pipelineCosimSummary(result: PipelineResult): string | null {
  const cosim = result.steps?.cosimulate as PipelineCosimStep | undefined;
  if (cosim === undefined) return null;
  if (!cosim.rtl_matches_bit_true_model) {
    return "Co-simulation: the RTL does not reproduce its bit-true model.";
  }
  const studio = cosim.studio_agreement.identical
    ? "and spikes as the Studio's run does on every step"
    : `and first differs from the Studio's run at step ${String(cosim.studio_agreement.first_divergent_step)}, where the fixed-point values round`;
  return `Co-simulation: the RTL reproduces its bit-true model on all ${String(cosim.steps)} steps ${studio}.`;
}
