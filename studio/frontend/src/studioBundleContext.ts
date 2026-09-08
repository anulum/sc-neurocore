// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Local evidence export context

import { canonicalSealText } from "./evidenceSeal";
import { evidenceBundleSurfaceKeys, type EvidenceBundleSurface } from "./evidenceBundles";
import { studioProjectSaveState } from "./studioProjectState";
import { studioExperimentKey } from "./studioExperimentKey";
import { studioSimulationConfigInput, type StudioSimulationConfigSource } from "./studioSimulationConfigInput";
import type { StudioProjectSnapshotInput } from "./studioProjectState";
import type { StudioState } from "./stores/studioTypes";

/** Local receipt binding; this is not independent server-manifest attestation. */
export interface StudioBundleContext { key: string; bundleId: string; jobId: string }

/** Explicit inputs shared by export acceptance and completion projections. */
export type StudioBundleContextSource = StudioSimulationConfigSource & StudioProjectSnapshotInput & Pick<StudioState,
  "projectRevision" | "resultExperimentKey" | "analysisExperimentKey" | "trainingExperimentKey"
  | "verilogSrc" | "svSource" | "compileTraceability" | "cosimResult" | "latestSynthesisJobId"
  | "latestMultiTargetSynthesisJobId" | "bundleContexts" | "projectEvidenceBundle"
  | "compileEvidenceBundle" | "synthesisEvidenceBundle">;

/**
 * Return the canonical local surface identity, excluding transport flags.
 *
 * @param surface - Export slot whose inputs are identified.
 * @param state - Current surface inputs.
 * @returns Local comparison key, not a cryptographic server receipt.
 */
export function studioBundleContextKey(surface: EvidenceBundleSurface, state: StudioBundleContextSource): string {
  if (surface === "admin") return "admin";
  if (surface === "project") return canonicalSealText({
    project: studioProjectSaveState(state), revision: state.projectRevision,
    experiment: studioExperimentKey(studioSimulationConfigInput(state)),
    simulation: state.resultExperimentKey, analysis: state.analysisExperimentKey,
    training: state.trainingExperimentKey,
  });
  return canonicalSealText({ source: state.sourceMode, rtl: state.verilogSrc, sv: state.svSource,
    trace: state.compileTraceability,
    ...(surface === "synthesis" ? { target: state.synthTarget, parity: state.cosimResult,
      single: state.latestSynthesisJobId, multi: state.latestMultiTargetSynthesisJobId } : {}),
  });
}

/**
 * Qualify a scoped export only for its captured context and exact receipt IDs.
 *
 * @param surface - Project, compile or synthesis export slot.
 * @param state - Live inputs, receipt and captured context.
 * @returns False for stale, missing or invalid context; history stays untouched.
 */
export function studioBundleIsCurrent(surface: Exclude<EvidenceBundleSurface, "admin">, state: StudioBundleContextSource): boolean {
  const bundle = state[evidenceBundleSurfaceKeys(surface).bundle];
  const context = state.bundleContexts[surface];
  if (!bundle || !context?.bundleId || !context.jobId
    || context.bundleId !== bundle.bundle_id || context.jobId !== bundle.job_id) return false;
  try { return context.key === studioBundleContextKey(surface, state); }
  catch { return false; }
}
