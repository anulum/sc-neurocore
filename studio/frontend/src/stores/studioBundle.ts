// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Evidence bundle asynchronous ownership

import { createStudioEvidenceBundle, fetchStudioOperatorStatus, fetchStudioJobs } from "../api/client";
import { evidenceBundleSurfaceKeys, adminEvidenceBundleCreatedState,
  scopedEvidenceBundleCreatedState, type EvidenceBundleSurface } from "../evidenceBundles";
import { canonicalSealText } from "../evidenceSeal";
import { studioProjectSaveState } from "../studioProjectState";
import { studioExperimentKey } from "../studioExperimentKey";
import { studioSimulationConfigInput } from "../studioSimulationConfigInput";
import type { StudioState } from "./studioTypes";

/**
 * Identify the live surface whose export completion may be displayed.
 *
 * @param surface - Independent bundle slot.
 * @param state - State defining that surface, excluding bundle transport flags.
 * @returns Canonical local identity, not a server evidence digest.
 */
function surfaceKey(surface: EvidenceBundleSurface, state: StudioState): string {
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
 * Export one surface, accepting completion only while it remains current.
 *
 * Each slot owns its loading flag independently. Reruns withdraw old completion;
 * stale results or failures cannot restore it after source invalidation. Admin
 * exports do not depend on the experiment selected on another panel.
 *
 * @param surface - Surface receiving the bundle.
 * @param request - Explicit evidence collection request.
 * @param get - Read live state across both export and operator-refresh awaits.
 * @param set - Apply owned patches without replacing unrelated state.
 */
export async function runStoreBundle(
  surface: EvidenceBundleSurface, request: Parameters<StudioState["createEvidenceBundle"]>[0],
  get: () => StudioState, set: (patch: Partial<StudioState>) => void,
): Promise<void> {
  const keys = surface === "admin" ? { bundle: "evidenceBundle", error: "evidenceBundleError", loading: "evidenceBundleLoading" } as const
    : evidenceBundleSurfaceKeys(surface);
  if (get()[keys.loading]) return;
  set({ [keys.bundle]: null, [keys.error]: null, [keys.loading]: true });
  let key: string | null = null;
  try {
    key = surfaceKey(surface, get());
    const bundle = await createStudioEvidenceBundle(request);
    const [operator, jobs] = await Promise.all([fetchStudioOperatorStatus(), fetchStudioJobs()]);
    if (surfaceKey(surface, get()) !== key) { set({ [keys.loading]: false }); return; }
    set(surface === "admin" ? adminEvidenceBundleCreatedState(bundle, operator, jobs)
      : scopedEvidenceBundleCreatedState(surface, bundle, operator, jobs));
  } catch (error: unknown) {
    let failure = error;
    try {
      if (key !== null && surfaceKey(surface, get()) !== key) { set({ [keys.loading]: false }); return; }
    } catch (currentInputError: unknown) { failure = currentInputError; }
    set({ [keys.loading]: false, [keys.error]: failure instanceof Error ? failure.message : String(failure) });
  }
}
