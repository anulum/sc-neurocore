// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Panel analysis to store ownership boundary

import { studioStageOutcomeState } from "./stores/studioStageFailure";
import type { StudioState } from "./stores/studioTypes";
import type { StudioAnalysisJobIntegrationPatch } from "./useStudioAnalysisJobIntegration";

/**
 * Apply panel-owned evidence without altering store-owned request diagnostics.
 *
 * Panel jobs track busy state in their own session. Shared result sinks also
 * serve exclusive store actions, so their isSimulating/error patches are not
 * owned here. Panel failures remain in the panel session's own error state,
 * and are also remembered as a failed analysis stage for the current
 * experiment; a delivered result withdraws that. Preserve live values
 * synchronously, not a render-time snapshot.
 *
 * @param patch - Validated panel result or diagnostic and context keys.
 * @param get - Read live store request state.
 * @param set - Apply evidence while preserving the current request's busy flag.
 */
export function applyStudioPanelAnalysisPatch(
  patch: StudioAnalysisJobIntegrationPatch,
  get: () => StudioState, set: (patch: Partial<StudioState>) => void,
): void {
  const state = get();
  const failure = "error" in patch && typeof patch.error === "string" ? patch.error : null;
  const delivered = typeof patch.analysisExperimentKey === "string";
  const stage = failure !== null || delivered
    ? studioStageOutcomeState("analyse", failure, state)
    : {};
  set({ ...patch, ...stage, isSimulating: state.isSimulating, error: state.error });
}
