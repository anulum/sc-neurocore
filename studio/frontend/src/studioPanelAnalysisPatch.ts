// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Panel analysis to store ownership boundary

import type { StudioState } from "./stores/studioTypes";
import type { StudioAnalysisJobIntegrationPatch } from "./useStudioAnalysisJobIntegration";

/**
 * Apply panel-owned evidence without altering store-owned request diagnostics.
 *
 * Panel jobs track busy state in their own session. Shared result sinks also
 * serve exclusive store actions, so their isSimulating/error patches are not
 * owned here. Panel failures remain in the panel session's own error state.
 * Preserve live values synchronously, not a render-time snapshot.
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
  set({ ...patch, isSimulating: state.isSimulating, error: state.error });
}
