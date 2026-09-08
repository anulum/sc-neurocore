// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Direct analysis lifecycle and experiment ownership

import { studioAnalysisFailureState, studioAnalysisStartState } from "../studioAnalysisState";
import { studioExperimentKey } from "../studioExperimentKey";
import { studioSimulationConfigInput } from "../studioSimulationConfigInput";
import type { StudioState } from "./studioTypes";

/**
 * Run an analysis, retaining plots but accepting only current outcomes.
 *
 * Busy requests leave evidence unchanged. Accepted requests withdraw completion
 * until success. Obsolete responses cannot replace current results or errors.
 * This owns request lifecycle, not analysis algorithms or request construction.
 *
 * @param get - Read the current store, including changes during the await.
 * @param set - Apply patches without replacing unrelated state.
 * @param request - Build and execute the analysis from its captured state.
 * @param tab - Optional destination tab when starting the analysis.
 * @param identity - Request-specific identity, including any extra inputs.
 */
export async function runStoreDirectAnalysis(
  get: () => StudioState,
  set: (patch: Partial<StudioState>) => void,
  request: (state: StudioState) => Promise<Partial<StudioState>>,
  tab?: Parameters<typeof studioAnalysisStartState>[0],
  identity: (state: StudioState) => string = (snapshot) => studioExperimentKey(studioSimulationConfigInput(snapshot)),
): Promise<void> {
  const state = get();
  if (state.isSimulating) return;
  set({ ...studioAnalysisStartState(tab), analysisExperimentKey: null });
  let requestedKey: string | null = null;
  try {
    requestedKey = identity(state);
    const patch = await request(state);
    if (identity(get()) !== requestedKey) {
      set({ isSimulating: false });
      return;
    }
    set({ ...patch, error: null, isSimulating: false, analysisExperimentKey: requestedKey });
  } catch (error: unknown) {
    let failure = error;
    try {
      if (requestedKey !== null
        && identity(get()) !== requestedKey) {
        set({ isSimulating: false });
        return;
      }
    } catch (currentInputError: unknown) {
      failure = currentInputError;
    }
    set({ ...studioAnalysisFailureState(failure), analysisExperimentKey: null });
  }
}
