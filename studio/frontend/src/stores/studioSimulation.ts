// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Simulation outcome ownership

import { simulateModel, simulateODE } from "../api/client";
import { studioExperimentKey } from "../studioExperimentKey";
import { studioSimulationConfigInput } from "../studioSimulationConfigInput";
import { studioSimulationConfig } from "../studioSimulationConfig";
import { studioAnalysisStartState, studioAnalysisIdleState,
  studioAnalysisFailureState, studioSimulationResultState } from "../studioAnalysisState";
import type { StudioState } from "./studioTypes";

/**
 * Accept simulation outcomes only for the experiment that submitted them.
 *
 * Retain historical traces while withdrawing completion on a rerun. Invalid
 * input is reported through store state, including for fire-and-forget callers.
 * Busy invocations do not change the active request or its evidence.
 *
 * @param get - Read live experiment inputs across the transport await.
 * @param set - Apply owned result, error and busy-state patches.
 */
export async function runStoreSimulation(
  get: () => StudioState, set: (patch: Partial<StudioState>) => void,
): Promise<void> {
  const state = get();
  if (state.isSimulating) return;
  set({ ...studioAnalysisStartState(), resultExperimentKey: null });
  let key: string | null = null;
  try {
    const input = studioSimulationConfigInput(state);
    key = studioExperimentKey(input);
    const config = studioSimulationConfig(input);
    const result = state.sourceMode === "model" && state.selectedModelName
      ? await simulateModel(config) : await simulateODE(config);
    if (studioExperimentKey(studioSimulationConfigInput(get())) !== key) {
      set(studioAnalysisIdleState()); return;
    }
    set({ ...studioSimulationResultState(result), resultExperimentKey: key });
  } catch (error: unknown) {
    let failure = error;
    try {
      if (key !== null && studioExperimentKey(studioSimulationConfigInput(get())) !== key) {
        set(studioAnalysisIdleState()); return;
      }
    } catch (inputError: unknown) { failure = inputError; }
    set(studioAnalysisFailureState(failure));
  }
}
