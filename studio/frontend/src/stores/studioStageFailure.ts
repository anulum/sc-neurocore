// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Which workflow stage last failed, and for which experiment

/**
 * Remembering that a stage failed, so the guided workflow can say so.
 *
 * The store's `error` is one message for whatever ran last; it cannot tell the
 * workflow that compilation failed while simulation succeeded. A stage failure
 * names the stage and the experiment it failed under. The workflow shows it
 * only while that experiment is still the current one, so a failure never
 * outlives the inputs it was about, and a new attempt at the same stage
 * withdraws it. Superseded runs end without an error and are never recorded:
 * the owners already refuse outcomes for inputs the reader has since changed.
 */

import type { GuidedFlowStepKey } from "../guidedFlowState";
import { studioExperimentKey } from "../studioExperimentKey";
import {
  studioSimulationConfigInput,
  type StudioSimulationConfigSource,
} from "../studioSimulationConfigInput";
import type { StudioState } from "./studioTypes";

/** The latest failed attempt at one workflow stage. */
export interface StudioStageFailure {
  stage: GuidedFlowStepKey;
  message: string;
  /** The experiment it failed under, or `null` when its inputs had no identity. */
  experimentKey: string | null;
}

/** What the stage-failure helpers read from the store. */
export interface StudioStageFailureSource extends StudioSimulationConfigSource {
  stageFailure: StudioStageFailure | null;
}

/**
 * Identify the experiment the store currently describes.
 *
 * @param state - The store's current state.
 * @returns Its key, or `null` when the inputs cannot be identified; a
 *   failure recorded under `null` then matches no experiment and is not shown.
 */
export function studioStageExperimentKey(state: StudioSimulationConfigSource): string | null {
  try {
    return studioExperimentKey(studioSimulationConfigInput(state));
  } catch {
    return null;
  }
}

/**
 * Record how an attempt at a stage ended.
 *
 * @param stage - The stage that ran.
 * @param failure - Its failure message, or `null` when it succeeded.
 * @param state - The store as the attempt ended.
 * @returns The patch: the new failure, or the old one withdrawn when it was
 *   this stage's and the stage has now succeeded.
 */
export function studioStageOutcomeState(
  stage: GuidedFlowStepKey,
  failure: string | null,
  state: StudioStageFailureSource,
): Pick<StudioState, "stageFailure"> {
  if (failure !== null) {
    return {
      stageFailure: { stage, message: failure, experimentKey: studioStageExperimentKey(state) },
    };
  }
  return { stageFailure: state.stageFailure?.stage === stage ? null : state.stageFailure };
}

/**
 * Wrap a stage owner's patch sink so its outcomes are remembered.
 *
 * A patch that starts a run withdraws this stage's earlier failure. A patch
 * that ends a run with an error records it against the current experiment:
 * owners emit such a patch only after confirming the inputs have not changed.
 * Every other patch passes through untouched.
 *
 * @param stage - The stage the owner runs.
 * @param get - Read the store as each patch arrives.
 * @param set - The owner's original patch sink.
 * @returns The wrapped sink.
 */
export function studioStageSet(
  stage: GuidedFlowStepKey,
  get: () => StudioState,
  set: (patch: Partial<StudioState>) => void,
): (patch: Partial<StudioState>) => void {
  return (patch) => {
    if (patch.isSimulating === true) {
      set({ ...patch, ...studioStageOutcomeState(stage, null, get()) });
      return;
    }
    if (patch.isSimulating === false && typeof patch.error === "string") {
      set({ ...patch, ...studioStageOutcomeState(stage, patch.error, get()) });
      return;
    }
    set(patch);
  };
}
