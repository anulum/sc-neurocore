// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio Zustand store
// Studio store composition: binds every action module to the store; the modules hold the mutations and side effects.

import { createStudioGraphActions } from "./studioGraphActions";
import { createStudioWorkspaceActions } from "./studioWorkspaceActions";
import { createStudioExportActions } from "./studioExportActions";
import { createStudioRunActions } from "./studioRunActions";
import { createStudioHardwareActions } from "./studioHardwareActions";
import { createStudioSessionAndShareActions } from "./studioSessionAndShareActions";
import { createStudioProjectActions } from "./studioProjectActions";
import { createStudioAuthActions } from "./studioAuthActions";
import { createStudioAdminActions } from "./studioAdminActions";
import { createStudioTrainingActions } from "./studioTrainingActions";
import {
  fetchStudioCapabilities,
} from "../api/client";
import {
  capabilityFailureState,
  capabilityLoadedState,
  capabilityLoadingState,
} from "../capabilityShell";
import type {
  StudioState,
  StudioStoreActions,
} from "./studioTypes";
import type { StudioEquationIrActions } from "./studioEquationIrActions";
import { studioStageOutcomeState } from "./studioStageFailure";

/**
 * Build every action the Studio store exposes.
 *
 * The actions are closures over `set` and `get` rather than methods, which is
 * what lets each one read the state as it stands when it runs rather than as
 * it stood when it was created -- an action that awaits a request must see the
 * store the user has since changed.
 *
 * **A run started by an action is deliberately not awaited.** Several actions
 * end by starting a simulation, and they mark it `void`: the caller changed a
 * parameter and the run that follows is a consequence, not a result. Awaiting
 * it would make every setter as slow as a simulation, and the run reports
 * itself through the store as it progresses.
 *
 * @param set - Writes a patch into the store.
 * @param get - Reads the store as it stands.
 * @returns Complete actions except native equation actions, which are composed
 *   separately. Every other callable state field remains required here.
 */
export function createStudioStoreActions(
  set: (partial: Partial<StudioState> | ((state: StudioState) => Partial<StudioState>)) => void,
  get: () => StudioState,
): Omit<StudioStoreActions, keyof StudioEquationIrActions> {
  const training = createStudioTrainingActions(set, get);
  const workspace = createStudioWorkspaceActions(set, get);
  const { runRestoredExperiment } = workspace;

  return {
  ...workspace.actions,

  loadCapabilities: async () => {
    set(capabilityLoadingState());
    try {
      const response = await fetchStudioCapabilities();
      set(capabilityLoadedState(response.capabilities));
    } catch (error: unknown) {
      set(capabilityFailureState(error));
    }
  },
  ...createStudioAuthActions(set, get, training),
  ...createStudioAdminActions(set, get),

  ...createStudioRunActions(set, get),

  ...createStudioHardwareActions(set, get),

  recordStageOutcome: (stage, failure) => {
    set(studioStageOutcomeState(stage, failure, get()));
  },

  ...createStudioExportActions(set, get),

  ...createStudioProjectActions(set, get, runRestoredExperiment),

  ...createStudioGraphActions(set, get),

  ...training.actions,

  ...createStudioSessionAndShareActions(set, get, runRestoredExperiment),
  };
}
