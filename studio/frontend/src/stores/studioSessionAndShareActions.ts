// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio saved session and share link actions

import { readStudioModelLink, readStudioStartupHashState } from "../studioStartupRuntime";
import { studioModelLinkDecision, studioShareLinkDecision } from "../shareLinkApplication";
import {
  studioSavedSessionRemovedState,
  studioSavedSessionRestoreState,
  captureAndStoreStudioSession,
  writeStoredStudioSessions,
} from "../studioSavedSessions";
import { studioProjectFailureState } from "../studioProjectState";
import {
  copyStudioShareUrlInRuntime,
  scheduleStudioShareStatusClear,
  studioShareStatusClearedState,
  studioShareStatusState,
} from "../studioShareRuntime";
import { compilerConfigurationInvalidatedState } from "../compilerStoreState";
import type { StudioState, StudioStoreActions } from "./studioTypes";

/** The saved session and share link actions: saving, restoring and deleting sessions kept in the browser, and reading and copying share links. */
export type StudioSessionAndShareActions = Pick<StudioStoreActions,
  "saveSession" | "loadSession" | "deleteSession" | "applyShareLink" | "shareURL">;

/**
 * Bind these actions to the Studio store.
 *
 * The saved session and share link actions: saving, restoring and deleting sessions kept in the
 * browser, and reading and copying share links. Restoring a session shows the restored experiment
 * through runRestoredExperiment. A share link opens the model it names, or applies a shared run's
 * settings and opens its model, and says so when the catalogue does not hold the model.
 *
 * @param set - Writes a patch into the store.
 * @param get - Reads the store as it stands.
 * @param runRestoredExperiment - Shows a restored experiment: loads the restored model's details
 *   without replacing the restored inputs, then runs the simulation.
 * @returns The actions.
 */
export function createStudioSessionAndShareActions(
  set: (partial: Partial<StudioState> | ((state: StudioState) => Partial<StudioState>)) => void,
  get: () => StudioState,
  runRestoredExperiment: () => void,
): StudioSessionAndShareActions {
  return {
  saveSession: (name) => {
    const s = get();
    const saved = captureAndStoreStudioSession(s, name, s.savedSessions);
    set(saved);
    return saved.sessionWriteResult;
  },

  loadSession: (name) => {
    const session = get().savedSessions.find((ss) => ss.name === name);
    if (!session) return;
    try {
      const restored = studioSavedSessionRestoreState(session.state);
      set({
        ...restored,
        ...compilerConfigurationInvalidatedState(),
      });
      runRestoredExperiment();
    } catch (error) {
      set(studioProjectFailureState(error, "Failed to restore the saved session"));
    }
  },

  deleteSession: (name) => {
    const nextState = studioSavedSessionRemovedState(get().savedSessions, name);
    const result = writeStoredStudioSessions(nextState.savedSessions);
    set({ ...nextState, sessionWriteResult: result });
    return result;
  },

  applyShareLink: async () => {
    // The Share button has always produced a link. Nothing ever read one back:
    // `readStudioStartupHashState` had no caller outside its own test, so a
    // colleague opening the link got the default Studio and no indication that
    // the link had carried anything. A link handed out that does nothing is
    // worse than no link, because the button promises otherwise.
    //
    // The judgement lives in `studioShareLinkDecision`; this only carries it out.
    const knownModels = get().models.map((model) => model.name);
    // A model link opens one model and leaves the run settings alone.
    const modelLink = studioModelLinkDecision(readStudioModelLink(), knownModels);
    if (modelLink.kind === "unknown-model") {
      set({ linkNotice: modelLink.message });
      return;
    }
    if (modelLink.kind === "select") {
      await get().selectModel(modelLink.modelName);
      return;
    }
    const decision = studioShareLinkDecision(readStudioStartupHashState(), knownModels);
    if (decision.kind === "none") return;
    if (decision.kind === "unknown-model") {
      set({ linkNotice: decision.message });
      return;
    }
    set({
      current: decision.current,
      duration: decision.duration,
      protocol: decision.protocol,
    });
    await get().selectModel(decision.modelName);
  },

  shareURL: () => {
    const s = get();
    void copyStudioShareUrlInRuntime({
      sourceMode: s.sourceMode,
      selectedModelName: s.selectedModelName,
      equations: s.equations,
      threshold: s.threshold,
      reset: s.reset,
      modelParams: s.modelParams,
      odeParams: s.odeParams,
      odeInit: s.odeInit,
      dt: s.dt,
      duration: s.duration,
      current: s.current,
      protocol: s.protocol,
    }).then((result) => {
      set(studioShareStatusState(result));
      scheduleStudioShareStatusClear(() => { set(studioShareStatusClearedState()); });
    });
  },
  };
}
