// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio sign-in actions

import {
  fetchStudioAuthSession,
  loginStudioBrowserUser,
  logoutStudioBrowserUser,
  setStudioAuthToken,
} from "../api/client";
import {
  clearStoredStudioAuthToken,
  studioAuthFailureState,
  studioAuthLoadingState,
  studioAuthLogoutCompleteState,
  studioAuthLogoutFailureState,
  studioAuthSessionLoadedState,
  studioAuthUnauthenticatedState,
  storeStudioAuthToken,
  syncStoredStudioAuthToken,
} from "../studioAuthSession";
import type { StudioTrainingActionBinding } from "./studioTrainingActions";
import type { StudioState, StudioStoreActions } from "./studioTypes";

/** The sign-in actions: the stored token's session check, signing in and signing out. */
export type StudioAuthActions = Pick<StudioStoreActions,
  "loadAuthSession" | "loginBrowserUser" | "logoutBrowserUser">;

/**
 * Bind these actions to the Studio store.
 *
 * The sign-in actions: the stored token's session check, signing in and signing out. Signing in
 * loads the operator status once the session is known. Signing out forgets the followed training
 * job through the training module's release hook, and clears the stored token even when the server
 * refuses.
 *
 * @param set - Writes a patch into the store.
 * @param get - Reads the store as it stands.
 * @param training - The training binding, whose release hook stops following the training job
 *   when the user signs out.
 * @returns The actions.
 */
export function createStudioAuthActions(
  set: (partial: Partial<StudioState> | ((state: StudioState) => Partial<StudioState>)) => void,
  get: () => StudioState,
  training: Pick<StudioTrainingActionBinding, "releaseTrainingSession">,
): StudioAuthActions {
  return {
  loadAuthSession: async () => {
    const currentToken = syncStoredStudioAuthToken(setStudioAuthToken);
    if (currentToken === null) {
      set(studioAuthUnauthenticatedState());
      return;
    }
    set(studioAuthLoadingState());
    try {
      const authSession = await fetchStudioAuthSession();
      set(studioAuthSessionLoadedState(authSession));
    } catch (error: unknown) {
      clearStoredStudioAuthToken();
      syncStoredStudioAuthToken(setStudioAuthToken);
      set(studioAuthFailureState(error, "Session check failed"));
    }
  },
  loginBrowserUser: async (username, password) => {
    set(studioAuthLoadingState());
    try {
      const login = await loginStudioBrowserUser(username, password);
      storeStudioAuthToken(login.access_token);
      syncStoredStudioAuthToken(setStudioAuthToken);
      const authSession = await fetchStudioAuthSession();
      set(studioAuthSessionLoadedState(authSession));
      await get().loadOperatorStatus();
    } catch (error: unknown) {
      clearStoredStudioAuthToken();
      syncStoredStudioAuthToken(setStudioAuthToken);
      set(studioAuthFailureState(error, "Login failed"));
    }
  },
  logoutBrowserUser: async () => {
    training.releaseTrainingSession();
    set({ ...studioAuthLoadingState(), graphJobRecovery: null, graphSimResult: null });
    try {
      await logoutStudioBrowserUser();
    } catch (error: unknown) {
      set(studioAuthLogoutFailureState(error));
    } finally {
      clearStoredStudioAuthToken();
      syncStoredStudioAuthToken(setStudioAuthToken);
      set(studioAuthLogoutCompleteState());
      set({
        trainingJobs: [], trainingJobsError: null, trainingJobsLoading: false,
        trainingJobId: null, trainingStatus: "idle", trainingEpochs: [],
        trainingObservedConfig: null, trainingExperimentKey: null,
      });
    }
  },
  };
}
