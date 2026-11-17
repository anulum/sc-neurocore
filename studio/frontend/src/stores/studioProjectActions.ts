// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio server project actions

import {
  branchRefusedEdit as apiBranchRefusedEdit,
  saveProject as apiSaveProject,
  loadProject as apiLoadProject,
  listDeletedProjects as apiListDeletedProjects,
  restoreProject as apiRestoreProject,
  listProjects as apiListProjects,
  deleteProject as apiDeleteProject,
} from "../api/client";
import {
  studioProjectSaveState,
  studioProjectDeletedListedState,
  studioProjectExpectedRevision,
  studioProjectFailureState,
  studioProjectListLoadedState,
  studioProjectRestoreState,
  studioProjectRevisionFromLoadResponse,
  studioProjectSaveFailureState,
  studioProjectSavedState,
  studioProjectStateFromLoadResponse,
} from "../studioProjectState";
import { compilerConfigurationInvalidatedState } from "../compilerStoreState";
import type { StudioState, StudioStoreActions } from "./studioTypes";

/** The server project actions: saving, branching a refused edit, loading, listing, deleting and restoring projects. */
export type StudioProjectActions = Pick<StudioStoreActions,
  "saveProjectToServer" | "keepRefusedEdit" | "loadProjectFromServer" | "listServerProjects" | "deleteServerProject" | "listDeletedServerProjects" | "restoreDeletedServerProject">;

/**
 * Bind these actions to the Studio store.
 *
 * The server project actions: saving, branching a refused edit, loading, listing, deleting and
 * restoring projects. A save states the revision it was based on, and a save the server refuses as
 * stale keeps the edit so that it can be branched. Loading a project restores its state and shows
 * the restored experiment through runRestoredExperiment; a delete or a restore refreshes both the
 * list and the trash.
 *
 * @param set - Writes a patch into the store.
 * @param get - Reads the store as it stands.
 * @param runRestoredExperiment - Shows a restored experiment: loads the restored model's details
 *   without replacing the restored inputs, then runs the simulation.
 * @returns The actions.
 */
export function createStudioProjectActions(
  set: (partial: Partial<StudioState> | ((state: StudioState) => Partial<StudioState>)) => void,
  get: () => StudioState,
  runRestoredExperiment: () => void,
): StudioProjectActions {
  return {
  saveProjectToServer: async (name) => {
    const expected = studioProjectExpectedRevision(get().projectRevision, name);
    try {
      const state = studioProjectSaveState(get());
      const projectSaveResult = await apiSaveProject(name, state, expected);
      set(studioProjectSavedState(projectSaveResult));
      await get().listServerProjects();
    } catch (e) { set(studioProjectSaveFailureState(e, name, expected)); }
  },

  keepRefusedEdit: async () => {
    // Only reachable after a save conflict, which is the only thing that sets
    // `refusedEdit`. The state sent is the one still held here — the edit that
    // was refused — not a reload of what won.
    const pending = get().refusedEdit;
    if (!pending) return;
    try {
      const branch = await apiBranchRefusedEdit(
        pending.name,
        studioProjectSaveState(get()),
        pending.baseRevision,
      );
      set({
        error: `Kept as "${branch.branched}". Reload ${pending.name} to see the other edit.`,
        refusedEdit: null,
      });
      await get().listServerProjects();
    } catch (e) { set(studioProjectFailureState(e, "Keeping the refused edit failed")); }
  },

  loadProjectFromServer: async (name, revision = null) => {
    try {
      const data = await apiLoadProject(name, revision);
      const projectState = studioProjectStateFromLoadResponse(data, get().trainingConfig);
      set({
        ...studioProjectRestoreState(projectState),
        ...compilerConfigurationInvalidatedState(),
        projectRevision: studioProjectRevisionFromLoadResponse(data, name),
      });
      runRestoredExperiment();
    } catch (e) { set(studioProjectFailureState(e, "Project load failed")); }
  },

  listServerProjects: async () => {
    try {
      const serverProjects = await apiListProjects();
      set(studioProjectListLoadedState(serverProjects));
    } catch { /* non-critical */ }
  },

  deleteServerProject: async (name) => {
    try {
      await apiDeleteProject(name);
      await get().listServerProjects();
      // A delete is recoverable, so the trash is refreshed with the list:
      // the workspace has to be visible somewhere the moment it leaves here.
      await get().listDeletedServerProjects();
    } catch (e) { set(studioProjectFailureState(e, "Project delete failed")); }
  },

  listDeletedServerProjects: async () => {
    try {
      const { deleted } = await apiListDeletedProjects();
      set(studioProjectDeletedListedState(deleted));
    } catch (e) { set(studioProjectFailureState(e, "Deleted project list failed")); }
  },

  restoreDeletedServerProject: async (token) => {
    try {
      await apiRestoreProject(token);
      await get().listServerProjects();
      await get().listDeletedServerProjects();
    } catch (e) { set(studioProjectFailureState(e, "Project restore failed")); }
  },
  };
}
