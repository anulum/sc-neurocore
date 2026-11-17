// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio server project revisions, refused edits and loading against a mocked server

import { createStore } from "zustand/vanilla";
import { afterEach, describe, expect, it, vi } from "vitest";

import { branchRefusedEdit, listProjects, loadProject, saveProject, StudioRequestError } from "../api/client";
import type { ProjectBranchResponse, ProjectSaveResponse } from "../api/types";
import { createStudioProjectActions } from "./studioProjectActions";
import { studioInitialData } from "./studioInitialState";
import { createStudioEquationIrActions } from "./studioEquationIrActions";
import { createStudioStoreActions } from "./studioStoreActions";
import type { StudioState } from "./studioTypes";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    branchRefusedEdit: vi.fn(),
    listProjects: vi.fn(),
    loadProject: vi.fn(),
    saveProject: vi.fn(),
  };
});

afterEach(() => {
  vi.resetAllMocks();
});

/** A store with the real actions over the initial data; its project actions show a restore by a recorded call. */
function projectStore() {
  const runRestoredExperiment = vi.fn();
  const store = createStore<StudioState>((set, get) => ({
    ...studioInitialData, ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
    ...createStudioProjectActions(set, get, runRestoredExperiment),
  }));
  vi.mocked(listProjects).mockResolvedValue([]);
  return { store, runRestoredExperiment };
}

describe("saving against a revision", () => {
  it("states the revision the open project was loaded at, and none for another project", async () => {
    const { store } = projectStore();
    store.setState({ projectRevision: { name: "alpha", revision: 3 } });
    vi.mocked(saveProject).mockResolvedValue({ name: "alpha", revision: 4 } as ProjectSaveResponse);
    await store.getState().saveProjectToServer("alpha");
    await store.getState().saveProjectToServer("beta");
    expect(vi.mocked(saveProject).mock.calls.map((call) => call[2])).toEqual([3, null]);
  });

  it("keeps a stale save's edit so that it can be branched, and branches it under a new name", async () => {
    const { store } = projectStore();
    store.setState({ projectRevision: { name: "alpha", revision: 3 } });
    vi.mocked(saveProject).mockRejectedValue(new StudioRequestError("Project alpha changed after revision 3.", 409, null));
    await store.getState().saveProjectToServer("alpha");
    expect(store.getState().error).toBe(
      "Project alpha changed after revision 3. Nothing was overwritten; this edit can be kept as its own branch.",
    );
    expect(store.getState().refusedEdit).toEqual({ name: "alpha", baseRevision: 3 });
    vi.mocked(branchRefusedEdit).mockResolvedValue({ branched: "alpha-branch-1" } as ProjectBranchResponse);
    vi.mocked(listProjects).mockClear();
    await store.getState().keepRefusedEdit();
    const [name, , baseRevision] = vi.mocked(branchRefusedEdit).mock.calls[0] ?? [];
    expect([name, baseRevision]).toEqual(["alpha", 3]);
    expect(store.getState().error).toBe('Kept as "alpha-branch-1". Reload alpha to see the other edit.');
    expect(store.getState().refusedEdit).toBeNull();
    expect(listProjects).toHaveBeenCalledOnce();
  });

  it("branches nothing when no edit was refused", async () => {
    const { store } = projectStore();
    await store.getState().keepRefusedEdit();
    expect(branchRefusedEdit).not.toHaveBeenCalled();
  });

  it("reports a branch the server refuses, and keeps the refused edit", async () => {
    const { store } = projectStore();
    store.setState({ refusedEdit: { name: "alpha", baseRevision: 3 } });
    vi.mocked(branchRefusedEdit).mockRejectedValue(new Error(""));
    await store.getState().keepRefusedEdit();
    expect(store.getState().error).toBe("Keeping the refused edit failed");
    expect(store.getState().refusedEdit).toEqual({ name: "alpha", baseRevision: 3 });
  });
});

describe("loading a project", () => {
  it("restores the project's state at the revision asked for, records the revision and shows the experiment", async () => {
    const { store, runRestoredExperiment } = projectStore();
    vi.mocked(loadProject).mockResolvedValue({ state: { sourceMode: "ode", duration: 250 }, revision: 7 });
    await store.getState().loadProjectFromServer("alpha", 7);
    expect(loadProject).toHaveBeenCalledWith("alpha", 7);
    expect(store.getState().sourceMode).toBe("ode");
    expect(store.getState().duration).toBe(250);
    expect(store.getState().projectRevision).toEqual({ name: "alpha", revision: 7 });
    expect(runRestoredExperiment).toHaveBeenCalledOnce();
    await store.getState().loadProjectFromServer("alpha");
    expect(vi.mocked(loadProject).mock.calls[1]).toEqual(["alpha", null]);
  });

  it("reports a load the server refuses, and shows no restored experiment", async () => {
    const { store, runRestoredExperiment } = projectStore();
    store.setState({ duration: 120 });
    vi.mocked(loadProject).mockRejectedValue(new Error(""));
    await store.getState().loadProjectFromServer("alpha");
    expect(store.getState().error).toBe("Project load failed");
    expect(store.getState().duration).toBe(120);
    expect(runRestoredExperiment).not.toHaveBeenCalled();
  });
});
