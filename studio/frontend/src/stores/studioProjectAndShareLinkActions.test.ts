// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio server projects and share links against a mocked server and URL

import { createStore } from "zustand/vanilla";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  deleteProject,
  listDeletedProjects,
  listProjects,
  restoreProject,
  saveProject,
} from "../api/client";
import type { DeletedProjectSummary, ModelSummary, ProjectSaveResponse, ProjectSummary } from "../api/types";
import { readStudioModelLink, readStudioStartupHashState } from "../studioStartupRuntime";
import { studioInitialData } from "./studioInitialState";
import { createStudioStoreActions } from "./studioStoreActions";
import { createStudioEquationIrActions } from "./studioEquationIrActions";
import type { StudioState } from "./studioTypes";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    deleteProject: vi.fn(),
    listDeletedProjects: vi.fn(),
    listProjects: vi.fn(),
    restoreProject: vi.fn(),
    saveProject: vi.fn(),
  };
});
vi.mock("../studioStartupRuntime", async () => {
  const actual = await vi.importActual<typeof import("../studioStartupRuntime")>("../studioStartupRuntime");
  return { ...actual, readStudioModelLink: vi.fn(), readStudioStartupHashState: vi.fn() };
});

afterEach(() => {
  vi.resetAllMocks();
});

/** A store with the real actions over the initial data. */
function studioStore() {
  return createStore<StudioState>((set, get) => ({
    ...studioInitialData, ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
  }));
}

const LISTED = [{ name: "alpha" }] as unknown as ProjectSummary[];
const DELETED = [{ name: "beta", token: "t-beta" }] as unknown as DeletedProjectSummary[];

describe("server projects", () => {
  it("saves under the name, keeps the revision the server gives and lists the projects again", async () => {
    const store = studioStore();
    vi.mocked(saveProject).mockResolvedValue({ name: "alpha", revision: 3 } as ProjectSaveResponse);
    vi.mocked(listProjects).mockResolvedValue(LISTED);
    await store.getState().saveProjectToServer("alpha");
    expect(vi.mocked(saveProject).mock.calls[0]?.[0]).toBe("alpha");
    expect(store.getState().projectRevision).toEqual({ name: "alpha", revision: 3 });
    expect(store.getState().serverProjects).toEqual(LISTED);
  });

  it("reports a save the server refuses", async () => {
    const store = studioStore();
    vi.mocked(saveProject).mockRejectedValue(new Error("disk full"));
    await store.getState().saveProjectToServer("alpha");
    expect(store.getState().error).toContain("disk full");
    expect(listProjects).not.toHaveBeenCalled();
  });

  it("refreshes both the list and the trash after a delete, since a delete can be undone", async () => {
    const store = studioStore();
    vi.mocked(deleteProject).mockResolvedValue(undefined as unknown as Awaited<ReturnType<typeof deleteProject>>);
    vi.mocked(listProjects).mockResolvedValue(LISTED);
    vi.mocked(listDeletedProjects).mockResolvedValue({ deleted: DELETED });
    await store.getState().deleteServerProject("beta");
    expect(vi.mocked(deleteProject).mock.calls[0]?.[0]).toBe("beta");
    expect(store.getState().serverProjects).toEqual(LISTED);
    expect(store.getState().deletedProjects).toEqual(DELETED);
  });

  it("reports a delete the server refuses", async () => {
    const store = studioStore();
    vi.mocked(deleteProject).mockRejectedValue(new Error(""));
    await store.getState().deleteServerProject("beta");
    expect(store.getState().error).toBe("Project delete failed");
  });

  it("reports a trash that cannot be listed", async () => {
    const store = studioStore();
    vi.mocked(listDeletedProjects).mockRejectedValue(new Error(""));
    await store.getState().listDeletedServerProjects();
    expect(store.getState().error).toBe("Deleted project list failed");
  });

  it("restores a deleted project by its token and refreshes the list and the trash", async () => {
    const store = studioStore();
    vi.mocked(restoreProject).mockResolvedValue(undefined as unknown as Awaited<ReturnType<typeof restoreProject>>);
    vi.mocked(listProjects).mockResolvedValue(LISTED);
    vi.mocked(listDeletedProjects).mockResolvedValue({ deleted: [] });
    await store.getState().restoreDeletedServerProject("t-beta");
    expect(vi.mocked(restoreProject).mock.calls[0]?.[0]).toBe("t-beta");
    expect(store.getState().serverProjects).toEqual(LISTED);
    expect(store.getState().deletedProjects).toEqual([]);
  });

  it("reports a restore the server refuses", async () => {
    const store = studioStore();
    vi.mocked(restoreProject).mockRejectedValue(new Error("token expired"));
    await store.getState().restoreDeletedServerProject("t-beta");
    expect(store.getState().error).toBe("token expired");
  });

  it("leaves the list as it was when the projects cannot be listed", async () => {
    const store = studioStore();
    store.setState({ serverProjects: LISTED });
    vi.mocked(listProjects).mockRejectedValue(new Error("offline"));
    await store.getState().listServerProjects();
    expect(store.getState().serverProjects).toEqual(LISTED);
    expect(store.getState().error).toBeNull();
  });
});

describe("share links", () => {
  /** A store that knows one model, with model selection recorded instead of run. */
  function linkStore() {
    const store = studioStore();
    const selectModel = vi.fn(() => Promise.resolve());
    store.setState({ models: [{ name: "LIF" }] as unknown as ModelSummary[], selectModel });
    return { store, selectModel };
  }

  it("says so when a model link names a model the catalogue does not hold", async () => {
    const { store, selectModel } = linkStore();
    vi.mocked(readStudioModelLink).mockReturnValue("Gone");
    await store.getState().applyShareLink();
    expect(store.getState().linkNotice).toContain('"Gone"');
    expect(selectModel).not.toHaveBeenCalled();
  });

  it("opens the model a model link names, leaving the run settings alone", async () => {
    const { store, selectModel } = linkStore();
    store.setState({ current: 4 });
    vi.mocked(readStudioModelLink).mockReturnValue("LIF");
    await store.getState().applyShareLink();
    expect(selectModel).toHaveBeenCalledWith("LIF");
    expect(store.getState().current).toBe(4);
  });

  it("does nothing when the address carries no link", async () => {
    const { store, selectModel } = linkStore();
    vi.mocked(readStudioModelLink).mockReturnValue(null);
    vi.mocked(readStudioStartupHashState).mockReturnValue(null);
    await store.getState().applyShareLink();
    expect(selectModel).not.toHaveBeenCalled();
    expect(store.getState().linkNotice).toBeNull();
  });

  it("says so when a shared run names a model the catalogue does not hold", async () => {
    const { store, selectModel } = linkStore();
    vi.mocked(readStudioModelLink).mockReturnValue(null);
    vi.mocked(readStudioStartupHashState).mockReturnValue({
      selectedModelName: "Gone", current: 2, duration: 50, protocol: "constant",
    });
    await store.getState().applyShareLink();
    expect(store.getState().linkNotice).toContain('"Gone"');
    expect(selectModel).not.toHaveBeenCalled();
  });

  it("applies a shared run's settings and opens its model", async () => {
    const { store, selectModel } = linkStore();
    vi.mocked(readStudioModelLink).mockReturnValue(null);
    vi.mocked(readStudioStartupHashState).mockReturnValue({
      selectedModelName: "LIF", current: 2, duration: 50, protocol: "constant",
    });
    await store.getState().applyShareLink();
    expect(store.getState().current).toBe(2);
    expect(store.getState().duration).toBe(50);
    expect(store.getState().protocol).toBe("constant");
    expect(selectModel).toHaveBeenCalledWith("LIF");
  });
});
