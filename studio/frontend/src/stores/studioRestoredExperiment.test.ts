// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — How the Studio store shows a restored experiment, through a restored session, against a mocked server

import { createStore } from "zustand/vanilla";
import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchModelDetail } from "../api/client";
import type { ModelDetail } from "../api/types";
import { studioInitialData } from "./studioInitialState";
import { createStudioEquationIrActions } from "./studioEquationIrActions";
import { createStudioStoreActions } from "./studioStoreActions";
import type { StudioState } from "./studioTypes";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return { ...actual, fetchModelDetail: vi.fn() };
});

afterEach(() => {
  vi.resetAllMocks();
});

const DETAIL = { name: "LIF" } as unknown as ModelDetail;

/** Saved sessions, one per source: an ODE, and a catalogue model with inputs of its own. */
const SESSIONS: StudioState["savedSessions"] = [
  { name: "ode run", state: { sourceMode: "ode", duration: 80 } },
  { name: "LIF run", state: { sourceMode: "model", selectedModelName: "LIF", duration: 300 } },
];

/** A store with every real action, its simulation recorded instead of run, and the two sessions saved. */
function restoringStore() {
  const store = createStore<StudioState>((set, get) => ({
    ...studioInitialData, ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
  }));
  const runSimulation = vi.fn(() => Promise.resolve());
  store.setState({ runSimulation, savedSessions: SESSIONS });
  return { store, runSimulation };
}

/** A promise the test settles. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((ok, fail) => { resolve = ok; reject = fail; });
  return { promise, resolve, reject };
}

describe("showing a restored experiment", () => {
  it("runs a restored ODE session at once, without asking for model details", () => {
    const { store, runSimulation } = restoringStore();
    store.getState().loadSession("ode run");
    expect(runSimulation).toHaveBeenCalledOnce();
    expect(fetchModelDetail).not.toHaveBeenCalled();
  });

  it("loads a restored model's details, keeps the restored inputs, and then runs it", async () => {
    const { store, runSimulation } = restoringStore();
    store.setState({ modelDetail: DETAIL });
    const detail = deferred<ModelDetail>();
    vi.mocked(fetchModelDetail).mockReturnValue(detail.promise);
    store.getState().loadSession("LIF run");
    expect(fetchModelDetail).toHaveBeenCalledWith("LIF");
    expect(store.getState().modelDetail).toBeNull();
    expect(runSimulation).not.toHaveBeenCalled();
    detail.resolve(DETAIL);
    await vi.waitFor(() => { expect(runSimulation).toHaveBeenCalledOnce(); });
    expect(store.getState().modelDetail).toBe(DETAIL);
    expect(store.getState().duration).toBe(300);
  });

  it("drops a model's details, or their failure, once a later restore has started", async () => {
    const { store, runSimulation } = restoringStore();
    const late = deferred<ModelDetail>();
    vi.mocked(fetchModelDetail).mockReturnValueOnce(late.promise);
    store.getState().loadSession("LIF run");
    store.getState().loadSession("ode run");
    late.resolve(DETAIL);
    await late.promise;
    await Promise.resolve();
    expect(store.getState().modelDetail).toBeNull();
    expect(runSimulation).toHaveBeenCalledOnce();
    const refused = deferred<ModelDetail>();
    vi.mocked(fetchModelDetail).mockReturnValueOnce(refused.promise);
    store.getState().loadSession("LIF run");
    store.getState().loadSession("ode run");
    refused.reject(new Error("late"));
    await refused.promise.catch(() => undefined);
    await Promise.resolve();
    expect(store.getState().error).toBeNull();
  });

  it("reports details it cannot load, and runs nothing", async () => {
    const { store, runSimulation } = restoringStore();
    vi.mocked(fetchModelDetail).mockRejectedValue(new Error(""));
    store.getState().loadSession("LIF run");
    await vi.waitFor(() => { expect(store.getState().error).toBe("Restored model details could not be loaded"); });
    expect(runSimulation).not.toHaveBeenCalled();
  });
});
