// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — switching the design source runs the new source

import { createStore } from "zustand/vanilla";
import { describe, expect, it, vi } from "vitest";

import type { ModelDetail } from "../api/client";
import { studioInitialData } from "./studioInitialState";
import { createStudioStoreActions } from "./studioStoreActions";
import type { StudioState } from "./studioTypes";

/**
 * A store whose scheduled simulation is observed rather than sent.
 *
 * The run itself needs a live server and is covered by the browser suites;
 * what this contract fixes is that a switch asks for one.
 *
 * @returns The store and the observer.
 */
function storeWithObservedRuns() {
  const store = createStore<StudioState>((set, get) => ({
    ...studioInitialData,
    ...createStudioStoreActions(set, get),
  }));
  const autoSimulate = vi.fn();
  store.setState({ autoSimulate, modelDetail: { name: "AdExNeuron" } as ModelDetail });
  return { store, autoSimulate };
}

describe("setSourceMode", () => {
  it("runs the equations when switching to them, so the model's run is not shown beside them", () => {
    const { store, autoSimulate } = storeWithObservedRuns();

    store.getState().setSourceMode("ode");

    expect(store.getState().sourceMode).toBe("ode");
    expect(autoSimulate).toHaveBeenCalledTimes(1);
  });

  it("runs the model again when switching back", () => {
    const { store, autoSimulate } = storeWithObservedRuns();
    store.setState({ sourceMode: "ode" });

    store.getState().setSourceMode("model");

    expect(autoSimulate).toHaveBeenCalledTimes(1);
  });

  it("does nothing when the source does not change", () => {
    const { store, autoSimulate } = storeWithObservedRuns();

    store.getState().setSourceMode("model");

    expect(autoSimulate).not.toHaveBeenCalled();
  });
});
