// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Panel analysis patch ownership tests
import { createStore } from "zustand/vanilla";
import { describe, expect, it } from "vitest";

import type { FICurveResponse } from "./api/client";
import { studioExperimentKey } from "./studioExperimentKey";
import { applyStudioPanelAnalysisPatch } from "./studioPanelAnalysisPatch";
import { studioSimulationConfigInput } from "./studioSimulationConfigInput";
import { studioInitialData } from "./stores/studioInitialState";
import { createStudioStoreActions } from "./stores/studioStoreActions";
import type { StudioState } from "./stores/studioTypes";

/**
 * A real Studio store with the production actions, mid-way through a store-owned run.
 *
 * @returns The store.
 */
function busyStore() {
  const store = createStore<StudioState>((set, get) => ({
    ...studioInitialData,
    ...createStudioStoreActions(set, get),
  }));
  store.setState({ error: "store-owned message", isSimulating: true });
  return store;
}

describe("applyStudioPanelAnalysisPatch", () => {
  it("remembers a panel job failure as a failed analysis of the current experiment", () => {
    const store = busyStore();

    applyStudioPanelAnalysisPatch(
      { error: "analysis_job_timed_out", isSimulating: false, analysisExperimentKey: null },
      store.getState,
      store.setState,
    );

    const state = store.getState();
    expect(state.stageFailure).toEqual({
      stage: "analyse",
      message: "analysis_job_timed_out",
      experimentKey: studioExperimentKey(studioSimulationConfigInput(state)),
    });
    // The store's own request diagnostics are not the panel's to change.
    expect(state.error).toBe("store-owned message");
    expect(state.isSimulating).toBe(true);
  });

  it("withdraws the analysis failure when the panel delivers a result", () => {
    const store = busyStore();
    const key = studioExperimentKey(studioSimulationConfigInput(store.getState()));
    store.setState({ stageFailure: { stage: "analyse", message: "analysis_job_failed", experimentKey: key } });

    applyStudioPanelAnalysisPatch(
      {
        activeTab: "fi-curve",
        analysisExperimentKey: key,
        error: null,
        fiResult: { currents: [], rates: [] } as unknown as FICurveResponse,
        isSimulating: false,
      },
      store.getState,
      store.setState,
    );

    expect(store.getState().stageFailure).toBeNull();
    expect(store.getState().analysisExperimentKey).toBe(key);
  });

  it("leaves another stage's failure alone when the panel only withdraws a result", () => {
    const store = busyStore();
    const failure = { stage: "compile" as const, message: "RTL emission failed", experimentKey: "k" };
    store.setState({ stageFailure: failure });

    applyStudioPanelAnalysisPatch({ analysisExperimentKey: null }, store.getState, store.setState);

    expect(store.getState().stageFailure).toEqual(failure);
    expect(store.getState().analysisExperimentKey).toBeNull();
  });
});
