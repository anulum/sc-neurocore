// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Stage failure record tests
import { createStore } from "zustand/vanilla";
import { describe, expect, it } from "vitest";

import { studioGuidedFlowFailures } from "../studioGuidedFlowInputs";
import { studioExperimentKey } from "../studioExperimentKey";
import { studioSimulationConfigInput } from "../studioSimulationConfigInput";
import { studioInitialData } from "./studioInitialState";
import {
  studioStageExperimentKey,
  studioStageOutcomeState,
  studioStageSet,
} from "./studioStageFailure";
import { createStudioStoreActions } from "./studioStoreActions";
import type { StudioState } from "./studioTypes";

/**
 * A real Studio store with the production actions.
 *
 * @returns The store.
 */
function studioStore() {
  return createStore<StudioState>((set, get) => ({
    ...studioInitialData,
    ...createStudioStoreActions(set, get),
  }));
}

/**
 * The experiment key the store currently describes.
 *
 * @param state - The store's state.
 * @returns Its key.
 */
function keyOf(state: StudioState): string {
  return studioExperimentKey(studioSimulationConfigInput(state));
}

describe("stage failure records", () => {
  it("records a real compile failure against the experiment it failed under", async () => {
    const store = studioStore();
    store.setState({ sourceMode: "ode" });

    // No Studio server answers in the test runner, so the request really fails.
    await store.getState().runCompile();

    const state = store.getState();
    expect(state.isSimulating).toBe(false);
    expect(state.error).not.toBeNull();
    expect(state.stageFailure).toEqual({
      stage: "compile",
      message: state.error,
      experimentKey: keyOf(state),
    });
    expect(studioGuidedFlowFailures(state, keyOf(state))).toEqual({ compile: state.error });
  });

  it("stops showing a failure once the reader changes the experiment", async () => {
    const store = studioStore();
    store.setState({ sourceMode: "ode" });
    await store.getState().runCompile();

    store.setState({ duration: store.getState().duration + 50 });

    const state = store.getState();
    expect(state.stageFailure?.stage).toBe("compile");
    expect(studioGuidedFlowFailures(state, keyOf(state))).toEqual({});
  });

  it("withdraws a stage's failure when that stage runs again, but not another stage's", () => {
    const store = studioStore();
    const failure = { stage: "compile" as const, message: "RTL emission failed", experimentKey: "k" };
    store.setState({ stageFailure: failure });

    studioStageSet("simulate", store.getState, store.setState)({ isSimulating: true, error: null });
    expect(store.getState().stageFailure).toEqual(failure);

    studioStageSet("compile", store.getState, store.setState)({ isSimulating: true, error: null });
    expect(store.getState().stageFailure).toBeNull();
    expect(store.getState().isSimulating).toBe(true);
  });

  it("passes superseded endings and messages that end nothing through unchanged", () => {
    const store = studioStore();
    const sink = studioStageSet("synthesise", store.getState, store.setState);

    sink({ isSimulating: false });
    sink({ error: "shown without ending a run" });

    expect(store.getState().stageFailure).toBeNull();
    expect(store.getState().error).toBe("shown without ending a run");
  });

  it("records a failure whose inputs have no identity under no experiment", () => {
    const store = studioStore();
    store.setState({ current: Number.NaN });

    expect(studioStageExperimentKey(store.getState())).toBeNull();
    studioStageSet("simulate", store.getState, store.setState)({
      isSimulating: false,
      error: "invalid current",
    });
    expect(store.getState().stageFailure).toEqual({
      stage: "simulate",
      message: "invalid current",
      experimentKey: null,
    });
  });

  it("keeps another stage's failure when a different stage succeeds", () => {
    const failure = { stage: "compile" as const, message: "RTL emission failed", experimentKey: "k" };
    const state = { ...studioInitialData, stageFailure: failure };

    expect(studioStageOutcomeState("simulate", null, state)).toEqual({ stageFailure: failure });
    expect(studioStageOutcomeState("compile", null, state)).toEqual({ stageFailure: null });
  });

  it("records a guided attempt's outcome through the store action", () => {
    const store = studioStore();

    store.getState().recordStageOutcome("export", "No evidence export target is available.");
    expect(store.getState().stageFailure).toEqual({
      stage: "export",
      message: "No evidence export target is available.",
      experimentKey: keyOf(store.getState()),
    });

    store.getState().recordStageOutcome("export", null);
    expect(store.getState().stageFailure).toBeNull();
  });
});
