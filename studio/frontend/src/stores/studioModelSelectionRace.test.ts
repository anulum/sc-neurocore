// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio model selection response ordering

import { createStore } from "zustand/vanilla";
import { describe, expect, it, vi } from "vitest";

import type { ModelDetail } from "../api/client";
import { fetchModelDetail, fetchReplayPack, loadProject } from "../api/client";
import { studioInitialData } from "./studioInitialState";
import { createStudioStoreActions } from "./studioStoreActions";
import { createStudioEquationIrActions } from "./studioEquationIrActions";
import type { StudioState } from "./studioTypes";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return { ...actual, fetchModelDetail: vi.fn(), fetchReplayPack: vi.fn(), loadProject: vi.fn() };
});

/** Build the contract fields consumed by model selection in this test. */
function detail(name: string, parameter: string): ModelDetail {
  return {
    name,
    dt: 0.05,
    integration_method: "euler",
    compile_configuration: null,
    params: [{ name: parameter, default: 1 }],
    parameter_kinds: { [parameter]: "float" },
    state_vars: [],
  } as unknown as ModelDetail;
}

describe("Studio model selection", () => {
  it.each(["session", "project"] as const)("loads restored %s metadata while keeping the saved constructor, clock and exact seed", async (source) => {
    const store = createStore<StudioState>((set, get) => ({
      ...studioInitialData, ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
    }));
    const runSimulation = vi.fn(() => Promise.resolve());
    vi.mocked(fetchModelDetail).mockResolvedValue(detail("SCStochasticRateAdaptationNeuron", "tau_a"));
    const state = {
      sourceMode: "model", selectedModelName: "SCStochasticRateAdaptationNeuron",
      modelParams: { tau_a: 37 }, dt: 0.5, duration: 20, seed: 9223372036854775807n,
    };
    store.setState({ runSimulation, savedSessions: [{ name: "saved", state }] });
    if (source === "project") {
      vi.mocked(loadProject).mockResolvedValue({ revision: 1, state });
      await store.getState().loadProjectFromServer("saved");
    } else store.getState().loadSession("saved");
    await vi.waitFor(() => { expect(runSimulation).toHaveBeenCalledOnce(); });
    expect(store.getState().modelDetail?.name).toBe("SCStochasticRateAdaptationNeuron");
    expect(store.getState().modelParams).toEqual({ tau_a: 37 });
    expect(store.getState().dt).toBe(0.5);
    expect(store.getState().seed).toBe(9223372036854775807n);
  });

  it.each(["success", "failure"] as const)("ignores a late restored metadata %s after a new selection", async (outcome) => {
    let resolveOld: (value: ModelDetail) => void = () => { throw new Error("No pending read"); };
    let rejectOld: (error: Error) => void = () => { throw new Error("No pending read"); };
    const old = new Promise<ModelDetail>((resolve, reject) => { resolveOld = resolve; rejectOld = reject; });
    vi.mocked(fetchModelDetail).mockImplementation((name) => name === "SCStochasticRateAdaptationNeuron"
      ? old : Promise.resolve(detail(name, "tau_w")));
    const store = createStore<StudioState>((set, get) => ({
      ...studioInitialData, ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
    }));
    const runSimulation = vi.fn(() => Promise.resolve());
    store.setState({ runSimulation, savedSessions: [{ name: "saved", state: {
      sourceMode: "model", selectedModelName: "SCStochasticRateAdaptationNeuron", seed: 42,
    } }] });
    store.getState().loadSession("saved");
    await store.getState().selectModel("AdExNeuron");
    if (outcome === "success") {
      resolveOld(detail("SCStochasticRateAdaptationNeuron", "tau_a"));
      await old;
    } else {
      rejectOld(new Error("Earlier metadata read failed"));
      await expect(old).rejects.toThrow("Earlier metadata read failed");
    }
    expect(store.getState().modelDetail?.name).toBe("AdExNeuron");
    expect(store.getState().modelParams).toEqual({ tau_w: 1 });
    expect(store.getState().seed).toBeNull();
    expect(store.getState().error).toBeNull();
    expect(runSimulation).toHaveBeenCalledOnce();
  });

  it("reports a failed restored metadata read before running an unbound model", async () => {
    vi.mocked(fetchModelDetail).mockRejectedValue(new Error("Model contract unavailable"));
    const store = createStore<StudioState>((set, get) => ({
      ...studioInitialData, ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
    }));
    const runSimulation = vi.fn(() => Promise.resolve());
    store.setState({ runSimulation, savedSessions: [{ name: "saved", state: {
      sourceMode: "model", selectedModelName: "SCStochasticRateAdaptationNeuron", seed: 42,
    } }] });
    store.getState().loadSession("saved");
    await vi.waitFor(() => { expect(store.getState().error).toContain("Model contract unavailable"); });
    expect(store.getState().modelDetail).toBeNull();
    expect(store.getState().seed).toBe(42);
    expect(runSimulation).not.toHaveBeenCalled();
  });

  it("keeps the last selected model when an older detail response arrives later", async () => {
    const replies = new Map<string, (value: ModelDetail) => void>();
    vi.mocked(fetchModelDetail).mockImplementation(
      (name) => new Promise((resolve) => { replies.set(name, resolve); }),
    );
    const store = createStore<StudioState>((set, get) => ({
      ...studioInitialData,
      ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
    }));
    store.setState({ runSimulation: vi.fn(() => Promise.resolve()) });

    const first = store.getState().selectModel("ATypeKNeuron");
    await store.getState().exportReplayPack();
    expect(fetchReplayPack).not.toHaveBeenCalled();
    expect(store.getState().error).toBe("Selected model is still loading");
    const second = store.getState().selectModel("HodgkinHuxleyNeuron");
    replies.get("HodgkinHuxleyNeuron")?.(detail("HodgkinHuxleyNeuron", "g_na"));
    await second;
    replies.get("ATypeKNeuron")?.(detail("ATypeKNeuron", "g_a"));
    await first;

    expect(store.getState().selectedModelName).toBe("HodgkinHuxleyNeuron");
    expect(store.getState().modelDetail?.name).toBe("HodgkinHuxleyNeuron");
    expect(store.getState().modelParams).toEqual({ g_na: 1 });
    expect(store.getState().runSimulation).toHaveBeenCalledTimes(1);
  });

  it("does not replace an ODE experiment with a late model response", async () => {
    let resolveDetail: ((value: ModelDetail) => void) | undefined;
    vi.mocked(fetchModelDetail).mockImplementation(
      () => new Promise((resolve) => { resolveDetail = resolve; }),
    );
    const store = createStore<StudioState>((set, get) => ({
      ...studioInitialData,
      ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
    }));
    store.setState({ runSimulation: vi.fn(() => Promise.resolve()) });

    const pending = store.getState().selectModel("ATypeKNeuron");
    store.getState().setSourceMode("ode");
    resolveDetail?.(detail("ATypeKNeuron", "g_a"));
    await pending;

    expect(store.getState().sourceMode).toBe("ode");
    expect(store.getState().modelDetail).toBeNull();
    expect(store.getState().modelParams).toEqual({});
    expect(store.getState().runSimulation).not.toHaveBeenCalled();
  });

  it("keeps a selected ODE template when an earlier model response arrives", async () => {
    let resolveDetail: ((value: ModelDetail) => void) | undefined;
    vi.mocked(fetchModelDetail).mockImplementation(
      () => new Promise((resolve) => { resolveDetail = resolve; }),
    );
    const store = createStore<StudioState>((set, get) => ({
      ...studioInitialData,
      ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
    }));
    store.setState({
      runSimulation: vi.fn(() => Promise.resolve()),
      templates: [{
        name: "ode",
        description: "A controlled ODE template",
        dt: 0.1,
        duration: 100,
        current: 10,
        equations: ["dv/dt = -v"],
        init: { v: -65 },
        params: {},
        threshold: "v > -50",
        reset: "v = -65",
      }],
    });

    const pending = store.getState().selectModel("ATypeKNeuron");
    store.getState().selectTemplate("ode");
    resolveDetail?.(detail("ATypeKNeuron", "g_a"));
    await pending;

    expect(store.getState().sourceMode).toBe("ode");
    expect(store.getState().odeParams).toEqual({});
    expect(store.getState().modelDetail).toBeNull();
    expect(store.getState().runSimulation).toHaveBeenCalledTimes(1);
  });
});
