// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — The experiment workspace's templates, library failures and presets, through the store

import { createStore } from "zustand/vanilla";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { fetchModelDetail, fetchModels, fetchPreset, fetchPresets, fetchTemplates } from "../api/client";
import type { ModelDetail, NeuronTemplate } from "../api/types";
import { presetSelection } from "../modelSelectionStoreState";
import { studioInitialData } from "./studioInitialState";
import { createStudioEquationIrActions } from "./studioEquationIrActions";
import { createStudioStoreActions } from "./studioStoreActions";
import type { StudioState } from "./studioTypes";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    fetchModelDetail: vi.fn(),
    fetchModels: vi.fn(),
    fetchPreset: vi.fn(),
    fetchPresets: vi.fn(),
    fetchTemplates: vi.fn(),
  };
});

// The real reading of a preset, with a seam for the one value its declared type allows and the reader never gives.
vi.mock("../modelSelectionStoreState", async () => {
  const actual = await vi.importActual<typeof import("../modelSelectionStoreState")>("../modelSelectionStoreState");
  return { ...actual, presetSelection: vi.fn() };
});

beforeEach(async () => {
  const actual = await vi.importActual<typeof import("../modelSelectionStoreState")>("../modelSelectionStoreState");
  vi.mocked(presetSelection).mockImplementation(actual.presetSelection);
});

afterEach(() => {
  vi.resetAllMocks();
});

const DETAIL = { name: "LIF", params: [], state_vars: [], dt: 0.05, integration_method: "euler" } as unknown as ModelDetail;

/** A store with every real action, its three kinds of run recorded instead of started. */
function workspaceStore() {
  const store = createStore<StudioState>((set, get) => ({
    ...studioInitialData, ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
  }));
  const runSimulation = vi.fn(() => Promise.resolve());
  const runFICurve = vi.fn(() => Promise.resolve());
  const runPrecision = vi.fn(() => Promise.resolve());
  store.setState({ runSimulation, runFICurve, runPrecision });
  return { store, runSimulation, runFICurve, runPrecision };
}

/** A promise the test settles. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((ok) => { resolve = ok; });
  return { promise, resolve };
}

describe("the template catalogue", () => {
  it("takes the served templates into the store", async () => {
    const { store } = workspaceStore();
    const templates = [{ name: "lif" }, { name: "izhikevich" }] as unknown as NeuronTemplate[];
    vi.mocked(fetchTemplates).mockResolvedValue(templates);
    await store.getState().loadTemplates();
    expect(store.getState().templates).toEqual(templates);
  });
});

describe("a library failure that is not an Error", () => {
  it("names the model library's failure in words of its own", async () => {
    const { store } = workspaceStore();
    vi.mocked(fetchModels).mockRejectedValue("connection reset");
    await store.getState().loadModels();
    expect(store.getState().modelsError).toBe("Model library initialization failed");
  });

  it("names the preset library's failure in words of its own", async () => {
    const { store } = workspaceStore();
    vi.mocked(fetchPresets).mockRejectedValue("connection reset");
    await store.getState().loadPresets();
    expect(store.getState().presetsError).toBe("Experiment library load failed");
  });
});

describe("loading a preset", () => {
  it("selects a model preset's model, applies its run settings once the details arrive, then runs its f-I curve", async () => {
    const { store, runFICurve, runPrecision } = workspaceStore();
    vi.mocked(fetchPreset).mockResolvedValue({
      mode: "model", model_name: "LIF", current: 42, duration: 350, protocol: "ramp", suggested_view: "fi-curve",
    });
    const detail = deferred<ModelDetail>();
    vi.mocked(fetchModelDetail).mockReturnValue(detail.promise);
    const loading = store.getState().loadPreset("lif-fi");
    await vi.waitFor(() => { expect(fetchModelDetail).toHaveBeenCalledWith("LIF"); });
    expect(fetchPreset).toHaveBeenCalledWith("lif-fi");
    expect(store.getState().selectedModelName).toBe("LIF");
    expect(store.getState()).toMatchObject({ current: 10, duration: 100, protocol: "constant" });
    detail.resolve(DETAIL);
    await loading;
    expect(store.getState().modelDetail).toBe(DETAIL);
    expect(store.getState()).toMatchObject({ current: 42, duration: 350, protocol: "ramp", activeTab: "trace" });
    expect(runFICurve).toHaveBeenCalledOnce();
    expect(runPrecision).not.toHaveBeenCalled();
  });

  it("takes an equation preset into the ODE editor, outdates the compiled design, then runs its precision study", async () => {
    const { store, runFICurve, runPrecision, runSimulation } = workspaceStore();
    store.setState({ svSource: "module stale; endmodule" });
    vi.mocked(fetchPreset).mockResolvedValue({
      equations: ["dv/dt = -v + I"], params: { tau: 12 }, current: 7, suggested_view: "precision",
    });
    await store.getState().loadPreset("ode-precision");
    expect(store.getState()).toMatchObject({
      sourceMode: "ode", equations: ["dv/dt = -v + I"], odeParams: { tau: 12 }, current: 7, svSource: "",
    });
    expect(fetchModelDetail).not.toHaveBeenCalled();
    expect(runPrecision).toHaveBeenCalledOnce();
    expect(runFICurve).not.toHaveBeenCalled();
    expect(runSimulation).not.toHaveBeenCalled();
  });

  it("opens the tab a preset suggests and runs the simulation, changing nothing else when it names no experiment", async () => {
    const { store, runFICurve, runPrecision, runSimulation } = workspaceStore();
    const before = store.getState();
    vi.mocked(fetchPreset).mockResolvedValue({ suggested_view: "phase" });
    await store.getState().loadPreset("view-only");
    expect(store.getState().activeTab).toBe("phase");
    expect(store.getState()).toMatchObject({
      sourceMode: before.sourceMode, equations: before.equations, current: before.current,
      selectedModelName: before.selectedModelName,
    });
    expect(fetchModelDetail).not.toHaveBeenCalled();
    expect(runSimulation).toHaveBeenCalledOnce();
    expect(runFICurve).not.toHaveBeenCalled();
    expect(runPrecision).not.toHaveBeenCalled();
  });

  it("keeps the run settings when a model selection carries none, as its declared type allows", async () => {
    const { store } = workspaceStore();
    vi.mocked(fetchPreset).mockResolvedValue({ mode: "model", model_name: "LIF" });
    vi.mocked(presetSelection).mockReturnValueOnce({
      action: { kind: "simulate", activeTab: "trace" }, modelName: "LIF", modelRuntimeState: null, odeState: null,
    });
    vi.mocked(fetchModelDetail).mockResolvedValue(DETAIL);
    await store.getState().loadPreset("lif-bare");
    expect(store.getState().modelDetail).toBe(DETAIL);
    expect(store.getState()).toMatchObject({ current: 10, duration: 100, protocol: "constant" });
  });
});
