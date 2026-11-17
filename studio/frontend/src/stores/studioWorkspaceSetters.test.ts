// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — What each Studio workspace setter writes, which built results it voids, and whether it re-simulates

import { createStore } from "zustand/vanilla";
import { describe, expect, it, vi } from "vitest";

import type { NeuronTemplate } from "../api/types";
import { studioInitialData } from "./studioInitialState";
import { createStudioEquationIrActions } from "./studioEquationIrActions";
import { createStudioStoreActions } from "./studioStoreActions";
import type { StudioState } from "./studioTypes";

/** Results built from an earlier configuration, which a change of configuration must void. */
const BUILT = {
  irText: "module lif;", svSource: "module lif_sv;",
  cosimResult: {} as StudioState["cosimResult"], synthResult: {} as StudioState["synthResult"],
};

/** A store with every real action over the initial data, holding built results, its re-simulation recorded. */
function workspaceStore(overrides: Partial<StudioState> = {}) {
  const store = createStore<StudioState>((set, get) => ({
    ...studioInitialData, ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
  }));
  const autoSimulate = vi.fn();
  store.setState({ ...BUILT, ...overrides, autoSimulate });
  return { store, autoSimulate };
}

type Setter = (state: StudioState) => void;

/** All four results built from the earlier configuration are still the ones held. */
function expectBuiltKept(state: StudioState): void {
  expect(state.irText).toBe(BUILT.irText);
  expect(state.svSource).toBe(BUILT.svSource);
  expect(state.cosimResult).toBe(BUILT.cosimResult);
  expect(state.synthResult).toBe(BUILT.synthResult);
}

/** Every field a setter should write holds exactly that value, nested records included. */
function expectWritten(state: StudioState, written: Partial<StudioState>): void {
  for (const [key, value] of Object.entries(written)) {
    expect(state[key as keyof StudioState], key).toEqual(value);
  }
}

describe("a change to the model's definition voids what was built from it and re-simulates", () => {
  it.each<[string, Setter, Partial<StudioState>]>([
    ["the equations", (s) => { s.setEquations(["dv/dt = -v / tau"]); }, { equations: ["dv/dt = -v / tau"] }],
    ["the threshold", (s) => { s.setThreshold("v > 1"); }, { threshold: "v > 1" }],
    ["the reset", (s) => { s.setReset("v = 0"); }, { reset: "v = 0" }],
    ["an ODE parameter", (s) => { s.setOdeParam("tau", 12); }, { odeParams: { ...studioInitialData.odeParams, tau: 12 } }],
    ["an initial value", (s) => { s.setOdeInit("v", 0.5); }, { odeInit: { ...studioInitialData.odeInit, v: 0.5 } }],
    ["a model parameter", (s) => { s.setModelParam("tau_m", 7); }, { modelParams: { ...studioInitialData.modelParams, tau_m: 7 } }],
    ["the time step", (s) => { s.setDt(0.05); }, { dt: 0.05 }],
  ])("%s", (_, change, written) => {
    const { store, autoSimulate } = workspaceStore();
    change(store.getState());
    expectWritten(store.getState(), written);
    expect(store.getState()).toMatchObject({ irText: "", svSource: "", cosimResult: null });
    expect(autoSimulate).toHaveBeenCalledOnce();
  });
});

describe("switching the source", () => {
  it("opens the remembered catalogue model when its details are not loaded, instead of re-simulating", () => {
    const { store, autoSimulate } = workspaceStore({ sourceMode: "ode", modelDetail: null, selectedModelName: "LIF" });
    const selectModel = vi.fn(() => Promise.resolve());
    store.setState({ selectModel });
    store.getState().setSourceMode("model");
    expect(selectModel).toHaveBeenCalledWith("LIF");
    expect(autoSimulate).not.toHaveBeenCalled();
    expect(store.getState()).toMatchObject({ sourceMode: "model", irText: "", svSource: "" });
  });

  it("re-simulates when the source changes and the model is at hand, and not when it stays", () => {
    const { store, autoSimulate } = workspaceStore({ sourceMode: "model" });
    const selectModel = vi.fn(() => Promise.resolve());
    store.setState({ selectModel });
    store.getState().setSourceMode("ode");
    expect(store.getState()).toMatchObject({ sourceMode: "ode", irText: "", svSource: "" });
    expect(autoSimulate).toHaveBeenCalledOnce();
    store.getState().setSourceMode("ode");
    expect(autoSimulate).toHaveBeenCalledOnce();
    expect(selectModel).not.toHaveBeenCalled();
  });
});

describe("resetting to the defaults", () => {
  it("restores a catalogue model's defaults, voids what was built and simulates", () => {
    const detail = { name: "LIF", dt: 0.25, params: [], state_vars: [] } as unknown as StudioState["modelDetail"];
    const { store } = workspaceStore({
      sourceMode: "model", modelDetail: detail, current: 3, duration: 900, seed: 7, dt: 0.01, modelParams: { tau_m: 9 },
    });
    const runSimulation = vi.fn(() => Promise.resolve());
    store.setState({ runSimulation });
    store.getState().resetDefaults();
    expectWritten(store.getState(), { current: 10, duration: 100, seed: null, dt: 0.25, modelParams: {} });
    expect(store.getState()).toMatchObject({ irText: "", svSource: "", cosimResult: null });
    expect(runSimulation).toHaveBeenCalledOnce();
  });

  it.each<[string, Partial<StudioState>]>([
    ["an ODE", { sourceMode: "ode" }],
    ["a catalogue model whose details have not arrived", { sourceMode: "model", modelDetail: null }],
  ])("only simulates again for %s, keeping its inputs and what was built", (_, source) => {
    const { store } = workspaceStore({ ...source, current: 3, duration: 900 });
    const runSimulation = vi.fn(() => Promise.resolve());
    store.setState({ runSimulation });
    store.getState().resetDefaults();
    expectWritten(store.getState(), { current: 3, duration: 900 });
    expectBuiltKept(store.getState());
    expect(runSimulation).toHaveBeenCalledOnce();
  });
});

describe("choosing an equation template", () => {
  const TEMPLATE: NeuronTemplate = {
    name: "Leaky", description: "A leaky integrator.", equations: ["dv/dt = (-v + I) / tau"],
    threshold: "v > 1", reset: "v = 0", params: { tau: 20 }, init: { v: 0 }, dt: 0.05, current: 1.5, duration: 200,
  };

  it("loads the template's model into the ODE editor, voids what was built and simulates it", () => {
    const { store } = workspaceStore({ sourceMode: "model", templates: [TEMPLATE] });
    const runSimulation = vi.fn(() => Promise.resolve());
    store.setState({ runSimulation });
    store.getState().selectTemplate("Leaky");
    expectWritten(store.getState(), {
      sourceMode: "ode", equations: TEMPLATE.equations, threshold: "v > 1", reset: "v = 0",
      odeParams: { tau: 20 }, odeInit: { v: 0 }, dt: 0.05, current: 1.5, duration: 200,
    });
    expect(store.getState()).toMatchObject({ irText: "", svSource: "", cosimResult: null });
    expect(runSimulation).toHaveBeenCalledOnce();
  });

  it("changes nothing and simulates nothing for a name the library does not hold", () => {
    const { store } = workspaceStore({ templates: [TEMPLATE] });
    const runSimulation = vi.fn(() => Promise.resolve());
    store.setState({ runSimulation });
    // The selection returns before any write, so the whole state is still the same object.
    const before = store.getState();
    store.getState().selectTemplate("Missing");
    expect(store.getState()).toBe(before);
    expect(runSimulation).not.toHaveBeenCalled();
  });
});

describe("a change to how the model is compiled voids what was built, without re-simulating", () => {
  it.each<[string, Setter, Partial<StudioState>]>([
    ["the integrator", (s) => { s.setModelIntegrator("rk4"); }, { modelIntegrator: "rk4" }],
    ["the fixed-point format", (s) => { s.setModelQFormat("Q8.8"); }, { modelQFormat: "Q8.8" }],
  ])("%s", (_, change, written) => {
    const { store, autoSimulate } = workspaceStore();
    change(store.getState());
    expectWritten(store.getState(), written);
    expect(store.getState()).toMatchObject({ irText: "", svSource: "", cosimResult: null });
    expect(autoSimulate).not.toHaveBeenCalled();
  });
});

describe("a change to the run's stimulus re-simulates and keeps what was built", () => {
  it.each<[string, Setter, Partial<StudioState>]>([
    ["the duration", (s) => { s.setDuration(400); }, { duration: 400 }],
    ["the protocol", (s) => { s.setProtocol("pulse"); }, { protocol: "pulse" }],
    ["the frequency", (s) => { s.setFrequencyHz(20); }, { frequencyHz: 20 }],
    ["the trial", (s) => { s.setTrial("fresh"); }, { trial: "fresh" }],
  ])("%s", (_, change, written) => {
    const { store, autoSimulate } = workspaceStore();
    change(store.getState());
    expectWritten(store.getState(), written);
    expectBuiltKept(store.getState());
    expect(autoSimulate).toHaveBeenCalledOnce();
  });

  it("voids the co-simulation on a new current, and the synthesis too when a catalogue model is driven", () => {
    const ode = workspaceStore({ sourceMode: "ode" });
    ode.store.getState().setCurrent(4);
    expect(ode.store.getState()).toMatchObject({ current: 4, cosimResult: null, synthResult: BUILT.synthResult });
    expect(ode.autoSimulate).toHaveBeenCalledOnce();
    const model = workspaceStore({ sourceMode: "model" });
    model.store.getState().setCurrent(4);
    expect(model.store.getState()).toMatchObject({ current: 4, cosimResult: null, synthResult: null });
    expect(model.store.getState().irText).toBe(BUILT.irText);
  });

  it("moves a catalogue model's own seed parameter into the run seed, and leaves an ODE's parameters alone", () => {
    const model = workspaceStore({ sourceMode: "model", modelParams: { tau_m: 10, seed: 3 } });
    model.store.getState().setSeed(42);
    expect(model.store.getState().seed).toBe(42);
    expect(model.store.getState().modelParams).toEqual({ tau_m: 10 });
    expect(model.autoSimulate).toHaveBeenCalledOnce();
    const ode = workspaceStore({ sourceMode: "ode", modelParams: { tau_m: 10, seed: 3 } });
    ode.store.getState().setSeed(42);
    expect(ode.store.getState().seed).toBe(42);
    expect(ode.store.getState().modelParams).toEqual({ tau_m: 10, seed: 3 });
  });
});

describe("a change to what is on screen neither voids nor re-simulates", () => {
  it.each<[string, Setter, Partial<StudioState>]>([
    ["the tab", (s) => { s.setActiveTab("train"); }, { activeTab: "train" }],
    ["the model filter", (s) => { s.setModelFilter("izh"); }, { modelFilter: "izh" }],
    ["the swept parameter", (s) => { s.setSweepParam("tau_m"); }, { sweepParam: "tau_m" }],
    ["the second swept parameter", (s) => { s.setSweepParamY("v_th"); }, { sweepParamY: "v_th" }],
    ["a candidate draft", (s) => { s.setCandidateDraft("dv/dt = -v"); }, { candidates: [{ text: "dv/dt = -v" }] }],
    ["an emptied candidate draft", (s) => { s.setCandidateDraft(""); }, { candidates: [] }],
  ])("%s", (_, change, written) => {
    const { store, autoSimulate } = workspaceStore();
    change(store.getState());
    expectWritten(store.getState(), written);
    expectBuiltKept(store.getState());
    expect(autoSimulate).not.toHaveBeenCalled();
  });
});
