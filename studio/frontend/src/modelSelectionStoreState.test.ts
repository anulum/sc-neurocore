// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio model selection store state helper tests
import { describe, expect, it } from "vitest";

import type { ModelDetail, ModelSummary, NeuronTemplate, PresetSummary } from "./api/client";
import {
  modelDefaultParameters,
  modelDetailLoadedState,
  modelSelectionStartedState,
  modelsLoadedState,
  presetSelection,
  presetsLoadedState,
  templateSelectedState,
  templatesLoadedState,
} from "./modelSelectionStoreState";

function template(overrides: Partial<NeuronTemplate> = {}): NeuronTemplate {
  return {
    current: overrides.current ?? 10,
    description: overrides.description ?? "Leaky integrate-and-fire",
    dt: overrides.dt ?? 0.1,
    duration: overrides.duration ?? 100,
    equations: overrides.equations ?? ["dv/dt = -(v - E_L) / tau_m + I / C"],
    init: overrides.init ?? { v: -65 },
    name: overrides.name ?? "lif",
    params: overrides.params ?? { C: 1, E_L: -65, tau_m: 10 },
    reset: overrides.reset ?? "v = -65",
    threshold: overrides.threshold ?? "v > -50",
  };
}

function modelSummary(overrides: Partial<ModelSummary> = {}): ModelSummary {
  return {
    name: "lif",
    module: "lif",
    category: "point",
    tier: 2,
    evidence_kind: "curated",
    science_tier: 2,
    science_label: "S2",
    silicon_tier: null,
    silicon_label: "none",
    validation_metric: "none",
    integration_method: "euler",
    terminal_silicon_tier: "",
    terminal_reason: "No terminal silicon target declared.",
    category_slug: "point",
    category_source: "declared",
    family: "point",
    maturity: "experimental",
    biophysical_detail: "point",
    n_params: 2,
    n_state_vars: 1,
    state_var_names: ["v"],
    dt: 0.1,
    description: "LIF",
    intended_use: [],
    hardware_fit: [],
    behavior_tags: [],
    provenance: null,
    ...overrides,
  };
}

function modelDetail(overrides: Partial<ModelDetail> = {}): ModelDetail {
  return {
    ...modelSummary(overrides),
    docstring: "Leaky integrate-and-fire neuron.",
    display_name: "",
    dt: 0.05,
    params: [
      { default: 10, name: "tau_m", unit: "", range: null, biological_range: null, meaning: "" },
      { default: -65, name: "E_L", unit: "", range: null, biological_range: null, meaning: "" },
    ],
    state_vars: [{ default: -65, name: "v", unit: "", meaning: "" }],
    dynamics: {},
    backends: [],
    reproducibility: { reference_config: "", golden_trace_sha256: "", reproducible: false },
    documentation_slug: "",
    compile_configuration: {
      schema_name: "lif",
      default_integrator: "euler",
      integrators: ["euler", "rk4"],
      default_q_format: "Q8.8",
      q_formats: ["Q8.8", "Q16.16"],
    },
    ...overrides,
  };
}

function presetSummary(overrides: Partial<PresetSummary> = {}): PresetSummary {
  return {
    description: overrides.description ?? "Baseline LIF preset",
    id: overrides.id ?? "lif-baseline",
    suggested_view: overrides.suggested_view ?? "trace",
    title: overrides.title ?? "LIF baseline",
  };
}

describe("model selection store state helpers", () => {
  it("builds loaded list patches", () => {
    const templates = [template()];
    const models = [modelSummary()];
    const presets = [presetSummary()];

    expect(templatesLoadedState(templates)).toEqual({ templates });
    expect(modelsLoadedState(models)).toEqual({ models });
    expect(presetsLoadedState(presets)).toEqual({ presets });
  });

  it("builds ODE template selection patches with copied mutable records", () => {
    const selected = template();
    const state = templateSelectedState(selected);

    expect(state).toEqual({
      current: 10,
      dt: 0.1,
      duration: 100,
      equations: ["dv/dt = -(v - E_L) / tau_m + I / C"],
      error: null,
      fiResult: null,
      odeInit: { v: -65 },
      odeParams: { C: 1, E_L: -65, tau_m: 10 },
      reset: "v = -65",
      result: null,
      sourceMode: "ode",
      threshold: "v > -50",
    });
    expect(state.equations).not.toBe(selected.equations);
    expect(state.odeInit).not.toBe(selected.init);
    expect(state.odeParams).not.toBe(selected.params);
  });

  it("builds model selection and default parameter patches", () => {
    const detail = modelDetail();

    expect(modelSelectionStartedState("lif")).toEqual({
      error: null,
      fiResult: null,
      result: null,
      selectedModelName: "lif",
    });
    expect(modelDefaultParameters(detail)).toEqual({
      E_L: -65,
      tau_m: 10,
      v: -65,
    });
    expect(modelDetailLoadedState(detail)).toEqual({
      dt: 0.05,
      modelDetail: detail,
      modelParams: { E_L: -65, tau_m: 10, v: -65 },
      modelIntegrator: "euler",
      modelQFormat: "Q8.8",
      sourceMode: "model",
    });
  });

  it("parses model presets into model runtime state and post-load action", () => {
    expect(presetSelection({
      current: 7,
      duration: 250,
      mode: "model",
      model_name: "adex",
      protocol: "step",
      suggested_view: "phase",
    })).toEqual({
      action: { activeTab: "phase", kind: "simulate" },
      modelName: "adex",
      modelRuntimeState: { current: 7, duration: 250, protocol: "step" },
      odeState: null,
    });
  });

  it("parses ODE presets and rejects malformed numeric payload fields", () => {
    expect(presetSelection({
      current: 0,
      dt: 0.2,
      duration: Number.NaN,
      equations: ["dv/dt = -v / tau"],
      init: { bad: "x", v: -65 },
      mode: "ode",
      params: { bad: "x", tau: 10 },
      protocol: "",
      reset: "v = -65",
      suggested_view: "fi-curve",
      threshold: "v > -50",
    })).toEqual({
      action: { kind: "fi-curve" },
      modelName: null,
      modelRuntimeState: { current: 0, duration: 200, protocol: "constant" },
      odeState: {
        current: 0,
        dt: 0.2,
        duration: 200,
        equations: ["dv/dt = -v / tau"],
        odeInit: { v: -65 },
        odeParams: { tau: 10 },
        protocol: "constant",
        reset: "v = -65",
        sourceMode: "ode",
        threshold: "v > -50",
      },
    });
  });

  it("falls back to trace simulation for unknown suggested views", () => {
    expect(presetSelection({
      equations: ["dv/dt = -v / tau"],
      suggested_view: "unknown-panel",
    }).action).toEqual({ activeTab: "trace", kind: "simulate" });
    expect(presetSelection({ equations: [1, "bad"] }).odeState).toBeNull();
  });
});
