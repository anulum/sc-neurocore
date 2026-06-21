// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio simulation request builder tests

import { describe, expect, it } from "vitest";

import {
  studioBifurcationRequest,
  studioCodegenRequest,
  studioFICurveRequest,
  studioFrequencyResponseRequest,
  studioHeatmapRequest,
  studioPrecisionRequest,
  studioSimulationConfig,
  type StudioSimulationConfigInput,
} from "./studioSimulationConfig";

const input: StudioSimulationConfigInput = {
  sourceMode: "model",
  selectedModelName: "lif",
  modelParams: { tau: 10, capacitance: 1 },
  equations: ["dv/dt = -(v - e_l) / tau + i"],
  threshold: "v > -50",
  reset: "v = -65",
  odeParams: { tau: 20, e_l: -65 },
  odeInit: { v: -65 },
  dt: 0.1,
  duration: 100,
  current: 12,
  protocol: "constant",
};

describe("Studio simulation request builders", () => {
  it("builds the model simulation request when a model is selected", () => {
    expect(studioSimulationConfig(input)).toEqual({
      model_name: "lif",
      params: { tau: 10, capacitance: 1 },
      dt: 0.1,
      duration: 100,
      current: 12,
      protocol: "constant",
    });
  });

  it("builds the ODE simulation request for ODE mode", () => {
    expect(studioSimulationConfig({ ...input, sourceMode: "ode" })).toEqual({
      equations: ["dv/dt = -(v - e_l) / tau + i"],
      threshold: "v > -50",
      reset: "v = -65",
      params: { tau: 20, e_l: -65 },
      init: { v: -65 },
      dt: 0.1,
      duration: 100,
      current: 12,
      protocol: "constant",
    });
  });

  it("falls back to the ODE request shape when model mode has no selected model", () => {
    expect(studioSimulationConfig({ ...input, selectedModelName: "" })).toMatchObject({
      equations: ["dv/dt = -(v - e_l) / tau + i"],
      params: { tau: 20, e_l: -65 },
      init: { v: -65 },
    });
  });

  it("normalizes blank ODE threshold and reset fields to null for simulation", () => {
    expect(studioSimulationConfig({
      ...input,
      sourceMode: "ode",
      threshold: "",
      reset: "",
    })).toMatchObject({
      threshold: null,
      reset: null,
    });
  });

  it("builds an FI curve request with the Studio current sweep defaults", () => {
    expect(studioFICurveRequest({ model_name: "lif" }, -15)).toEqual({
      model_name: "lif",
      i_min: 0,
      i_max: 30,
      i_steps: 25,
    });
    expect(studioFICurveRequest({ model_name: "lif" }, 0)).toMatchObject({ i_max: 50 });
  });

  it("builds a bifurcation request from the selected parameter value", () => {
    expect(studioBifurcationRequest({ model_name: "lif" }, {
      sweepParam: "tau",
      parameterValue: 10,
    })).toEqual({
      model_name: "lif",
      sweep_param: "tau",
      sweep_min: 2,
      sweep_max: 30,
      sweep_steps: 40,
    });
  });

  it("builds a two-parameter heatmap request from selected parameter values", () => {
    expect(studioHeatmapRequest({ model_name: "lif" }, {
      sweepParamX: "tau",
      parameterValueX: 10,
      sweepParamY: "capacitance",
      parameterValueY: 2,
    })).toEqual({
      model_name: "lif",
      param_x: "tau",
      x_min: 2,
      x_max: 30,
      x_steps: 15,
      param_y: "capacitance",
      y_min: 0.4,
      y_max: 6,
      y_steps: 15,
    });
  });

  it("builds the ODE-only precision request without protocol fields", () => {
    expect(studioPrecisionRequest(input)).toEqual({
      equations: ["dv/dt = -(v - e_l) / tau + i"],
      threshold: "v > -50",
      reset: "v = -65",
      params: { tau: 20, e_l: -65 },
      init: { v: -65 },
      dt: 0.1,
      duration: 100,
      current: 12,
    });
  });

  it("builds code generation requests for both supported source modes", () => {
    expect(studioCodegenRequest(input)).toMatchObject({
      mode: "model",
      model_name: "lif",
      equations: null,
      params: { tau: 10, capacitance: 1 },
      init: null,
    });
    expect(studioCodegenRequest({ ...input, sourceMode: "ode" })).toMatchObject({
      mode: "ode",
      model_name: null,
      equations: ["dv/dt = -(v - e_l) / tau + i"],
      params: { tau: 20, e_l: -65 },
      init: { v: -65 },
    });
  });

  it("builds a frequency-response request with the Studio sweep defaults", () => {
    expect(studioFrequencyResponseRequest({ equations: ["dv/dt = -v"] }, 0)).toEqual({
      equations: ["dv/dt = -v"],
      amplitude: 10,
      freq_min: 1,
      freq_max: 200,
      n_freqs: 20,
    });
  });
});
