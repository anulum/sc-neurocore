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
  studioExperimentExportRequest,
  studioFICurveRequest,
  studioFrequencyResponseRequest,
  studioHeatmapRequest,
  studioNullclineRequest,
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
  frequencyHz: 10,
  seed: null,
  trial: "replay",
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
      frequency_hz: 10,
      trial: "replay",
    });
  });

  it("sends an explicit seed and trial so the effective config is the same in GUI and API", () => {
    expect(studioSimulationConfig({ ...input, seed: 7, trial: "fresh", protocol: "sine", frequencyHz: 25 })).toEqual({
      model_name: "lif",
      params: { tau: 10, capacitance: 1 },
      dt: 0.1,
      duration: 100,
      current: 12,
      protocol: "sine",
      frequency_hz: 25,
      trial: "fresh",
      seed: 7,
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
      frequency_hz: 10,
      trial: "replay",
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

  it("builds the ODE-only precision request with the explicit protocol and word format", () => {
    expect(studioPrecisionRequest(input)).toEqual({
      equations: ["dv/dt = -(v - e_l) / tau + i"],
      threshold: "v > -50",
      reset: "v = -65",
      params: { tau: 20, e_l: -65 },
      init: { v: -65 },
      dt: 0.1,
      duration: 100,
      current: 12,
      protocol: input.protocol,
      frequency_hz: input.frequencyHz,
      q_format: "Q8.8",
    });
    expect(studioPrecisionRequest(input, "Q16.16")).toMatchObject({ q_format: "Q16.16" });
  });

  it("builds the nullcline request holding every further variable and stating the input", () => {
    const request = studioNullclineRequest({
      equations: ["dv/dt = w + i", "dw/dt = -v", "du/dt = -u"],
      odeParams: { a: 1 },
      odeInit: { v: -65, w: 0, u: 2 },
      protocol: "constant",
      current: 12,
      ranges: { v: [-80, 40], w: [-1, 1] },
      gridSize: 60,
    });
    expect(request).toEqual({
      equations: ["dv/dt = w + i", "dw/dt = -v", "du/dt = -u"],
      params: { a: 1 },
      var_names: ["v", "w"],
      ranges: { v: [-80, 40], w: [-1, 1] },
      grid_size: 60,
      current: 12,
      held: { u: 2 },
    });
    expect(
      studioNullclineRequest({
        equations: ["dv/dt = w", "dw/dt = -v"],
        odeParams: {},
        odeInit: { v: 0, w: 0 },
        protocol: "sine",
        current: 12,
        ranges: {},
        gridSize: 20,
      }),
    ).toMatchObject({ current: 0, held: {} });
  });

  it("exports the same experiment fields the run would use, and only its own branch", () => {
    const modelRequest = studioExperimentExportRequest(input);
    expect(modelRequest).toMatchObject({
      mode: "model",
      model_name: "lif",
      params: { tau: 10, capacitance: 1 },
      dt: input.dt,
      duration: input.duration,
      current: input.current,
      protocol: input.protocol,
      frequency_hz: input.frequencyHz,
      trial: input.trial,
    });
    // The export endpoints are fail-closed: a null of the other branch is a
    // rejected field, not an empty one.
    expect(modelRequest).not.toHaveProperty("equations");
    expect(modelRequest).not.toHaveProperty("init");

    const odeRequest = studioExperimentExportRequest({ ...input, sourceMode: "ode" });
    expect(odeRequest).toMatchObject({
      mode: "ode",
      equations: ["dv/dt = -(v - e_l) / tau + i"],
      params: { tau: 20, e_l: -65 },
      init: { v: -65 },
      protocol: input.protocol,
    });
    expect(odeRequest).not.toHaveProperty("model_name");

    // An export request is the simulation request plus its discriminator.
    const { mode, ...withoutMode } = modelRequest;
    expect(mode).toBe("model");
    expect(withoutMode).toEqual(studioSimulationConfig(input));
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
