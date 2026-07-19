// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — studioSimulationConfigInput pure extraction tests
import { describe, expect, it } from "vitest";

import {
  studioSimulationConfig,
  type StudioSimulationConfigInput,
} from "./studioSimulationConfig";
import {
  studioSimulationConfigInput,
  type StudioSimulationConfigSource,
} from "./studioSimulationConfigInput";

const modelSource: StudioSimulationConfigSource = {
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

const odeSource: StudioSimulationConfigSource = {
  ...modelSource,
  sourceMode: "ode",
  selectedModelName: "",
};

describe("studioSimulationConfigInput", () => {
  it("extracts model fields field-for-field without mutation", () => {
    const beforeParams = { ...modelSource.modelParams };
    const beforeEquations = [...modelSource.equations];
    const input = studioSimulationConfigInput(modelSource);
    expect(input).toEqual({
      sourceMode: "model",
      selectedModelName: "lif",
      modelParams: modelSource.modelParams,
      equations: modelSource.equations,
      threshold: "v > -50",
      reset: "v = -65",
      odeParams: modelSource.odeParams,
      odeInit: modelSource.odeInit,
      dt: 0.1,
      duration: 100,
      current: 12,
      protocol: "constant",
    } satisfies StudioSimulationConfigInput);
    expect(input.modelParams).toBe(modelSource.modelParams);
    expect(input.equations).toBe(modelSource.equations);
    expect(input.odeParams).toBe(modelSource.odeParams);
    expect(input.odeInit).toBe(modelSource.odeInit);
    expect(modelSource.modelParams).toEqual(beforeParams);
    expect(modelSource.equations).toEqual(beforeEquations);
  });

  it("extracts ODE fields and preserves sourceMode and empty model name", () => {
    const input = studioSimulationConfigInput(odeSource);
    expect(input.sourceMode).toBe("ode");
    expect(input.selectedModelName).toBe("");
    expect(input.equations).toBe(odeSource.equations);
    expect(input.odeParams).toBe(odeSource.odeParams);
    expect(input.odeInit).toBe(odeSource.odeInit);
    expect(input.threshold).toBe(odeSource.threshold);
    expect(input.reset).toBe(odeSource.reset);
    expect(input.dt).toBe(odeSource.dt);
    expect(input.duration).toBe(odeSource.duration);
    expect(input.current).toBe(odeSource.current);
    expect(input.protocol).toBe(odeSource.protocol);
  });

  it("matches the existing studioSimulationConfig oracle for model and ODE", () => {
    const modelInput = studioSimulationConfigInput(modelSource);
    const odeInput = studioSimulationConfigInput(odeSource);
    expect(studioSimulationConfig(modelInput)).toEqual(
      studioSimulationConfig({
        sourceMode: "model",
        selectedModelName: "lif",
        modelParams: modelSource.modelParams,
        equations: modelSource.equations,
        threshold: modelSource.threshold,
        reset: modelSource.reset,
        odeParams: modelSource.odeParams,
        odeInit: modelSource.odeInit,
        dt: modelSource.dt,
        duration: modelSource.duration,
        current: modelSource.current,
        protocol: modelSource.protocol,
      }),
    );
    expect(studioSimulationConfig(odeInput)).toEqual(
      studioSimulationConfig({
        sourceMode: "ode",
        selectedModelName: "",
        modelParams: odeSource.modelParams,
        equations: odeSource.equations,
        threshold: odeSource.threshold,
        reset: odeSource.reset,
        odeParams: odeSource.odeParams,
        odeInit: odeSource.odeInit,
        dt: odeSource.dt,
        duration: odeSource.duration,
        current: odeSource.current,
        protocol: odeSource.protocol,
      }),
    );
  });

  it("does not invent defaults when optional-looking strings are empty", () => {
    const sparse: StudioSimulationConfigSource = {
      sourceMode: "ode",
      selectedModelName: "",
      modelParams: {},
      equations: [],
      threshold: "",
      reset: "",
      odeParams: {},
      odeInit: {},
      dt: 0,
      duration: 0,
      current: 0,
      protocol: "",
    };
    const input = studioSimulationConfigInput(sparse);
    expect(input.threshold).toBe("");
    expect(input.reset).toBe("");
    expect(input.protocol).toBe("");
    expect(input.modelParams).toBe(sparse.modelParams);
    expect(input.equations).toBe(sparse.equations);
  });
});
