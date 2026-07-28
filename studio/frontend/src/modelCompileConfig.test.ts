// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio selected-model compile configuration tests

import { describe, expect, it } from "vitest";

import type { ModelDetail } from "./api/client";
import { modelCompileRequest, modelCosimRequest } from "./modelCompileConfig";

function detail(overrides: Partial<ModelDetail> = {}): ModelDetail {
  return {
    name: "LapicqueNeuron",
    params: [{
      name: "tau", default: 20, unit: "ms", range: null,
      biological_range: null, meaning: "membrane time constant",
    }],
    state_vars: [{ name: "v", default: 0, unit: "", meaning: "state" }],
    compile_configuration: {
      schema_name: "lapicque",
      default_integrator: "exp_euler",
      integrators: ["exp_euler"],
      cosim_integrators: [],
      default_q_format: "Q8.8",
      q_formats: ["Q8.8", "Q16.16"],
    },
    ...overrides,
  } as ModelDetail;
}

describe("modelCompileRequest", () => {
  it("carries the selected configuration and excludes state initials from params", () => {
    expect(modelCompileRequest({
      dt: 1,
      integrator: "exp_euler",
      modelDetail: detail(),
      modelParams: { tau: 15, v: -1 },
      qFormat: "Q16.16",
      selectedModelName: "LapicqueNeuron",
    })).toEqual({
      dt: 1,
      integrator: "exp_euler",
      model_name: "LapicqueNeuron",
      params: { tau: 15 },
      q_format: "Q16.16",
    });
  });

  it("builds co-simulation stimulus over the exact compile configuration", () => {
    const mapDetail = detail({
      compile_configuration: {
        schema_name: "adaptive_threshold_if",
        default_integrator: "map",
        integrators: ["map"],
        cosim_integrators: ["map"],
        default_q_format: "Q8.8",
        q_formats: ["Q8.8", "Q16.16"],
      },
    });

    expect(modelCosimRequest({
      dt: 0.1,
      integrator: "map",
      modelDetail: mapDetail,
      modelParams: { tau: 20, v: 0 },
      qFormat: "Q8.8",
      selectedModelName: "AdaptiveThresholdIFNeuron",
    }, { current: 7.5, nSteps: 64 })).toEqual({
      current: 7.5,
      dt: 0.1,
      integrator: "map",
      model_name: "AdaptiveThresholdIFNeuron",
      n_steps: 64,
      params: { tau: 20 },
      q_format: "Q8.8",
    });

    expect(() => modelCosimRequest({
      dt: 1,
      integrator: "exp_euler",
      modelDetail: detail(),
      modelParams: { tau: 20 },
      qFormat: "Q8.8",
      selectedModelName: "LapicqueNeuron",
    }, { current: 10 })).toThrow("no bit-exact selected-model co-simulation path");
  });

  it("fails closed for an unsupported model or configuration", () => {
    expect(() => modelCompileRequest({
      dt: 1, integrator: "", modelDetail: null, modelParams: {}, qFormat: "Q8.8",
      selectedModelName: "",
    })).toThrow("Choose a catalogue model");
    expect(() => modelCompileRequest({
      dt: 1, integrator: "rk4", modelDetail: detail(), modelParams: {}, qFormat: "Q8.8",
      selectedModelName: "LapicqueNeuron",
    })).toThrow("Integrator rk4 is not declared");
  });
});
