// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio selected-model compile configuration tests

import { describe, expect, it } from "vitest";

import type { ModelDetail } from "./api/client";
import { modelCompileRequest } from "./modelCompileConfig";

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
