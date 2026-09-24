// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio selected-model compile configuration tests

import { describe, expect, it } from "vitest";

import type { ModelDetail } from "./api/client";
import { modelCompileRequest, modelCosimRequest, qFormatRefusals } from "./modelCompileConfig";

/**
 * Build a model detail carrying a compile configuration.
 *
 * @param overrides - The fields to change.
 * @returns The detail.
 */
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
      numeric_contracts: {},
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
        numeric_contracts: {},
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

/**
 * A detail whose model fits only Q16.16, as the served contract states it.
 *
 * @param offered - The formats the model is representable in.
 * @returns The detail.
 */
function narrowDetail(offered: string[]): ModelDetail {
  const contract = (qFormat: string, refusal: string) => ({
    schema_version: "sc-neurocore.hardware-numeric-contract.v1",
    q_format: qFormat,
    resolution: 1 / 256,
    min_value: -128,
    max_value: 127.99609375,
    representable: refusal === "",
    refusal,
    bit_true_mirror: { available: true, refusal: "", evidence: "" },
    not_stated: [],
  });
  const refusal = "Q8.8 cannot hold this neuron: parameter C=200.0 becomes -56.0";
  return detail({
    compile_configuration: {
      schema_name: "adex",
      default_integrator: "euler",
      integrators: ["euler"],
      cosim_integrators: ["euler"],
      default_q_format: offered[0] ?? null,
      q_formats: offered,
      numeric_contracts: {
        "Q8.8": contract("Q8.8", refusal),
        "Q16.16": contract("Q16.16", offered.length > 0 ? "" : "Q16.16 cannot hold this neuron"),
      },
    },
  });
}

describe("Q-formats the model does not fit", () => {
  it("lists each refused format with the contract's reason", () => {
    const configuration = narrowDetail(["Q16.16"]).compile_configuration;
    expect(configuration && qFormatRefusals(configuration)).toEqual([
      { qFormat: "Q8.8", refusal: "Q8.8 cannot hold this neuron: parameter C=200.0 becomes -56.0" },
    ]);
  });

  it("refuses a format the model does not fit, naming why", () => {
    const input = {
      dt: 1, integrator: "euler", modelParams: {}, selectedModelName: "AdExNeuron",
    };
    expect(() => modelCompileRequest({
      ...input, modelDetail: narrowDetail(["Q16.16"]), qFormat: "Q8.8",
    })).toThrow("Q-format Q8.8 is not offered for the selected model. Q8.8 cannot hold this neuron: parameter C=200.0 becomes -56.0");
    expect(() => modelCompileRequest({
      ...input, modelDetail: narrowDetail(["Q16.16"]), qFormat: "Q4.12",
    })).toThrow(/^Q-format Q4.12 is not offered for the selected model\.$/);
    expect(modelCompileRequest({
      ...input, modelDetail: narrowDetail(["Q16.16"]), qFormat: "",
    }).q_format).toBe("Q16.16");
  });

  it("refuses to compile a model no format can hold instead of guessing one", () => {
    expect(() => modelCompileRequest({
      dt: 1, integrator: "euler", modelDetail: narrowDetail([]), modelParams: {}, qFormat: "",
      selectedModelName: "AdExNeuron",
    })).toThrow(
      "No Q-format Studio compiles at can hold the selected model. Q8.8 cannot hold this neuron: parameter C=200.0 becomes -56.0 Q16.16 cannot hold this neuron",
    );
  });
});
