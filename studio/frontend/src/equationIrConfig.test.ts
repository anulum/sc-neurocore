// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Authored compiler settings and request custody

import { describe, expect, it } from "vitest";
import {
  defaultEquationIrConfiguration, equationIrRequest, readEquationIrConfiguration,
  type EquationIrModelDraft, type EquationIrProfileDraft, type EquationIrSourceInput,
} from "./equationIrConfig";

/** Current playground values used by the real native HTTP counterexample. */
function source(): EquationIrSourceInput {
  return {
    equations: ["dv/dt = I"], threshold: "", reset: "", odeParams: {}, odeInit: { v: 0 },
    dt: 0.5, equationIrConfig: defaultEquationIrConfiguration(),
  };
}

/** Apply explicit model drafts without changing the playground's authored values. */
function model(values: Partial<EquationIrModelDraft>): EquationIrSourceInput {
  const input = source();
  Object.assign(input.equationIrConfig.model, values);
  return input;
}

/** Apply explicit numeric drafts without substituting invalid values. */
function profile(values: Partial<EquationIrProfileDraft>): EquationIrSourceInput {
  const input = source();
  Object.assign(input.equationIrConfig.profile, values);
  return input;
}

describe("ordered native equation requests", () => {
  it("keeps distinct equations and nonzero initial values, without a fixed LIF body", () => {
    const first = equationIrRequest(source());
    const changed = source();
    changed.equations = ["dv/dt = 2 * I"];
    changed.odeInit = { v: 7 };
    const second = equationIrRequest(changed);
    expect(first.schema_version).toBe("studio.equation-ir-request.v1");
    expect(first.model.equations).toEqual([{ name: "v", expression: "I" }]);
    expect(second.model.equations).toEqual([{ name: "v", expression: "2 * I" }]);
    expect(second.model.initial_state).toEqual([{ name: "v", value: 7, unit: null }]);
    expect(second).not.toEqual(first);
    expect(first.model.rng_seed).toBe("44257");
    expect(first.profile).toMatchObject({ data_width: 16, fraction: 8, function_lowering: "lut-v1" });
  });

  it("preserves coupled Gauss–Seidel and reset declaration order through JSON", () => {
    const input = model({ method: "gauss_seidel" });
    input.equations = ["dz/dt = I - a", "da/dt = z - a"];
    input.odeInit = { z: 7, a: 2 };
    input.reset = "z = a;\na = 0;";
    const body = equationIrRequest(input);
    const transported: unknown = JSON.parse(JSON.stringify(body));
    expect(transported).toEqual(body);
    expect(body.model.equations.map((item) => item.name)).toEqual(["z", "a"]);
    expect(body.model.initial_state.map((item) => item.name)).toEqual(["z", "a"]);
    expect(body.model.reset).toEqual([{ name: "z", expression: "a" }, { name: "a", expression: "0" }]);
  });

  it("retains every arithmetic choice and exact random key", () => {
    const input = model({
      detection: "poisson", probability_expression: "p", spike_value: "v + 1",
      threshold_rng: "philox", rng_seed: "18446744073709551615", random_stream: "4294967295",
    });
    input.odeParams = { p: 0.5 };
    Object.assign(input.equationIrConfig.profile, {
      module_name: "authored_cell", data_width: "32", fraction: "16", signed: false,
      overflow: "wrap", rounding: "bankers", pipeline_stages: "1", function_lowering: "v2",
      random_precision: "exact",
    });
    const body = equationIrRequest(input);
    expect(body.model).toMatchObject({
      detection: "poisson", probability_expression: "p", spike_value: "v + 1",
      threshold_rng: "philox", rng_seed: "18446744073709551615", random_stream: 4294967295,
    });
    expect(body.profile).toEqual({
      module_name: "authored_cell", data_width: 32, fraction: 16, signed: false,
      overflow: "wrap", rounding: "bankers", pipeline_stages: 1, pipeline_points: [],
      function_lowering: "v2", random_precision: "exact",
    });
    expect(typeof body.model.rng_seed).toBe("string");
  });

  it("freezes only explicitly selected parameters and preserves physical quantities", () => {
    const input = model({
      method: "rk4", units: "strict", constant_names: "tau",
      state_units: '[{"name":"v","unit":"millivolt"}]',
      parameter_units: '[{"name":"tau","unit":"millisecond"},{"name":"gain","unit":"dimensionless"}]',
      dt_unit: "millisecond", input_unit: "millivolt", input_magnitude: "1e0",
    });
    input.equations = ["dv/dt = (I - v) * gain / tau"];
    input.odeParams = { gain: 2, tau: 10 };
    input.odeInit = { v: -65 };
    input.threshold = " v > -50 ";
    input.reset = "v = -65";
    const body = equationIrRequest(input);
    expect(body.model.parameters).toEqual([{ name: "gain", value: 2, unit: "dimensionless" }]);
    expect(body.model.constants).toEqual([{ name: "tau", value: 10, unit: "millisecond" }]);
    expect(body.model.initial_state).toEqual([{ name: "v", value: -65, unit: "millivolt" }]);
    expect(body.model.dt).toEqual({ value: 0.5, unit: "millisecond" });
    expect(body.model.input_unit).toEqual({ value: 1, unit: "millivolt" });
    expect(body.model.threshold).toBe(" v > -50 ");
  });

  it("keeps crossing substeps, escape-rate expressions and explicit product points", () => {
    const crossing = model({ detection: "crossing", substeps: "2" });
    crossing.threshold = "v > 8";
    expect(equationIrRequest(crossing).model).toMatchObject({ detection: "crossing", substeps: 2 });
    const escape = model({ detection: "escape_rate", rate_expression: "rate" });
    escape.odeParams = { rate: 0.1 };
    escape.equationIrConfig.profile.pipeline_points = "mul_0, mul_1";
    expect(equationIrRequest(escape).model.rate_expression).toBe("rate");
    expect(equationIrRequest(escape).profile.pipeline_points).toEqual(["mul_0", "mul_1"]);
  });

  it.each(["", "-1", "01", "1e3", " 1", "18446744073709551616", "9".repeat(21)])(
    "refuses a seed that cannot be transported exactly: %s", (rng_seed) => {
      expect(() => equationIrRequest(model({ rng_seed }))).toThrow("unsigned 64-bit decimal string");
    },
  );

  it.each(["", "true", "-1", "1.5", "9007199254740992"])(
    "refuses an incomplete or inexact numeric draft: %s", (fraction) => {
      expect(() => equationIrRequest(profile({ fraction }))).toThrow("exact integer");
    },
  );

  it.each(["", "0", "-1"])("refuses nonpositive or blank substeps: %s", (substeps) => {
    expect(() => equationIrRequest(model({ substeps }))).toThrow("exact integer");
  });

  it("refuses out-of-domain streams and conflicting product selections", () => {
    expect(() => equationIrRequest(model({ random_stream: "4294967296" }))).toThrow("below 2^32");
    expect(() => equationIrRequest(profile({ pipeline_stages: "1", pipeline_points: "mul_0" })))
      .toThrow("global product pipeline stages or explicit product points");
    expect(() => equationIrRequest(profile({ pipeline_points: "mul_0,mul_0" }))).toThrow("repeat");
    expect(() => equationIrRequest(profile({ pipeline_points: "mul_0," }))).toThrow("empty name");
  });

  it("refuses declaration loss instead of collapsing or filtering it", () => {
    const duplicate = source(); duplicate.equations.push("dv/dt = 2 * I");
    expect(() => equationIrRequest(duplicate)).toThrow("repeat");
    const missing = model({ constant_names: "tau" });
    expect(() => equationIrRequest(missing)).toThrow("undeclared symbol tau");
    const repeated = model({ constant_names: "tau,tau" }); repeated.odeParams = { tau: 10 };
    expect(() => equationIrRequest(repeated)).toThrow("repeat");
    const initial = source(); initial.odeInit = { unknown: 1 };
    expect(() => equationIrRequest(initial)).toThrow("undeclared symbol unknown");
    const reset = source(); reset.reset = "unknown = 0";
    expect(() => equationIrRequest(reset)).toThrow("undeclared symbol unknown");
    reset.reset = "v=0;v=1";
    expect(() => equationIrRequest(reset)).toThrow("repeat");
  });

  it("refuses a partial initial state while retaining an explicit constructor default", () => {
    const input = source();
    input.equations = ["dv/dt = -v + w + I", "dw/dt = -w"];
    input.odeInit = { v: 2 };
    expect(() => equationIrRequest(input)).toThrow("Initial state must initialise every equation state.");
    input.odeInit = {};
    expect(equationIrRequest(input).model.initial_state).toEqual([]);
  });

  it("refuses invalid source syntax and nonfinite playground values", () => {
    const empty = source(); empty.equations = [];
    expect(() => equationIrRequest(empty)).toThrow("Add an equation");
    const invalid = source(); invalid.equations = ["v = I"];
    expect(() => equationIrRequest(invalid)).toThrow("dstate/dt");
    invalid.equations = ["dv/dt = "];
    expect(() => equationIrRequest(invalid)).toThrow("dstate/dt");
    invalid.equations = ["dv/dt = I"]; invalid.reset = "v += 1";
    expect(() => equationIrRequest(invalid)).toThrow("state = expression");
    const nonfinite = source(); nonfinite.odeParams = { tau: Infinity };
    expect(() => equationIrRequest(nonfinite)).toThrow("finite");
    for (const dt of [NaN, Infinity, 0, -0.1]) {
      const input = source(); input.dt = dt;
      expect(() => equationIrRequest(input)).toThrow();
    }
  });

  it("refuses unknown, duplicate, malformed or unused unit declarations", () => {
    for (const state_units of [
      "{}", "[", "[1]", '[{"name":"v","unit":1}]', '[{"name":"v","unit":" "}]',
      '[{"name":"v","unit":"volt","ignored":true}]',
      '[{"name":"v","unit":"volt"},{"name":"v","unit":"millivolt"}]',
      '[{"name":"missing","unit":"volt"}]',
    ]) {
      expect(() => equationIrRequest(model({ units: "strict", state_units }))).toThrow();
    }
    const unused = model({ units: "strict", state_units: '[{"name":"v","unit":"volt"}]' });
    unused.odeInit = {};
    expect(() => equationIrRequest(unused)).toThrow("Initial state units");
    const unknown = model({ units: "strict", parameter_units: '[{"name":"missing","unit":"second"}]' });
    expect(() => equationIrRequest(unknown)).toThrow("undeclared");
    const disabled = model({ state_units: '[{"name":"v","unit":"volt"}]' });
    expect(() => equationIrRequest(disabled)).toThrow("strict units mode");
    for (const input_magnitude of ["", "0x10", "NaN", "1e999"]) {
      expect(() => equationIrRequest(model({ units: "strict", input_unit: "volt", input_magnitude }))).toThrow();
    }
  });
});

describe("persisted compiler settings", () => {
  it("uses independent defaults only for an absent legacy block", () => {
    const first = readEquationIrConfiguration(undefined);
    first.profile.data_width = "32";
    expect(readEquationIrConfiguration(undefined).profile.data_width).toBe("16");
    expect(() => readEquationIrConfiguration(null)).toThrow();
    expect(() => readEquationIrConfiguration({})).toThrow();
  });

  it("retains incomplete drafts and the full exact seed through save and reload", () => {
    const config = defaultEquationIrConfiguration();
    config.profile.data_width = "";
    config.model.rng_seed = "18446744073709551615";
    config.model.state_units = "[";
    const restored = readEquationIrConfiguration(JSON.parse(JSON.stringify(config)));
    expect(restored).toEqual(config);
    expect(restored).not.toBe(config);
    expect(restored.profile).not.toBe(config.profile);
    expect(() => equationIrRequest({ ...source(), equationIrConfig: restored })).toThrow();
  });

  it("refuses explicit shape, type and enum changes without resetting research settings", () => {
    const config = defaultEquationIrConfiguration();
    for (const invalid of [
      { ...config, schema_version: "old" },
      { ...config, ignored: true },
      { ...config, profile: { ...config.profile, signed: 1 } },
      { ...config, profile: { ...config.profile, fraction: 8 } },
      { ...config, profile: { ...config.profile, rounding: "up" } },
      { ...config, model: { ...config.model, rng_seed: 18446744073709551615n } },
      { ...config, model: { ...config.model, method: "adaptive" } },
      { ...config, model: { ...config.model, units: "unchecked" } },
    ]) expect(() => readEquationIrConfiguration(invalid)).toThrow();
  });
});
