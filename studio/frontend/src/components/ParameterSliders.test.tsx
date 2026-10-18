// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio parameter and RTL configuration rendering tests
// @vitest-environment happy-dom

import { renderToStaticMarkup } from "react-dom/server";
import { createRoot } from "react-dom/client";
import { act } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ModelDetail, ModelParameter, ModelStateVariable } from "../api/client";
import type { SimulationSeedInput } from "../simulationSeed";

import { MacroStepRow, sliderBounds } from "./ParameterSliders";

const editorInputs = vi.hoisted<{ sourceMode: "model" | "ode"; modelParams: Record<string, number | bigint> }>(() => ({ sourceMode: "model", modelParams: {} }));

const seedInput = vi.hoisted(() => ({ value: null as SimulationSeedInput | null }));

const inputHandlers = vi.hoisted(() => ({
  setModelParam: vi.fn<(name: string, value: number) => void>(),
  setSeed: vi.fn<(seed: SimulationSeedInput | null) => void>(),
}));

/** Declared field metadata supplied to the real parameter component. */
interface DeclaredControls {
  params: ModelParameter[];
  state_vars: ModelStateVariable[];
  parameter_kinds: ModelDetail["parameter_kinds"];
  timestep_parameter: string;
}

const declaredFields = vi.hoisted<DeclaredControls>(() => ({
  params: [],
  state_vars: [],
  parameter_kinds: {},
  timestep_parameter: "dt",
}));

vi.mock("../stores/studio", () => ({
  useStudioStore: () => ({
    sourceMode: editorInputs.sourceMode,
    modelDetail: {
      ...declaredFields,
      compile_configuration: {
        schema_name: "adex",
        default_integrator: "euler",
        integrators: ["euler", "rk4"],
        cosim_integrators: ["euler"],
        default_q_format: "Q8.8",
        q_formats: ["Q8.8", "Q16.16"],
        numeric_contracts: {},
      },
    },
    modelParams: editorInputs.modelParams, modelIntegrator: "rk4", modelQFormat: "Q16.16",
    setModelParam: inputHandlers.setModelParam, setSeed: inputHandlers.setSeed, setModelIntegrator: () => undefined,
    setModelQFormat: () => undefined, odeParams: {}, odeInit: {},
    setOdeParam: () => undefined, setOdeInit: () => undefined,
    current: 10, dt: 0.1, duration: 100, protocol: "constant", seed: seedInput.value, trial: "replay",
    setCurrent: () => undefined, setDt: () => undefined,
    setDuration: () => undefined, setProtocol: () => undefined,
  }),
}));

beforeEach(() => {
  vi.clearAllMocks();
  seedInput.value = null;
  editorInputs.sourceMode = "model";
  editorInputs.modelParams = {};
  declaredFields.params = [];
  declaredFields.state_vars = [];
  declaredFields.parameter_kinds = {};
  declaredFields.timestep_parameter = "dt";
});

describe("sliderBounds", () => {
  it("uses the curated range when it is a valid interval", () => {
    const [lo, hi, step] = sliderBounds(20, [1, 100]);
    expect(lo).toBe(1);
    expect(hi).toBe(100);
    expect(step).toBeCloseTo(99 / 200);
  });

  it("falls back to the value heuristic when no range is given", () => {
    const [lo, hi] = sliderBounds(10, null);
    expect(lo).toBeLessThan(10);
    expect(hi).toBeGreaterThan(10);
  });

  it("falls back when the range is degenerate (lo >= hi)", () => {
    const heuristic = sliderBounds(5, null);
    expect(sliderBounds(5, [3, 3])).toEqual(heuristic);
    expect(sliderBounds(5, [9, 1])).toEqual(heuristic);
  });

  it("keeps a positive step for a zero-width admissible interval guard", () => {
    const [, , step] = sliderBounds(0, [0, 0.000001]);
    expect(step).toBeGreaterThan(0);
  });
});

describe("ParameterSliders model compile configuration", () => {
  it("surfaces descriptor/schema-backed integrator and Q-format choices", async () => {
    const { default: ParameterSliders } = await import("./ParameterSliders");
    const html = renderToStaticMarkup(<ParameterSliders />);

    expect(html).toContain("data-testid=\"model-compile-configuration\"");
    expect(html).toContain("integrator");
    expect(html).toContain("rk4");
    expect(html).toContain("Q-format");
    expect(html).toContain("Q16.16");
  });
});

describe("ParameterSliders accessible names", () => {
  it("names every slider and states its value, since its visible label is only a sibling", async () => {
    const { default: ParameterSliders } = await import("./ParameterSliders");
    const html = renderToStaticMarkup(<ParameterSliders />);
    const sliders = html.match(/<input type="range"[^>]*>/g) ?? [];

    expect(sliders.length).toBeGreaterThanOrEqual(3);
    expect(sliders.every((slider) => /aria-label="[^"]+"/.test(slider))).toBe(true);
    expect(sliders.every((slider) => /aria-valuetext="[^"]+"/.test(slider))).toBe(true);
    expect(html).toContain('aria-label="T (ms)"');
    expect(html).toContain('aria-label="dt"');
  });
});

describe("MacroStepRow", () => {
  it("states the fixed sub-step and the macro step instead of offering a dt", () => {
    // Hodgkin-Huxley: one step() is 100 sub-steps of 0.01 ms, and the server
    // refuses any other dt for it.
    const html = renderToStaticMarkup(<MacroStepRow macro={{ sub_step_ms: 0.01, step_ms: 1 }} />);

    expect(html.replace(/\s+/g, " ")).toContain("0.01 ms, fixed: one step is 1 ms of 100 sub-steps");
    expect(html).not.toContain('type="range"');
  });
});

describe("integer constructor slider bounds", () => {
  it("offers integral endpoints and unit steps for declared ranges", () => {
    expect(sliderBounds(3, [0.2, 9.8], "int")).toEqual([1, 9, 1]);
  });
  it("keeps heuristic integer sliders integral, including zero and negative values", () => {
    expect(sliderBounds(0, null, "int")).toEqual([-2, 2, 1]);
    expect(sliderBounds(-3, null, "int")).toEqual([-9, 3, 1]);
    expect(sliderBounds(3, [9, 1], "int")).toEqual([-3, 9, 1]);
  });
});

describe("constructor metadata rendering", () => {
  it("shows admitted initial state while leaving derived state and separate bindings out of sliders", async () => {
    declaredFields.params = [
      { name: "rate_hz", default: 100, unit: "Hz", range: [0, 1000], biological_range: null, meaning: "Event rate" },
      { name: "seed", default: 44257, unit: "", range: null, biological_range: null, meaning: "Replay seed" },
      { name: "dt_ms", default: 1, unit: "ms", range: null, biological_range: null, meaning: "Bin width" },
    ];
    declaredFields.state_vars = [{ name: "rng_state", default: 44257, unit: "integer", meaning: "LFSR state" }];
    declaredFields.parameter_kinds = { rate_hz: "float", seed: "int", dt_ms: "float" };
    declaredFields.timestep_parameter = "dt_ms";
    const { default: ParameterSliders } = await import("./ParameterSliders");
    const html = renderToStaticMarkup(<ParameterSliders />);
    expect(html).toContain('data-testid="slider-rate_hz"');
    expect(html).toContain('data-testid="read-only-state-rng_state"');
    expect(html).not.toContain('data-testid="slider-rng_state₀"');
    expect(html).not.toContain('data-testid="slider-seed"');
    expect(html).not.toContain('data-testid="slider-dt_ms"');

    declaredFields.params = [];
    declaredFields.state_vars = [{ name: "v", default: 128, unit: "", meaning: "Integer state" }];
    declaredFields.parameter_kinds = { v: "int" };
    const writable = renderToStaticMarkup(<ParameterSliders />);
    expect(writable).toContain('data-testid="slider-v₀"');
    expect(writable).not.toContain('data-testid="read-only-state-v"');
  });

  it("states that fields are read-only when constructor admission is unavailable", async () => {
    declaredFields.params = [
      { name: "rate_hz", default: 100, unit: "Hz", range: [0, 1000], biological_range: null, meaning: "Event rate" },
    ];
    declaredFields.state_vars = [{ name: "rng_state", default: 44257, unit: "", meaning: "" }];
    declaredFields.parameter_kinds = null;
    const { default: ParameterSliders } = await import("./ParameterSliders");
    const html = renderToStaticMarkup(<ParameterSliders />);
    expect(html).toContain("Constructor input metadata unavailable");
    expect(html).toContain("100.0 Hz (read-only)");
    expect(html).not.toContain('data-testid="slider-rate_hz"');
    expect(html).toContain('data-testid="read-only-state-rng_state"');
  });
});

/**
 * Enter an input value through the DOM setter and React's native input event.
 *
 * @param input - The rendered input.
 * @param value - The user's value before server validation.
 */
function enterInput(input: HTMLInputElement, value: string): void {
  const descriptor = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
  if (!descriptor?.set) throw new Error("The DOM does not implement input values.");
  descriptor.set.call(input, value);
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

it("binds initial state and seed DOM input without clamping or rounding", async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  declaredFields.state_vars = [{ name: "v", default: 128, unit: "", meaning: "Integer state" }];
  declaredFields.parameter_kinds = { v: "int" };
  const { default: ParameterSliders } = await import("./ParameterSliders");
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  try {
    await act(async () => { root.render(<ParameterSliders />); });
    const state = container.querySelector('[data-testid="slider-v₀"]');
    const seed = container.querySelector('[aria-label="seed"]');
    if (!(state instanceof HTMLInputElement) || !(seed instanceof HTMLInputElement)) {
      throw new Error("The initial-state or seed input is missing.");
    }
    await act(async () => { enterInput(state, "129"); });
    expect(inputHandlers.setModelParam).toHaveBeenLastCalledWith("v", 129);
    for (const [value, expected] of [
      ["1.5", 1.5], ["-1", -1], ["0", 0], ["9007199254740993", 9007199254740993n],
      ["9223372036854775807", 9223372036854775807n], ["1e3", "1e3"], ["", null],
    ] as const) {
      await act(async () => { enterInput(seed, value); });
      expect(inputHandlers.setSeed).toHaveBeenLastCalledWith(expected);
      seedInput.value = expected;
      await act(async () => { root.render(<ParameterSliders />); });
    }
  } finally {
    await act(async () => { root.unmount(); });
    container.remove();
    Reflect.deleteProperty(globalThis, "IS_REACT_ACT_ENVIRONMENT");
  }
});

it("does not display an inactive model seed in the ODE seed editor", async () => {
  editorInputs.sourceMode = "ode";
  editorInputs.modelParams = { seed: 9223372036854775807n };
  const { default: ParameterSliders } = await import("./ParameterSliders");
  const container = document.createElement("div");
  container.innerHTML = renderToStaticMarkup(<ParameterSliders />);
  expect(container.querySelector('[aria-label="seed"]')?.getAttribute("value")).toBe("");
});

it("renders writable state without inventing a missing unit or description", async () => {
  declaredFields.state_vars = [{ name: "v", default: 0, unit: "", meaning: "" }];
  declaredFields.parameter_kinds = { v: "float" };
  const { default: ParameterSliders } = await import("./ParameterSliders");
  const container = document.createElement("div");
  container.innerHTML = renderToStaticMarkup(<ParameterSliders />);
  const control = container.querySelector('[data-testid="slider-v₀"]');
  expect(control).not.toBeNull();
  expect(control?.getAttribute("title")).toBeNull();
});
