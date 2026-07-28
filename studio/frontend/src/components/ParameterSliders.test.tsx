// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio parameter and RTL configuration rendering tests

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import { sliderBounds } from "./ParameterSliders";

vi.mock("../stores/studio", () => ({
  useStudioStore: () => ({
    sourceMode: "model",
    modelDetail: {
      params: [],
      state_vars: [],
      compile_configuration: {
        schema_name: "adex",
        default_integrator: "euler",
        integrators: ["euler", "rk4"],
        default_q_format: "Q8.8",
        q_formats: ["Q8.8", "Q16.16"],
      },
    },
    modelParams: {}, modelIntegrator: "rk4", modelQFormat: "Q16.16",
    setModelParam: () => undefined, setModelIntegrator: () => undefined,
    setModelQFormat: () => undefined, odeParams: {}, odeInit: {},
    setOdeParam: () => undefined, setOdeInit: () => undefined,
    current: 10, dt: 0.1, duration: 100, protocol: "constant",
    setCurrent: () => undefined, setDt: () => undefined,
    setDuration: () => undefined, setProtocol: () => undefined,
  }),
}));

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
