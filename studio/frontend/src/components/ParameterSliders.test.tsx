// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { describe, expect, it } from "vitest";

import { sliderBounds } from "./ParameterSliders";

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
