// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/** Cases for the indexed reads used in place of a non-null assertion. */

import { describe, expect, it } from "vitest";

import { at, atOr } from "./arrayAt";

describe("at", () => {
  it("returns the element at the index", () => {
    expect(at([10, 20, 30], 1)).toBe(20);
  });

  it("names the index and the length when there is nothing there", () => {
    expect(() => at([10, 20], 5)).toThrow(RangeError);
    expect(() => at([10, 20], 5)).toThrow("index 5 is outside a list of 2");
  });

  it("refuses a negative index rather than reading from the end", () => {
    expect(() => at([10, 20], -1)).toThrow(RangeError);
  });

  it("refuses a hole in a sparse list", () => {
    const sparse: number[] = [];
    sparse[2] = 30;
    expect(() => at(sparse, 0)).toThrow(RangeError);
    expect(at(sparse, 2)).toBe(30);
  });

  it("refuses a stored undefined, which it cannot tell from a hole", () => {
    const holding: (number | undefined)[] = [undefined, 1];
    expect(() => at(holding, 0)).toThrow(RangeError);
    expect(at(holding, 1)).toBe(1);
  });
});

describe("atOr", () => {
  it("returns the element when it is there", () => {
    expect(atOr([10, 20], 0, 99)).toBe(10);
  });

  it("returns the fallback when it is not", () => {
    expect(atOr([10, 20], 7, 99)).toBe(99);
  });

  it("returns the fallback for a stored undefined", () => {
    expect(atOr([undefined, 1], 0, 99)).toBe(99);
  });
});
