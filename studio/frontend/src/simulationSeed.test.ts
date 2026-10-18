// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Simulation seed domain and decimal input contracts

import { describe, expect, it } from "vitest";
import { readSimulationSeed, seedFromInput } from "./simulationSeed";

describe("simulation seed values", () => {
  it.each([0, 42, 2 ** 53, 0n, 9007199254740993n, 9223372036854775807n])("retains the exact valid seed %s", (seed) => {
    expect(readSimulationSeed(seed)).toBe(seed);
  });
  it.each([undefined, null])("retains a default request %s", (seed) => {
    expect(readSimulationSeed(seed)).toBeNull();
  });
  it.each([-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY, 2 ** 63, -1n, 2n ** 63n, "42", false, {}])("refuses invalid stored seeds %s", (seed) => {
    expect(() => readSimulationSeed(seed)).toThrow("Saved simulation seed must be a non-negative integer below 2^63");
  });
  it.each([
    ["", null], ["0", 0], ["-1", -1], ["42", 42],
    ["9007199254740993", 9007199254740993n], ["9223372036854775807", 9223372036854775807n],
    ["-9007199254740993", -9007199254740993n], ["1.5", 1.5], ["1e3", "1e3"], ["bad", "bad"],
  ])("reads editor text %s without integer rounding", (text, expected) => {
    expect(seedFromInput(text)).toBe(expected);
  });
});
