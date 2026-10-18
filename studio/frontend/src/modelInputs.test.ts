// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Stored model seed and parameter projection contracts

import { describe, expect, it } from "vitest";
import { modelParameterValue, readModelParameters } from "./modelInputs";
import { parseStudioJson, stringifyStudioJson } from "./studioJson";

describe("model inputs", () => {
  it("retains a legacy numeric seed across the actual JSON boundary without dropping inactive model inputs", () => {
    const input = parseStudioJson('{"rate_hz":100,"seed":9223372036854775807,"unused":"text"}');
    const result = readModelParameters(input);
    expect(result).toEqual({ rate_hz: 100, seed: 9223372036854775807n });
    expect(stringifyStudioJson(result)).toBe('{"rate_hz":100,"seed":9223372036854775807}');
    expect(modelParameterValue(result, "rate_hz", 10)).toBe(100);
    expect(modelParameterValue(result, "v", -65)).toBe(-65);
    expect(() => modelParameterValue(result, "seed", 0)).toThrow("seed control");
  });

  it.each([undefined, null, [], "text", 1])("reads an absent parameter map as empty (%s)", (value) => {
    expect(readModelParameters(value)).toEqual({});
  });

  it("keeps finite values and explicitly refuses malformed seed overrides", () => {
    expect(readModelParameters({ seed: null, v: -65, bad: Number.NaN })).toEqual({ v: -65 });
    expect(() => readModelParameters({ seed: "42" })).toThrow("Saved simulation seed");
    expect(() => readModelParameters({ seed: 2n ** 63n })).toThrow("Saved simulation seed");
  });
});
