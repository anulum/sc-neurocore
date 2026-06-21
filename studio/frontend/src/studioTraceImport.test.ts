// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio trace import request builder tests

import { describe, expect, it } from "vitest";

import {
  parseStudioTraceVoltageValues,
  studioTraceImportRequest,
} from "./studioTraceImport";

describe("Studio trace import request builder", () => {
  it("parses the last numeric column from CSV rows", () => {
    expect(parseStudioTraceVoltageValues([
      "time_ms,voltage_mv",
      "0.0,-65.0",
      "0.1,-64.5",
      "0.2,-64.0",
    ].join("\n"))).toEqual([-65, -64.5, -64]);
  });

  it("accepts whitespace and tab delimited trace rows", () => {
    expect(parseStudioTraceVoltageValues("0.0\t-65\n0.1 -64.5\n0.2   -64")).toEqual([
      -65,
      -64.5,
      -64,
    ]);
  });

  it("ignores header, blank, and non-finite rows", () => {
    expect(parseStudioTraceVoltageValues([
      "",
      "sample,voltage",
      "0,NaN",
      "1,Infinity",
      "2,-61.5",
      "3,not-a-number",
    ].join("\n"))).toEqual([-61.5]);
  });

  it("builds the backend import request when enough samples are present", () => {
    const csv = Array.from({ length: 10 }, (_, index) => `${index},${-65 + index}`).join("\n");

    expect(studioTraceImportRequest(csv, 0.1)).toEqual({
      voltage: [-65, -64, -63, -62, -61, -60, -59, -58, -57, -56],
      dt: 0.1,
    });
  });

  it("rejects traces below the minimum sample count", () => {
    expect(() => studioTraceImportRequest("0,-65\n1,-64", 0.1)).toThrow(
      "Need at least 10 data points",
    );
  });
});
