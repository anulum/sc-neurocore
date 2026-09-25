// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fit workbench helpers

import { describe, expect, it } from "vitest";

import type { FitResult } from "./api/fitsApi";
import { fitIdentifiabilityNotes, fitParameterRows, parseFixed, parseRecordingCsv } from "./fitWorkbench";

describe("parseRecordingCsv", () => {
  it("reads current and observed per line, with or without a header", () => {
    expect(parseRecordingCsv("a", "current,observed\n0,-65\n5,-64.5\n\n")).toEqual({
      ok: true,
      value: { name: "a", current: [0, 5], observed: [-65, -64.5] },
    });
    expect(parseRecordingCsv("b", "0,-65\r\n5,-64")).toEqual({
      ok: true,
      value: { name: "b", current: [0, 5], observed: [-65, -64] },
    });
  });

  it("names the line that is not two numbers", () => {
    expect(parseRecordingCsv("c", "current,observed\n0,-65\n5,oops\n")).toEqual({
      ok: false,
      message: 'c, line 3: expected "current,observed" numbers',
    });
    expect(parseRecordingCsv("d", "0,-65\n1,2,3")).toEqual({
      ok: false,
      message: 'd, line 2: expected "current,observed" numbers',
    });
    expect(parseRecordingCsv("e", "0,-65\n5,")).toEqual({
      ok: false,
      message: 'e, line 2: expected "current,observed" numbers',
    });
  });

  it("refuses a recording too short to fit", () => {
    expect(parseRecordingCsv("f", "current,observed\n0,-65\n")).toEqual({
      ok: false,
      message: "f: a recording needs at least two samples",
    });
  });
});

describe("parseFixed", () => {
  it("reads name=value lines and skips blank ones", () => {
    expect(parseFixed("R = 1\n\nC=0.5")).toEqual({ ok: true, value: { R: 1, C: 0.5 } });
    expect(parseFixed("")).toEqual({ ok: true, value: {} });
  });

  it("names the line that is not name=number", () => {
    for (const text of ["R", "R=", "=1", "R=x", "R=1=2"]) {
      expect(parseFixed(text)).toEqual({
        ok: false,
        message: "Fixed parameters, line 1: expected name=number",
      });
    }
  });
});

/**
 * A fit result with the fields the helpers read.
 *
 * @param overrides - Fields to change.
 * @returns The result.
 */
function result(overrides: Partial<FitResult> = {}): FitResult {
  return {
    fitted: { R: 0.15, C: 0.15 },
    identifiability: { identifiable: true },
    uncertainty: { method: "gauss-newton-asymptotic", standard_errors: { R: 0.01, C: 0.02 }, correlation: null },
    ...overrides,
  } as FitResult;
}

describe("fitParameterRows", () => {
  it("pairs each value with its standard error, or none when none is stated", () => {
    expect(fitParameterRows(result())).toEqual([
      { name: "R", value: 0.15, standardError: 0.01 },
      { name: "C", value: 0.15, standardError: 0.02 },
    ]);
    expect(
      fitParameterRows(
        result({ uncertainty: { method: "m", standard_errors: null, correlation: null } }),
      ).map((row) => row.standardError),
    ).toEqual([null, null]);
    expect(
      fitParameterRows(
        result({ uncertainty: { method: "m", standard_errors: { R: 0.1 }, correlation: null } }),
      ).map((row) => row.standardError),
    ).toEqual([0.1, null]);
  });
});

describe("fitIdentifiabilityNotes", () => {
  it("says nothing when every parameter is identifiable", () => {
    expect(fitIdentifiabilityNotes(result())).toEqual([]);
  });

  it("names each unconstrained combination", () => {
    const notes = fitIdentifiabilityNotes(
      result({
        identifiability: {
          identifiable: false,
          unconstrained_directions: [{ relative_eigenvalue: 0, direction: { R: -0.7071, C: 0.7071 } }],
        },
      }),
    );
    expect(notes).toEqual([
      "The data do not constrain the combination −0.71·R +0.71·C (in search space); no standard errors are stated.",
    ]);
    expect(fitIdentifiabilityNotes(result({ identifiability: { identifiable: false } }))).toEqual([]);
  });

  it("states why when the model diverged near the optimum", () => {
    expect(
      fitIdentifiabilityNotes(
        result({ identifiability: { identifiable: false, reason: "the model diverged near the optimum" } }),
      ),
    ).toEqual(["Not identifiable: the model diverged near the optimum."]);
  });
});
