// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — tests for the refused-analysis sentences

import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { analysisJobErrorMessage, analysisRequestMessage } from "./analysisRequestMessages";

const SRC = new URL(".", import.meta.url).pathname;

/** Every refusal identifier a builder the header runs can produce. */
function refusalCodes(): string[] {
  const codes = new Set<string>();
  for (const file of ["studioAnalysisJobSelection.ts", "analysisJobRequest.ts"]) {
    const source = readFileSync(join(SRC, file), "utf8");
    for (const match of source.matchAll(/"(analysis_(?:selection|request)_[a-z_]+)"/g)) {
      const code = match[1];
      if (code !== undefined) codes.add(code);
    }
  }
  return [...codes].sort();
}

describe("analysis refusal sentences", () => {
  it("has a sentence for every identifier the request builders can refuse with", () => {
    const codes = refusalCodes();
    expect(codes.length).toBeGreaterThan(10);
    const unexplained = codes.filter((code) => analysisRequestMessage(code).includes(code));
    expect(unexplained).toEqual([]);
  });

  it("tells the reader what to choose when a sweep axis is missing", () => {
    expect(analysisRequestMessage("analysis_selection_heatmap_param_x_blank")).toBe(
      "Choose parameters in Sweep X and Sweep Y to run a 2-D sweep.",
    );
  });

  it("names an identifier it does not know instead of hiding it", () => {
    expect(analysisRequestMessage("analysis_request_new_rule")).toBe(
      "The analysis request was refused (analysis_request_new_rule).",
    );
  });
});

describe("analysis job failure sentences", () => {
  it("explains a numerical model failure and what to try", () => {
    expect(analysisJobErrorMessage("ModelSimulationFailure")).toContain("safety bounds");
    expect(analysisJobErrorMessage("ModelSimulationFailure")).toContain("smaller time step");
  });

  it("names a failure class it does not know", () => {
    expect(analysisJobErrorMessage("ZeroDivisionError")).toBe(
      "The analysis failed on the server (ZeroDivisionError).",
    );
  });
});
