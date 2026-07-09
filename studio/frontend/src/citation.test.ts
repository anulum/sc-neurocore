// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { describe, expect, it } from "vitest";

import type { ModelProvenance } from "./api/client";
import { formatCitation } from "./citation";

const prov = (o: Partial<ModelProvenance> = {}): ModelProvenance => ({
  authors: ["Brette, R.", "Gerstner, W."],
  year: 2005,
  doi: "10.1152/jn.00686.2005",
  paper_title: "Adaptive exponential integrate-and-fire model",
  url: "",
  citeable: true,
  ...o,
});

describe("formatCitation", () => {
  it("builds an author–year–title–doi citation ending in the model context", () => {
    const c = formatCitation(prov(), "AdExNeuron");
    expect(c).toContain("Brette, R., Gerstner, W. (2005).");
    expect(c).toContain("Adaptive exponential integrate-and-fire model.");
    expect(c).toContain("https://doi.org/10.1152/jn.00686.2005");
    expect(c.endsWith("Implemented as AdExNeuron in SC-NeuroCore.")).toBe(true);
  });

  it("falls back to the url when no doi is present", () => {
    const c = formatCitation(prov({ doi: "", url: "https://example.org/paper" }), "X");
    expect(c).toContain("https://example.org/paper");
    expect(c).not.toContain("doi.org");
  });

  it("returns empty for models without citeable provenance", () => {
    expect(formatCitation(null, "X")).toBe("");
    expect(formatCitation(prov({ doi: "", authors: [] }), "X")).toBe("");
  });
});
