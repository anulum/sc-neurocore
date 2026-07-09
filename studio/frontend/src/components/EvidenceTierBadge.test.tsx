// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { describe, expect, it } from "vitest";

import { tierMeta } from "./EvidenceTierBadge";

describe("tierMeta", () => {
  it("marks Tier 3 as engineering-verified and shows it", () => {
    const m = tierMeta(3, "measured");
    expect(m.short).toBe("T3");
    expect(m.label).toContain("engineering-verified");
    expect(m.label).toContain("measured");
    expect(m.show).toBe(true);
  });

  it("marks Tier 2 as scientifically curated and shows it", () => {
    const m = tierMeta(2, "curated");
    expect(m.short).toBe("T2");
    expect(m.label).toContain("scientifically curated");
    expect(m.show).toBe(true);
  });

  it("hides the badge below the curated bar (Tier 0/1)", () => {
    expect(tierMeta(1, "").show).toBe(false);
    expect(tierMeta(0, "").show).toBe(false);
    expect(tierMeta(1, "").short).toBe("T1");
  });

  it("falls back to a default modality when none is supplied", () => {
    expect(tierMeta(3, "").label).toContain("measured");
    expect(tierMeta(2, "").label).toContain("curated");
  });
});
