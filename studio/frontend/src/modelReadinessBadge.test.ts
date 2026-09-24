// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Readiness badge tests

import { describe, expect, it } from "vitest";

import type { ModelReadiness } from "./api/client";
import { perfectBadge, verifiedTiers, verifiedTiersSource } from "./modelReadinessBadge";

/**
 * A readiness declared perfect whose receipts verify less.
 *
 * @param overrides - Fields to change.
 * @returns The readiness.
 */
function readiness(overrides: Partial<ModelReadiness> = {}): ModelReadiness {
  return {
    science_tier: 5,
    science_label: "S5",
    silicon_tier: 1,
    silicon_label: "H1",
    is_perfect: true,
    is_perfect_verified: false,
    verified: {
      profile: "adex",
      science_tier: 3,
      science_label: "S3",
      silicon_tier: 0,
      silicon_label: "H0",
      source: "receipts",
    },
    ...overrides,
  };
}

describe("perfectBadge", () => {
  it("calls a model perfect only when receipts verify it", () => {
    expect(perfectBadge(readiness({ is_perfect_verified: true }))).toEqual({
      label: "perfect",
      verified: true,
      title: "Verified: fresh facet receipts reach science S5 and the terminal silicon tier",
    });
  });

  it("labels a declared-only claim as unverified", () => {
    const badge = perfectBadge(readiness());
    expect(badge?.label).toBe("declared perfect · unverified");
    expect(badge?.verified).toBe(false);
    expect(badge?.title).toContain("no fresh facet receipts verify it");
  });

  it("says nothing when nothing is claimed", () => {
    expect(perfectBadge(readiness({ is_perfect: false }))).toBeNull();
    expect(perfectBadge(undefined)).toBeNull();
  });
});

describe("verifiedTiers", () => {
  it("names the verified tiers next to the declared ones", () => {
    expect(verifiedTiers(readiness())).toBe("verified S3 / H0");
  });

  it("is absent without a verification block", () => {
    expect(verifiedTiers(readiness({ verified: undefined }))).toBeNull();
    expect(verifiedTiers(undefined)).toBeNull();
  });
});

describe("where the verified tiers come from", () => {
  it("says a checkout re-derived them from receipts", () => {
    expect(verifiedTiers(readiness())).toBe("verified S3 / H0");
    expect(verifiedTiersSource(readiness())).toContain("fresh facet receipts");
  });

  it("says an installation serves them sealed at build", () => {
    const sealed = readiness({
      verified: { profile: "adex", science_tier: 3, science_label: "S3", silicon_tier: 0, silicon_label: "H0", source: "sealed" },
    });
    expect(verifiedTiers(sealed)).toBe("verified S3 / H0 (sealed at build)");
    expect(verifiedTiersSource(sealed)).toContain("cannot re-check receipts");
  });

  it("shows nothing as verified without a sealed record, and says why", () => {
    const unsealed = readiness({
      verified: {
        profile: null, science_tier: 0, science_label: "S0", silicon_tier: null, silicon_label: "none",
        source: "unsealed", unsealed_reason: "AdExNeuron: the readiness seal holds no record of this model",
      },
    });
    expect(verifiedTiers(unsealed)).toBe("not verified in this installation");
    expect(verifiedTiersSource(unsealed)).toBe("AdExNeuron: the readiness seal holds no record of this model");
    const reasonless = readiness({
      verified: { profile: null, science_tier: 0, science_label: "S0", silicon_tier: null, silicon_label: "none", source: "unsealed" },
    });
    expect(verifiedTiersSource(reasonless)).toBe("No sealed verification record");
  });

  it("has nothing to explain before readiness loads", () => {
    expect(verifiedTiersSource(undefined)).toBeNull();
  });
});
