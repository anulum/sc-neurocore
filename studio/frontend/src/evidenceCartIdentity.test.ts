// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — analysisResultIdentity pure tests
import { describe, expect, it } from "vitest";

import { analysisResultIdentity } from "./evidenceCartIdentity";

const DIGEST_A = "A".repeat(64);
const DIGEST_B = "b".repeat(64);

describe("analysisResultIdentity", () => {
  it("normalizes uppercase hex digests to lowercase", () => {
    expect(
      analysisResultIdentity({
        analysis_metadata: { result_sha256: DIGEST_A },
        currents: [0],
        rates: [1],
      }),
    ).toBe("a".repeat(64));
  });

  it("accepts exact 64 lowercase hex digests", () => {
    expect(
      analysisResultIdentity({
        analysis_metadata: { result_sha256: DIGEST_B },
      }),
    ).toBe(DIGEST_B);
  });

  it("returns null for missing or malformed metadata and digests", () => {
    expect(analysisResultIdentity(null)).toBeNull();
    expect(analysisResultIdentity(undefined)).toBeNull();
    expect(analysisResultIdentity(42)).toBeNull();
    expect(analysisResultIdentity({})).toBeNull();
    expect(analysisResultIdentity({ analysis_metadata: null })).toBeNull();
    expect(analysisResultIdentity({ analysis_metadata: {} })).toBeNull();
    expect(
      analysisResultIdentity({
        analysis_metadata: { result_sha256: "not-a-digest" },
      }),
    ).toBeNull();
    expect(
      analysisResultIdentity({
        analysis_metadata: { result_sha256: "ab".repeat(31) },
      }),
    ).toBeNull();
    expect(
      analysisResultIdentity({
        analysis_metadata: { result_sha256: `${"c".repeat(63)}g` },
      }),
    ).toBeNull();
    expect(
      analysisResultIdentity({
        analysis_metadata: { result_sha256: 123 },
      }),
    ).toBeNull();
  });

  it("does not use object shape or client hashing for identity", () => {
    const left = {
      analysis_metadata: { result_sha256: DIGEST_B },
      currents: [0, 1],
      rates: [0, 5],
    };
    const right = {
      analysis_metadata: { result_sha256: DIGEST_B },
      totally: "different",
      shape: true,
    };
    expect(analysisResultIdentity(left)).toBe(analysisResultIdentity(right));
    const changed = {
      analysis_metadata: { result_sha256: "d".repeat(64) },
      currents: [0, 1],
      rates: [0, 5],
    };
    expect(analysisResultIdentity(left)).not.toBe(analysisResultIdentity(changed));
  });
});
