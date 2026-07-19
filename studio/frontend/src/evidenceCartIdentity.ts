// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Pure analysis evidence identity (metadata digests only)

/**
 * Content-faithful identity for analysis results used by the evidence cart.
 * Never JSON-serialises payloads and never hashes client-side objects.
 * Assertion-free structural narrowing at the unknown boundary.
 */

const HEX_64 = /^[0-9a-f]{64}$/;

/**
 * True for plain objects and object-like values that are not arrays or null.
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Return normalized lowercase ``analysis_metadata.result_sha256`` when it is
 * exactly 64 hexadecimal characters; otherwise ``null``.
 */
export function analysisResultIdentity(result: unknown): string | null {
  if (!isRecord(result)) {
    return null;
  }
  const metadata = result.analysis_metadata;
  if (!isRecord(metadata)) {
    return null;
  }
  const digest = metadata.result_sha256;
  if (typeof digest !== "string") {
    return null;
  }
  const normalized = digest.trim().toLowerCase();
  if (!HEX_64.test(normalized)) {
    return null;
  }
  return normalized;
}
