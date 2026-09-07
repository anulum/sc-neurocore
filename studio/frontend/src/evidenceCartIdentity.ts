// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Pure analysis evidence identity (metadata digests only)

/**
 * Identifying an analysis result by the digest the server gave it.
 *
 * The digest is read, never computed. Hashing the payload in the browser would
 * produce a number that agrees with nothing: the server digests its own
 * canonical form, and a second digest taken over a JavaScript object would
 * differ for reasons that have nothing to do with the result. So this reads
 * `analysis_metadata.result_sha256` and refuses anything that is not exactly a
 * 64-character hexadecimal digest.
 *
 * The narrowing is structural throughout -- no assertions -- because the value
 * arrives as `unknown` from a payload nothing has validated.
 */

/** A 64-character lowercase hexadecimal digest, and nothing else. */
const HEX_64 = /^[0-9a-f]{64}$/;

/**
 * Whether a value is a plain object.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Read an analysis result's digest.
 *
 * @param result - The result, whatever shape it is.
 * @returns The digest in lowercase, or `null` when the result carries none or
 *   carries something that is not a digest. `null` means unidentifiable, and
 *   the cart treats that as a reason to skip rather than to queue.
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
