// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio cross-runtime evidence seal

/**
 * Browser half of the Studio evidence seal.
 *
 * `sc_neurocore/studio/evidence_seal.py` is the other half. Both encode a
 * value — not a runtime's rendering of it — so a payload the server sealed and
 * the browser handed back seals to the same digest. `JSON.stringify` alone does
 * not: it writes `1` where Python writes `1.0` and `1e-7` where Python writes
 * `1e-07`, which is why a recorded server seal could not be rechecked against
 * anything that had passed through this runtime.
 *
 * The rules, identical on both sides: object keys ordered by code point; no
 * insignificant whitespace; an integral number written as an integer and any
 * other written in scientific form from its shortest round-tripping digits;
 * minimal RFC 8259 string escapes; and a refusal — never a silent change — for
 * a non-finite number or an unpaired surrogate. The Python half additionally
 * refuses an integer no double holds exactly; this runtime has no such value to
 * refuse, because every number here is already a double.
 */

import { at } from "./arrayAt";

/** Contract version of the canonical form. */
export const EVIDENCE_SEAL_SCHEMA_VERSION = "studio.evidence-seal.v1" as const;

/** Digest algorithm named in every receipt that carries a seal. */
export const EVIDENCE_SEAL_ALGORITHM = "sha256" as const;

/** Raised when a value cannot be sealed identically in both runtimes. */
export class EvidenceSealError extends Error {
  /**
   * Build the error, naming it so a caller can tell it from a generic one.
   *
   * @param message - What could not be sealed, and why.
   */
  constructor(message: string) {
    super(message);
    this.name = "EvidenceSealError";
  }
}

const SHORT_ESCAPES: Record<string, string> = {
  "\b": "\\b",
  "\t": "\\t",
  "\n": "\\n",
  "\f": "\\f",
  "\r": "\\r",
  '"': '\\"',
  "\\": "\\\\",
};

/**
 * Return the canonical JSON text used as the seal input.
 *
 * @param value - The value to render.
 * @returns Its canonical text, byte for byte what the server writes for the
 *   same value.
 * @throws {EvidenceSealError} When the value would not survive a round trip
 *   between the two runtimes unchanged.
 */
export function canonicalSealText(value: unknown): string {
  const parts: string[] = [];
  encode(value, parts);
  return parts.join("");
}

/**
 * Return the SHA-256 hex digest of the canonical form of `value`.
 *
 * Requires Web Crypto, which the Studio frontend already depends on for cart
 * digests.
 *
 * @param value - The value to seal.
 * @returns Its SHA-256 digest as lower-case hex.
 * @throws {EvidenceSealError} When the value cannot be sealed, or when Web
 *   Crypto is unavailable.
 */
export async function sealSha256(value: unknown): Promise<string> {
  const data = new TextEncoder().encode(canonicalSealText(value));
  // The DOM types declare `crypto.subtle` as always present. It is not: a page
  // served over plain HTTP is not a secure context and has no SubtleCrypto at
  // all, which is exactly when a silent wrong digest would be worst.
  // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
  if (!globalThis.crypto?.subtle) {
    throw new EvidenceSealError("Web Crypto SHA-256 is required for evidence seals");
  }
  const digest = await globalThis.crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

/**
 * Append one value's canonical text to `parts`.
 *
 * Recursion carries the accumulator rather than concatenating, because a deep
 * payload would otherwise copy its own prefix once per node.
 *
 * @param value - The value to encode.
 * @param parts - Accumulator the canonical text is appended to.
 */
function encode(value: unknown, parts: string[]): void {
  if (value === null) {
    parts.push("null");
    return;
  }
  const kind = typeof value;
  if (kind === "boolean") {
    parts.push(value === true ? "true" : "false");
    return;
  }
  if (kind === "number") {
    parts.push(canonicalNumber(value as number));
    return;
  }
  if (kind === "string") {
    parts.push(canonicalString(value as string));
    return;
  }
  if (Array.isArray(value)) {
    parts.push("[");
    value.forEach((item, index) => {
      if (index) {
        parts.push(",");
      }
      encode(item, parts);
    });
    parts.push("]");
    return;
  }
  if (kind === "object") {
    encodeObject(value as Record<string, unknown>, parts);
    return;
  }
  throw new EvidenceSealError(`${kind} is not a sealable JSON value.`);
}

/**
 * Append an object's canonical text, its keys ordered by code point.
 *
 * Code-point order rather than UTF-16 order: the two disagree above the basic
 * plane, and the server orders by code point.
 *
 * @param value - The object to encode.
 * @param parts - Accumulator the canonical text is appended to.
 */
function encodeObject(value: Record<string, unknown>, parts: string[]): void {
  const keys = Object.keys(value).sort(compareCodePoints);
  parts.push("{");
  keys.forEach((key, index) => {
    if (index) {
      parts.push(",");
    }
    parts.push(canonicalString(key));
    parts.push(":");
    encode(value[key], parts);
  });
  parts.push("}");
}

/**
 * Order strings by code point.
 *
 * The default sort compares UTF-16 code units, which disagrees with Python
 * above the basic multilingual plane; comparing code points makes both
 * runtimes agree.
 *
 * @param left - First key.
 * @param right - Second key.
 * @returns A negative number, zero or a positive number, as `Array.sort` uses.
 */
function compareCodePoints(left: string, right: string): number {
  const leftPoints = Array.from(left, (character) => character.codePointAt(0) ?? 0);
  const rightPoints = Array.from(right, (character) => character.codePointAt(0) ?? 0);
  const shared = Math.min(leftPoints.length, rightPoints.length);
  for (let index = 0; index < shared; index += 1) {
    const leftPoint = at(leftPoints, index);
    const rightPoint = at(rightPoints, index);
    if (leftPoint !== rightPoint) {
      return leftPoint < rightPoint ? -1 : 1;
    }
  }
  return leftPoints.length - rightPoints.length;
}

/**
 * Render a number in the one form both runtimes reach from the same double.
 *
 * An integral double is written as an integer, because that is what
 * `JSON.stringify` writes for it and the server agrees; everything else goes
 * through the shared fractional form.
 *
 * @param value - The number to render.
 * @returns Its canonical text.
 */
function canonicalNumber(value: number): string {
  if (!Number.isFinite(value)) {
    throw new EvidenceSealError(`${String(value)} is not a sealable JSON number.`);
  }
  if (Number.isInteger(value)) {
    // BigInt renders an integral double exactly; toString(10) would switch to
    // exponent notation above 1e21 and disagree with the Python half.
    return BigInt(value).toString(10);
  }
  return canonicalFraction(value);
}

/**
 * Render a non-integral double in the shared scientific form.
 *
 * The runtimes disagree about when to use exponent notation and about how to
 * write the exponent, so neither runtime's own rendering is used: the digits
 * are taken from the shortest round-tripping decimal and re-assembled the same
 * way on both sides.
 *
 * @param value - A finite, non-integral double.
 * @returns Its canonical scientific text.
 * @throws {EvidenceSealError} When the runtime renders it in a form this
 *   contract cannot read.
 */
function canonicalFraction(value: number): string {
  const negative = value < 0;
  const text = Math.abs(value).toString(10);
  const matched = /^(\d+)(?:\.(\d+))?(?:[eE]([+-]?\d+))?$/.exec(text);
  if (matched === null) {
    throw new EvidenceSealError(`${text} has no canonical numeric form.`);
  }
  // A group that did not participate is `undefined`.
  const fraction = matched[2] ?? "";
  // The shortest round-tripping form never ends in a redundant zero, in either
  // runtime, so only the leading zeros of a value below one are dropped.
  const digits = `${matched[1] ?? ""}${fraction}`.replace(/^0+(?=\d)/, "");
  const exponent = Number(matched[3] ?? "0") - fraction.length + digits.length - 1;
  const mantissa = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
  return `${negative ? "-" : ""}${mantissa}e${exponent}`;
}

/**
 * Render a string with the minimal RFC 8259 escapes, refusing what cannot travel.
 *
 * An unpaired surrogate has no valid UTF-8 encoding, so it is refused rather
 * than replaced: a silently altered string would seal to a digest for a value
 * nobody wrote.
 *
 * @param value - The string to render.
 * @returns Its canonical JSON text, quotes included.
 * @throws {EvidenceSealError} When it carries an unpaired surrogate.
 */
function canonicalString(value: string): string {
  const parts = ['"'];
  for (const character of value) {
    const escape = SHORT_ESCAPES[character];
    // This record carries only the few characters that need a short escape.
    if (escape !== undefined) {
      parts.push(escape);
      continue;
    }
    const codePoint = character.codePointAt(0) ?? 0;
    if (codePoint < 0x20) {
      parts.push(`\\u${codePoint.toString(16).padStart(4, "0")}`);
      continue;
    }
    if (codePoint >= 0xd800 && codePoint <= 0xdfff) {
      throw new EvidenceSealError(
        "A sealable string must not contain an unpaired surrogate.",
      );
    }
    parts.push(character);
  }
  parts.push('"');
  return parts.join("");
}
