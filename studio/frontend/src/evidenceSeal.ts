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

/** Contract version of the canonical form. */
export const EVIDENCE_SEAL_SCHEMA_VERSION = "studio.evidence-seal.v1" as const;

/** Digest algorithm named in every receipt that carries a seal. */
export const EVIDENCE_SEAL_ALGORITHM = "sha256" as const;

/** Raised when a value cannot be sealed identically in both runtimes. */
export class EvidenceSealError extends Error {
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
 * @throws EvidenceSealError when the value would not survive a round trip
 * between the two runtimes unchanged.
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
 */
export async function sealSha256(value: unknown): Promise<string> {
  const data = new TextEncoder().encode(canonicalSealText(value));
  if (typeof globalThis.crypto === "undefined" || !globalThis.crypto.subtle) {
    throw new EvidenceSealError("Web Crypto SHA-256 is required for evidence seals");
  }
  const digest = await globalThis.crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

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
 */
function compareCodePoints(left: string, right: string): number {
  const leftPoints = Array.from(left, (character) => character.codePointAt(0) ?? 0);
  const rightPoints = Array.from(right, (character) => character.codePointAt(0) ?? 0);
  const shared = Math.min(leftPoints.length, rightPoints.length);
  for (let index = 0; index < shared; index += 1) {
    if (leftPoints[index] !== rightPoints[index]) {
      return leftPoints[index] < rightPoints[index] ? -1 : 1;
    }
  }
  return leftPoints.length - rightPoints.length;
}

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

function canonicalFraction(value: number): string {
  const negative = value < 0;
  const text = Math.abs(value).toString(10);
  const matched = /^(\d+)(?:\.(\d+))?(?:[eE]([+-]?\d+))?$/.exec(text);
  if (matched === null) {
    throw new EvidenceSealError(`${text} has no canonical numeric form.`);
  }
  const fraction = matched[2] ?? "";
  // The shortest round-tripping form never ends in a redundant zero, in either
  // runtime, so only the leading zeros of a value below one are dropped.
  const digits = `${matched[1]}${fraction}`.replace(/^0+(?=\d)/, "");
  const exponent = Number(matched[3] ?? "0") - fraction.length + digits.length - 1;
  const mantissa = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
  return `${negative ? "-" : ""}${mantissa}e${exponent}`;
}

function canonicalString(value: string): string {
  const parts = ['"'];
  for (const character of value) {
    const escape = SHORT_ESCAPES[character];
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
