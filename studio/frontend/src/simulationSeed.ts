// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Exact simulation seed values and editor input

/** An integral simulation seed, including values beyond JavaScript's safe integer range. */
export type SimulationSeed = number | bigint;

/** A seed draft; invalid numeric notation remains text for the API's explicit refusal. */
export type SimulationSeedInput = SimulationSeed | string;

/**
 * Read a saved seed in the complete API domain without numeric coercion.
 *
 * @param value - A parsed seed, with missing and null values selecting the default.
 * @returns The original number or bigint, or null for a default.
 * @throws {Error} When the seed is not an integer in the unsigned 63-bit domain.
 */
export function readSimulationSeed(value: unknown): SimulationSeed | null {
  if (value === undefined || value === null) return null;
  if (typeof value === "bigint" && value >= 0n && value < 2n ** 63n) return value;
  if (typeof value === "number" && Number.isInteger(value) && value >= 0 && value < 2 ** 63) return value;
  throw new Error("Saved simulation seed must be a non-negative integer below 2^63");
}

/**
 * Keep decimal integer input exact, retaining invalid drafts for authoritative refusal.
 *
 * @param text - The number input's original text, before any Number conversion.
 * @returns Null for an empty field, an exact integer, a fractional number, or invalid text.
 */
export function seedFromInput(text: string): SimulationSeedInput | null {
  if (text === "") return null;
  if (/^-?\d+$/.test(text)) {
    const value = BigInt(text);
    return value >= BigInt(Number.MIN_SAFE_INTEGER) && value <= BigInt(Number.MAX_SAFE_INTEGER)
      ? Number(value) : value;
  }
  const fractional = Number(text);
  if (Number.isFinite(fractional) && !Number.isInteger(fractional)) return fractional;
  return text;
}
