// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — JSON transport with exact integer seed tokens

/** Original numeric token supplied by runtimes implementing JSON parse source access. */
interface JsonParseContext {
  source?: string;
}

/** A JSON implementation that can emit a raw, validated JSON number. */
interface JsonRawNumberFactory {
  rawJSON(text: string): unknown;
}

/**
 * Detect native raw-number output without assuming it exists on older browsers.
 *
 * @param value - The runtime JSON implementation.
 * @returns Whether it can preserve a bigint as a numeric JSON token.
 */
function supportsRawNumbers(value: object): value is object & JsonRawNumberFactory {
  return "rawJSON" in value && typeof value.rawJSON === "function";
}

/**
 * Normalize seed tokens after their enclosing numeric domain is known.
 *
 * @param value - Decoded JSON, still holding native numeric values.
 * @param sources - Original unsafe seed tokens indexed by their containing object.
 * @returns The document with exact RNG seeds and ordinary Float64 coefficient maps.
 * @throws {Error} When an unsafe seed has no exact integer source token.
 */
function normalizeSeedTokens(value: unknown, sources: WeakMap<object, string | undefined>): unknown {
  if (value === null || typeof value !== "object") return value;
  const record = value as Record<string, unknown>;
  for (const [key, field] of Object.entries(record)) {
    // These maps describe Float64 coefficients or observed state, including variables named seed.
    if (["odeParams", "odeInit", "initial_state", "final_state", "state_ranges"].includes(key)
      || key === "parameters" && record.source === "ode"
      || key === "params" && Array.isArray(record.equations)) continue;
    if (key === "seed" && typeof field === "number" && !Number.isSafeInteger(field)
      && (Number.isInteger(field) || !Number.isFinite(field))) {
      const source = sources.get(value);
      if (source === undefined) {
        throw new Error("This browser cannot read large Studio seeds without precision loss");
      }
      if (!/^-?\d+$/.test(source)) throw new Error("Studio seed JSON must use an exact integer token");
      record[key] = BigInt(source);
    } else {
      record[key] = normalizeSeedTokens(field, sources);
    }
  }
  return value;
}

/**
 * Parse JSON while retaining the original integer token for unsafe numeric seeds.
 *
 * @param text - The actual JSON response or persisted document.
 * @returns Exact RNG seeds as bigint; declared Float64 coefficient and state maps stay numeric.
 * @throws {Error} When an unsafe seed has no exact integer source token.
 */
export function parseStudioJson(text: string): unknown {
  const sources = new WeakMap<object, string | undefined>();
  const value: unknown = JSON.parse(text, function (
    this: object, key: string, item: unknown, context?: JsonParseContext,
  ): unknown {
    if (key === "seed" && typeof item === "number" && !Number.isSafeInteger(item)) {
      sources.set(this, context?.source);
    }
    return item;
  });
  return normalizeSeedTokens(value, sources);
}

/**
 * Write bigint values as exact JSON integers, preserving ordinary JSON behaviour.
 *
 * @param value - The request, workspace or exported evidence.
 * @param space - Optional indentation for an exported document.
 * @returns JSON text with exact numeric seed tokens.
 * @throws {Error} When the browser cannot emit exact integers or the root is not JSON.
 */
export function stringifyStudioJson(value: unknown, space?: number): string {
  const text: unknown = JSON.stringify(value, (key: string, item: unknown): unknown => {
    if (key === "seed" && typeof item === "number" && !Number.isFinite(item)) {
      throw new Error("Studio JSON cannot replace a non-finite seed with a default");
    }
    const integer = typeof item === "bigint" ? item
      : key === "seed" && typeof item === "number" && Number.isInteger(item)
        && !Number.isSafeInteger(item) ? BigInt(item) : undefined;
    if (integer === undefined) return item;
    if (!supportsRawNumbers(JSON)) {
      throw new Error("This browser cannot write large Studio seeds without precision loss");
    }
    return JSON.rawJSON(integer.toString());
  }, space);
  if (typeof text !== "string") throw new Error("Studio JSON requires a serialisable root value");
  return text;
}
