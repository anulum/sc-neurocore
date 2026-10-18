// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Numeric model inputs with exact legacy seed custody

import { readSimulationSeed } from "./simulationSeed";

/** Constructor overrides; legacy seed entries may exceed the safe Number range. */
export type ModelParameters = Record<string, number | bigint>;

/**
 * Read stored model overrides, preserving exact integer seeds.
 *
 * @param value - The stored parameter map.
 * @returns Finite numeric overrides plus exact seed values.
 * @throws {Error} When an explicit legacy seed is outside the simulation seed domain.
 */
export function readModelParameters(value: unknown): ModelParameters {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return {};
  const result: ModelParameters = {};
  for (const [name, item] of Object.entries(value)) {
    if (name === "seed") {
      const seed = readSimulationSeed(item);
      if (seed !== null) result.seed = seed;
    } else if (typeof item === "number" && Number.isFinite(item)) {
      result[name] = item;
    }
  }
  return result;
}

/**
 * Read a slider's scalar value without converting a large integer seed to Number.
 *
 * @param parameters - Current constructor overrides.
 * @param name - The non-seed parameter or writable state being edited.
 * @param fallback - Its declared default.
 * @returns The stored numeric scalar or its default.
 * @throws {Error} When a large integer is offered to a numeric slider.
 */
export function modelParameterValue(parameters: ModelParameters, name: string, fallback: number): number {
  const value = parameters[name];
  if (typeof value === "bigint") {
    throw new Error("Large model seeds must be edited through the seed control");
  }
  return value ?? fallback;
}
