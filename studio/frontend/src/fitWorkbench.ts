// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fit workbench: reading recordings and results

/**
 * Turn files and form fields into a fit request, and a result into rows.
 *
 * Everything here refuses rather than guesses: a recording file with a row
 * that is not two numbers, or a fixed value that is not a number, is reported
 * with where it breaks, and nothing is sent.
 */

import type { FitRecording, FitResult } from "./api/fitsApi";

/** A value read, or the reason it could not be. */
export type Parsed<T> = { ok: true; value: T } | { ok: false; message: string };

/**
 * Read one recording from CSV text: a current and an observed value per line.
 *
 * A first line that is not two numbers is taken as a header; blank lines are
 * skipped.
 *
 * @param name - The recording's name.
 * @param text - The file's text.
 * @returns The recording, or the line that breaks it.
 */
export function parseRecordingCsv(name: string, text: string): Parsed<FitRecording> {
  const current: number[] = [];
  const observed: number[] = [];
  const lines = text.split(/\r?\n/);
  for (const [index, line] of lines.entries()) {
    if (line.trim() === "") continue;
    const cells = line.split(",").map((cell) => cell.trim());
    const numbers = cells.map(Number);
    const valid = cells.length === 2 && cells.every((cell) => cell !== "") && numbers.every(Number.isFinite);
    if (!valid) {
      if (index === 0) continue;
      return { ok: false, message: `${name}, line ${index + 1}: expected "current,observed" numbers` };
    }
    current.push(numbers[0] ?? 0);
    observed.push(numbers[1] ?? 0);
  }
  if (current.length < 2) {
    return { ok: false, message: `${name}: a recording needs at least two samples` };
  }
  return { ok: true, value: { name, current, observed } };
}

/**
 * Read fixed parameters, one `name=value` per line.
 *
 * @param text - The field's text.
 * @returns The values, or the line that breaks them.
 */
export function parseFixed(text: string): Parsed<Record<string, number>> {
  const fixed: Record<string, number> = {};
  for (const [index, line] of text.split(/\r?\n/).entries()) {
    if (line.trim() === "") continue;
    const [name = "", value = "", ...rest] = line.split("=").map((part) => part.trim());
    const number = Number(value);
    if (name === "" || value === "" || rest.length > 0 || !Number.isFinite(number)) {
      return { ok: false, message: `Fixed parameters, line ${index + 1}: expected name=number` };
    }
    fixed[name] = number;
  }
  return { ok: true, value: fixed };
}

/** One fitted parameter as the panel lists it. */
export interface FitParameterRow {
  name: string;
  value: number;
  standardError: number | null;
}

/**
 * List the fitted parameters with their standard errors, when there are any.
 *
 * @param result - The fit.
 * @returns One row per fitted parameter.
 */
export function fitParameterRows(result: FitResult): FitParameterRow[] {
  const errors = result.uncertainty.standard_errors;
  return Object.entries(result.fitted).map(([name, value]) => ({
    name,
    value,
    standardError: errors === null ? null : errors[name] ?? null,
  }));
}

/**
 * State what the data leave unconstrained, one sentence per direction.
 *
 * @param result - The fit.
 * @returns The sentences; empty when every parameter is identifiable.
 */
export function fitIdentifiabilityNotes(result: FitResult): string[] {
  const diagnosis = result.identifiability;
  if (diagnosis.identifiable) return [];
  if (diagnosis.reason !== undefined) return [`Not identifiable: ${diagnosis.reason}.`];
  return (diagnosis.unconstrained_directions ?? []).map((direction) => {
    const combination = Object.entries(direction.direction)
      .map(([name, weight]) => `${weight >= 0 ? "+" : "−"}${Math.abs(weight).toFixed(2)}·${name}`)
      .join(" ");
    return `The data do not constrain the combination ${combination} (in search space); no standard errors are stated.`;
  });
}
