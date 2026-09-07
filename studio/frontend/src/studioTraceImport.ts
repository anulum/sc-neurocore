// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio trace import request builder

/**
 * Reading a voltage trace out of whatever CSV the reader has.
 *
 * The parser is forgiving about shape and strict about content. Columns may be
 * separated by commas, tabs or spaces, and the **last** column of each line is
 * taken as the voltage, which is what a two-column `time, voltage` export and a
 * one-column dump both give. A line whose last column is not a number is
 * skipped rather than refused, because a header row is the common case and
 * refusing the file over it would help nobody.
 *
 * What is not forgiven is a trace too short to mean anything: fewer than ten
 * samples is not a trace, and importing one would put a line on a plot that
 * cannot be read.
 */

/** The fewest samples an imported trace can have and still be one. */
export const MIN_TRACE_IMPORT_SAMPLES = 10;

/** An imported trace: its samples, and the step between them. */
export interface StudioTraceImportRequest {
  voltage: number[];
  dt: number;
}

/**
 * Read the voltage column out of a CSV.
 *
 * @param csv - The file's text.
 * @returns The values that parsed, in file order. Lines that do not end in
 *   a number -- a header, a blank, a comment -- are skipped.
 */
export function parseStudioTraceVoltageValues(csv: string): number[] {
  const values: number[] = [];
  for (const line of csv.trim().split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const parts = trimmed.split(/[,\t\s]+/);
    const value = Number.parseFloat(parts[parts.length - 1] ?? "");
    if (Number.isFinite(value)) values.push(value);
  }
  return values;
}

/**
 * Read a CSV as an import request.
 *
 * @param csv - The file's text.
 * @param dt - The step between samples, which the file does not carry.
 * @returns The request.
 * @throws {Error} When fewer than ten samples parsed. The message says how
 *   many are needed, because the reader can act on that.
 */
export function studioTraceImportRequest(csv: string, dt: number): StudioTraceImportRequest {
  const voltage = parseStudioTraceVoltageValues(csv);
  if (voltage.length < MIN_TRACE_IMPORT_SAMPLES) {
    throw new Error(`Need at least ${MIN_TRACE_IMPORT_SAMPLES} data points`);
  }
  return { voltage, dt };
}
