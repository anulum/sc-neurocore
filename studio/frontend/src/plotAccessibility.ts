// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — What the plot canvas says to a reader who cannot see it

/**
 * Words and a table for the plot canvas.
 *
 * A canvas is pixels; a screen reader finds nothing in it. The plot therefore
 * names itself with a sentence and offers the trace as a table. Both are read
 * from the display projection the server sends, whose bucket extrema keep
 * every minimum and maximum and whose last sample is the run's last, so the
 * ranges and final values stated here are the run's own, not estimates.
 */

import type { SimulateResponse } from "./api/types";

/** One variable of a trace, summarised. */
export interface TraceDataRow {
  variable: string;
  samples: number;
  minimum: number | null;
  maximum: number | null;
  final: number | null;
}

/**
 * Format a value for reading aloud or listing.
 *
 * @param value - The number, or `null` when there is none.
 * @returns Four significant digits, or "none".
 */
export function formatReading(value: number | null): string {
  return value === null ? "none" : Number(value.toPrecision(4)).toString();
}

/**
 * Summarise each variable of a run: its range and where it ended.
 *
 * Non-finite samples are left out of the range; a variable with no finite
 * sample has none.
 *
 * @param result - The run.
 * @returns One row per variable, in the run's order.
 */
export function traceDataRows(result: SimulateResponse): TraceDataRow[] {
  return Object.entries(result.states).map(([variable, values]) => {
    const finite = values.filter((value) => Number.isFinite(value));
    const last = finite.at(-1);
    return {
      variable,
      samples: values.length,
      minimum: finite.length === 0 ? null : finite.reduce((a, b) => Math.min(a, b)),
      maximum: finite.length === 0 ? null : finite.reduce((a, b) => Math.max(a, b)),
      final: last ?? null,
    };
  });
}

/**
 * Describe the trace view in one sentence.
 *
 * @param result - The run the trace view draws.
 * @returns The sentence the canvas is named with.
 */
export function traceDescription(result: SimulateResponse): string {
  const rows = traceDataRows(result);
  const duration = result.n_steps * result.dt;
  const spikes = `${result.spike_count} ${result.spike_count === 1 ? "spike" : "spikes"}`;
  const ranges = rows
    .map((row) => `${row.variable} from ${formatReading(row.minimum)} to ${formatReading(row.maximum)}, ending at ${formatReading(row.final)}`)
    .join("; ");
  return (
    `Trace of ${rows.map((row) => row.variable).join(", ")} over ${formatReading(duration)} ms ` +
    `(${result.n_steps} steps of ${formatReading(result.dt)} ms): ${spikes}. ${ranges}.`
  );
}

/**
 * What to call a run: its model, or its position when it has no name.
 *
 * @param run - The run.
 * @param index - Its position among the runs drawn, from zero.
 * @returns The name.
 */
export function runName(run: SimulateResponse, index: number): string {
  const name = run.model_name ?? "";
  return name === "" ? `Model ${String(index + 1)}` : name;
}

/**
 * Say which drive the overlaid runs had, and that its units are each model's.
 *
 * The overlay sends one current to every model, but models do not share a
 * current unit (a conductance model reads µA/cm², an integrate-and-fire one
 * pA or a dimensionless drive), so the same number is a different stimulus
 * for each. The reader has to be told that before comparing the traces.
 *
 * @param runs - The overlaid runs.
 * @returns The sentence, or `null` when a run does not record its drive.
 */
export function multiModelDriveNote(runs: readonly SimulateResponse[]): string | null {
  const drives = runs.map((run) => run.experiment?.protocol);
  if (runs.length === 0 || drives.some((drive) => drive === undefined)) return null;
  const said = drives.map((drive) => {
    const kind = String(drive?.kind ?? "unknown");
    const current = typeof drive?.current === "number" ? formatReading(drive.current) : "none";
    const frequency = typeof drive?.frequency_hz === "number" ? ` at ${formatReading(drive.frequency_hz)} Hz` : "";
    return `${kind}, I = ${current}${frequency}`;
  });
  const drive = new Set(said).size === 1
    ? `Every model had the same drive (${said[0] ?? ""})`
    : `The drives differ (${runs.map((run, i) => `${runName(run, i)}: ${said[i] ?? ""}`).join("; ")})`;
  return `${drive}, read in each model's own current units.`;
}

/**
 * Describe the multi-model overlay in one sentence.
 *
 * @param runs - The overlaid runs, in the order they are drawn.
 * @returns The sentence the canvas is named with.
 */
export function multiModelDescription(runs: readonly SimulateResponse[]): string {
  const parts = runs.map((run, i) => {
    const variable = Object.keys(run.states)[0];
    const row = traceDataRows(run)[0];
    const name = runName(run, i);
    const range = variable === undefined || row === undefined
      ? "no recorded state"
      : `${variable} from ${formatReading(row.minimum)} to ${formatReading(row.maximum)}`;
    const spikes = `${run.spike_count} ${run.spike_count === 1 ? "spike" : "spikes"}`;
    return `${name}: ${range}, ${spikes} (${formatReading(run.stats.rate_hz)} Hz)`;
  });
  const note = multiModelDriveNote(runs);
  const steps = runs.map((run, i) => `${runName(run, i)} ${formatReading(run.dt)} ms`);
  return `Multi-model overlay of ${runs.length} ${runs.length === 1 ? "run" : "runs"}, ` +
    `each model's first state on one shared axis. ${parts.join("; ")}.` +
    `${note === null ? "" : ` ${note}`} Time steps: ${steps.join(", ")}.`;
}

/**
 * Name the canvas for whatever it currently shows.
 *
 * The trace view is described in numbers, and so is any view that passes its
 * own sentence. Any other view says what it is and that it is not in words:
 * the data table and the CSV and JSON exports hold the trace run, not the
 * view, and saying otherwise sent a reader to files without its numbers.
 *
 * @param viewTitle - The active view's title.
 * @param result - The run, when there is one.
 * @param showsTrace - Whether the canvas is drawing the trace view.
 * @param viewSentence - The view's own description, when it has one.
 * @returns The canvas's accessible name.
 */
export function plotDescription(
  viewTitle: string,
  result: SimulateResponse | null,
  showsTrace: boolean,
  viewSentence: string | null = null,
): string {
  if (viewSentence !== null) return viewSentence;
  if (result === null) return `${viewTitle} plot: nothing has run yet.`;
  if (showsTrace) return traceDescription(result);
  return `${viewTitle} plot. This view is not put into words; the data table and the CSV and ` +
    "JSON exports hold the trace run, not this view.";
}
