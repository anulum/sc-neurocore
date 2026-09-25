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
 * Name the canvas for whatever it currently shows.
 *
 * Only the trace view is described in numbers; for any other view the
 * sentence says what the view is and where its numbers are, rather than
 * pretending the picture has been put into words.
 *
 * @param viewTitle - The active view's title.
 * @param result - The run, when there is one.
 * @param showsTrace - Whether the canvas is drawing the trace view.
 * @returns The canvas's accessible name.
 */
export function plotDescription(
  viewTitle: string,
  result: SimulateResponse | null,
  showsTrace: boolean,
): string {
  if (result === null) return `${viewTitle} plot: nothing has run yet.`;
  if (showsTrace) return traceDescription(result);
  return `${viewTitle} plot. Its values are in the CSV and JSON exports; the data table describes the trace.`;
}
