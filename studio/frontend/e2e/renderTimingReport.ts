// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — render timing artefact: statistics and schema

/**
 * The statistics and the artefact of the live render timing spec.
 *
 * A measure keeps every sample it took, in order, with its p50 and p95 by the
 * nearest-rank method; it needs at least {@link MIN_SAMPLES} samples. The
 * artefact (schema {@link RENDER_TIMING_SCHEMA}) records when and where the
 * run happened: host, CPU model, load average before and after, browser,
 * commit, and whether the run had reserved cores. A run without them is
 * `functional-non-isolated`: its numbers describe the code on that host at
 * that load, not an isolated benchmark.
 */

export const RENDER_TIMING_SCHEMA = "studio.render-timing.v1";
export const MIN_SAMPLES = 10;

/** How the run was isolated from other work on its host. */
export type Isolation = "functional-non-isolated" | "isolated-reserved-cores";

/** Where and on what a run took its samples. */
export interface RenderTimingEnvironment {
  readonly host: string;
  readonly cpuModel: string;
  /** The 1, 5 and 15 minute load averages before the first workload. */
  readonly loadBefore: readonly number[];
  /** The same after the last workload. */
  readonly loadAfter: readonly number[];
  readonly browser: string;
  readonly commit: string;
  readonly isolation: Isolation;
}

/** The samples of one measure under one workload, in milliseconds. */
export interface MeasureSamples {
  readonly name: string;
  readonly workload: string;
  readonly samplesMs: readonly number[];
}

/** One measure with its count and percentiles. */
export interface MeasureSummary extends MeasureSamples {
  readonly count: number;
  readonly p50Ms: number;
  readonly p95Ms: number;
}

/** The artefact a run writes. */
export interface RenderTimingArtefact {
  readonly schema: typeof RENDER_TIMING_SCHEMA;
  readonly recordedAt: string;
  readonly environment: RenderTimingEnvironment;
  readonly measures: readonly MeasureSummary[];
}

/** Input the statistics or the schema refuse. */
export class RenderTimingError extends Error {}

/**
 * Return the nearest-rank percentile of sorted samples.
 *
 * @param sorted - The samples, ascending.
 * @param p - The percentile, in (0, 100].
 * @returns The sample at rank `ceil(p / 100 * n)`.
 */
export function percentile(sorted: readonly number[], p: number): number {
  if (!(p > 0 && p <= 100)) {
    throw new RenderTimingError(`the percentile must lie in (0, 100], got ${String(p)}`);
  }
  const value = sorted[Math.ceil((p / 100) * sorted.length) - 1];
  if (value === undefined) throw new RenderTimingError("no samples to take a percentile of");
  return value;
}

/**
 * Summarise one measure.
 *
 * @param measure - Its samples.
 * @returns The samples as taken, their count, p50 and p95.
 */
export function summarise(measure: MeasureSamples): MeasureSummary {
  const label = `${measure.name} (${measure.workload})`;
  for (const sample of measure.samplesMs) {
    if (!Number.isFinite(sample) || sample < 0) {
      throw new RenderTimingError(`${label} has a sample that is not a duration: ${String(sample)}`);
    }
  }
  if (measure.samplesMs.length < MIN_SAMPLES) {
    throw new RenderTimingError(
      `${label} has ${String(measure.samplesMs.length)} samples; a measure needs at least ${String(MIN_SAMPLES)}`,
    );
  }
  const sorted = [...measure.samplesMs].sort((a, b) => a - b);
  return {
    name: measure.name,
    workload: measure.workload,
    samplesMs: measure.samplesMs,
    count: measure.samplesMs.length,
    p50Ms: percentile(sorted, 50),
    p95Ms: percentile(sorted, 95),
  };
}

/**
 * Build the artefact of one run.
 *
 * @param measures - The samples of every measure taken.
 * @param environment - Where and on what they were taken.
 * @param recordedAt - The UTC time of the run.
 * @returns The artefact.
 */
export function buildArtefact(
  measures: readonly MeasureSamples[],
  environment: RenderTimingEnvironment,
  recordedAt: string,
): RenderTimingArtefact {
  if (measures.length === 0) throw new RenderTimingError("the artefact names no measure");
  return { schema: RENDER_TIMING_SCHEMA, recordedAt, environment, measures: measures.map(summarise) };
}

/**
 * Return the isolation label of a run from the reserved core set it was given.
 *
 * @param reservedCores - The value of `SC_NEUROCORE_TIMING_RESERVED_CORES`, if any.
 * @returns `isolated-reserved-cores` only when a core set was named.
 */
export function isolationLabel(reservedCores: string | undefined): Isolation {
  return reservedCores !== undefined && reservedCores.trim() !== "" ? "isolated-reserved-cores" : "functional-non-isolated";
}

/**
 * Return how the artefact names the code it measured.
 *
 * @param head - The commit the checkout is at.
 * @param status - What `git status --porcelain --untracked-files=no` printed for that checkout.
 * @returns The commit alone when no tracked file differs from it; otherwise the
 *   commit followed by `with uncommitted changes`, so an artefact never names a
 *   commit as the code it measured when it measured something else.
 */
export function commitLabel(head: string, status: string): string {
  return status.trim() === "" ? head : `${head} with uncommitted changes`;
}

/**
 * Return a field of an object read from JSON when it is text.
 *
 * @param record - The object.
 * @param key - The field.
 * @param where - Its path, for the message.
 * @returns The text.
 */
function text(record: Record<string, unknown>, key: string, where: string): string {
  const value = record[key];
  if (typeof value !== "string") throw new RenderTimingError(`the artefact's ${where}${key} is not text`);
  return value;
}

/**
 * Return a load-average triple read from JSON.
 *
 * @param record - The environment.
 * @param key - `loadBefore` or `loadAfter`.
 * @returns The three numbers.
 */
function loads(record: Record<string, unknown>, key: string): number[] {
  const value = record[key];
  if (!Array.isArray(value) || value.length !== 3 || !value.every((item) => typeof item === "number")) {
    throw new RenderTimingError(`the artefact's environment.${key} is not three numbers`);
  }
  return value;
}

/**
 * Return whether a value read from JSON is an object (not null, not a list).
 *
 * @param value - Any value.
 * @returns Whether it is a plain object.
 */
function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/**
 * Read an artefact back from parsed JSON, checking every field and every summary against its samples.
 *
 * @param value - The parsed artefact.
 * @returns The artefact.
 */
export function parseArtefact(value: unknown): RenderTimingArtefact {
  if (!isObject(value)) throw new RenderTimingError("the artefact is not an object");
  if (value.schema !== RENDER_TIMING_SCHEMA) {
    throw new RenderTimingError(`the artefact's schema is not ${RENDER_TIMING_SCHEMA}`);
  }
  const recordedAt = text(value, "recordedAt", "");
  const raw = value.environment;
  if (!isObject(raw)) throw new RenderTimingError("the artefact's environment is not an object");
  const isolation = raw.isolation;
  if (isolation !== "functional-non-isolated" && isolation !== "isolated-reserved-cores") {
    throw new RenderTimingError("the artefact's environment.isolation is not a known label");
  }
  const environment: RenderTimingEnvironment = {
    host: text(raw, "host", "environment."),
    cpuModel: text(raw, "cpuModel", "environment."),
    loadBefore: loads(raw, "loadBefore"),
    loadAfter: loads(raw, "loadAfter"),
    browser: text(raw, "browser", "environment."),
    commit: text(raw, "commit", "environment."),
    isolation,
  };
  if (!Array.isArray(value.measures)) throw new RenderTimingError("the artefact's measures are not a list");
  const measures = value.measures.map((item: unknown, index: number): MeasureSummary => {
    if (!isObject(item)) throw new RenderTimingError(`measure ${String(index)} is not an object`);
    if (!Array.isArray(item.samplesMs)) throw new RenderTimingError(`measure ${String(index)} has no list of samples`);
    if (typeof item.name !== "string" || typeof item.workload !== "string") {
      throw new RenderTimingError(`measure ${String(index)} has no name and workload`);
    }
    const summary = summarise({ name: item.name, workload: item.workload, samplesMs: item.samplesMs as number[] });
    if (item.count !== summary.count || item.p50Ms !== summary.p50Ms || item.p95Ms !== summary.p95Ms) {
      throw new RenderTimingError(`measure ${String(index)} (${item.name}, ${item.workload}) does not match its samples`);
    }
    return summary;
  });
  if (measures.length === 0) throw new RenderTimingError("the artefact names no measure");
  return { schema: RENDER_TIMING_SCHEMA, recordedAt, environment, measures };
}
