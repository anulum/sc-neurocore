// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio characterisation store state helpers

/**
 * Characterisation's state, and the guards that decide what a result is.
 *
 * Characterisation runs long enough to report progress, so its transitions
 * carry a message as well as a phase. The larger half of this module is type
 * guards: the response is validated field by field before it is taken into the
 * store, because a partially-shaped response accepted here would surface as an
 * unreadable plot rather than as a failed request.
 */
import type { CharacterizeResponse, ProgressMessage, SpikeStats } from "./api/client";

/** What a characterisation is asked to run over. */
export interface CharacterizeRequestInput {
  current: number;
  dt: number;
  duration: number;
  modelParams: Record<string, number>;
  selectedModelName: string;
}

/** A characterisation has begun. */
export interface CharacterizeRunStartStatePatch {
  activeTab: "characterize";
  error: null;
  isSimulating: true;
  progressMsg: string;
  progressPct: number;
}

/** A progress message arrived while it runs. */
export interface CharacterizeProgressStatePatch {
  progressMsg: string;
  progressPct: number;
}

/** A characterisation finished, with its result. */
export interface CharacterizeCompleteStatePatch {
  charResult: CharacterizeResponse;
  isSimulating: false;
  progressMsg: "";
  progressPct: 100;
}

/** A characterisation failed, with the message to show. */
export interface CharacterizeFailureStatePatch {
  error: string;
  isSimulating: false;
  progressMsg: "";
  progressPct: 0;
}

/**
 * Build the request a characterisation is run with.
 *
 * @param input - The experiment to characterise.
 * @returns The request body.
 */
export function characterizeRequestConfig(
  input: CharacterizeRequestInput,
): Record<string, unknown> {
  return {
    current: input.current,
    dt: input.dt,
    duration: input.duration,
    name: input.selectedModelName,
    params: input.modelParams,
  };
}

/**
 * Mark a characterisation as begun.
 *
 * @returns The patch.
 */
export function characterizeRunStartState(): CharacterizeRunStartStatePatch {
  return {
    activeTab: "characterize",
    error: null,
    isSimulating: true,
    progressMsg: "Starting characterisation...",
    progressPct: 0,
  };
}

/**
 * Show a progress message without changing the phase.
 *
 * @param message - What to show.
 * @returns The patch.
 */
export function characterizeProgressState(
  message: ProgressMessage,
): CharacterizeProgressStatePatch {
  return {
    progressMsg: typeof message.msg === "string" ? message.msg : "",
    progressPct: percentValue(message.pct),
  };
}

/**
 * Take a finished characterisation into the store.
 *
 * @param result - The characterisation.
 * @returns The patch.
 */
export function characterizeCompleteState(
  result: CharacterizeResponse,
): CharacterizeCompleteStatePatch {
  return {
    charResult: result,
    isSimulating: false,
    progressMsg: "",
    progressPct: 100,
  };
}

/**
 * Report a failed characterisation.
 *
 * @param error - Whatever was thrown or rejected.
 * @param fallbackMessage - What to show when the error carries no message.
 * @returns The patch.
 */
export function characterizeFailureState(
  error: unknown,
  fallbackMessage = "Characterisation failed",
): CharacterizeFailureStatePatch {
  return {
    error: errorMessage(error, fallbackMessage),
    isSimulating: false,
    progressMsg: "",
    progressPct: 0,
  };
}

/**
 * The progress message for one step of a characterisation.
 *
 * @param message - The step's message.
 * @returns The patch.
 */
export function characterizeProgressMessageState(
  message: ProgressMessage,
): CharacterizeProgressStatePatch | CharacterizeCompleteStatePatch | CharacterizeFailureStatePatch | null {
  if (message.type === "progress") return characterizeProgressState(message);
  if (message.type === "complete") {
    return isCharacterizeResponse(message.result)
      ? characterizeCompleteState(message.result)
      : characterizeFailureState("Malformed characterisation result");
  }
  if (message.type === "error") {
    return characterizeFailureState(message.msg, "Characterisation failed");
  }
  return null;
}

/**
 * Read a progress percentage, ignoring anything that is not one.
 *
 * @param value - The reported value.
 * @returns The percentage, or `null`.
 */
function percentValue(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
}

/**
 * The message to show for a thrown value.
 *
 * @param error - Whatever was thrown.
 * @param fallbackMessage - What to show when it carries no message.
 * @returns The message.
 */
function errorMessage(error: unknown, fallbackMessage: string): string {
  if (error instanceof Error && error.message.length > 0) return error.message;
  return typeof error === "string" && error.length > 0 ? error : fallbackMessage;
}

/**
 * Whether a value is a complete characterisation.
 *
 * Checked field by field: a partially-shaped response accepted here would
 * surface as an unreadable plot rather than as a failed request.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isCharacterizeResponse(value: unknown): value is CharacterizeResponse {
  const record = recordValue(value);
  return isPattern(record.pattern)
    && isFICurve(record.fi_curve)
    && (record.threshold_current === null || finiteNumber(record.threshold_current))
    && finiteNumber(record.max_rate)
    && isStateRanges(record.state_ranges)
    && isTopSensitivities(record.top_sensitivities)
    && finiteNumber(record.spike_count)
    && isSpikeStats(record.stats);
}

/**
 * Whether a value is a firing-pattern classification.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isPattern(value: unknown): boolean {
  const record = recordValue(value);
  return typeof record.pattern === "string" && typeof record.description === "string";
}

/**
 * Whether a value is an f-I curve.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isFICurve(value: unknown): boolean {
  const record = recordValue(value);
  return numberArray(record.currents) && numberArray(record.rates);
}

/**
 * Whether a value is a record of state-variable ranges.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isStateRanges(value: unknown): boolean {
  const ranges = recordValue(value);
  return Object.values(ranges).every((item) => {
    const range = recordValue(item);
    return finiteNumber(range.min) && finiteNumber(range.max) && finiteNumber(range.mean);
  });
}

/**
 * Whether a value is a list of parameter sensitivities.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isTopSensitivities(value: unknown): boolean {
  return Array.isArray(value) && value.every((item) => {
    const sensitivity = recordValue(item);
    return typeof sensitivity.param === "string"
      && finiteNumber(sensitivity.rate_change);
  });
}

/**
 * Whether a value is a spike-statistics block.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isSpikeStats(value: unknown): value is SpikeStats {
  const stats = recordValue(value);
  return finiteNumber(stats.rate_hz)
    && nullableFiniteNumber(stats.isi_mean_ms)
    && nullableFiniteNumber(stats.isi_cv)
    && (stats.isi_histogram === null || isISIHistogram(stats.isi_histogram));
}

/**
 * Whether a value is an inter-spike-interval histogram.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isISIHistogram(value: unknown): boolean {
  const histogram = recordValue(value);
  return numberArray(histogram.counts) && numberArray(histogram.edges);
}

/**
 * Whether a value is a finite number or `null`.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function nullableFiniteNumber(value: unknown): boolean {
  return value === null || finiteNumber(value);
}

/**
 * Whether a value is a finite number.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * Whether a value is a list of finite numbers.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function numberArray(value: unknown): boolean {
  return Array.isArray(value) && value.every(finiteNumber);
}

/**
 * Whether a value is a plain object.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function recordValue(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}
