// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio characterisation store state helpers
import type { CharacterizeResponse, ProgressMessage, SpikeStats } from "./api/client";

export interface CharacterizeRequestInput {
  current: number;
  dt: number;
  duration: number;
  modelParams: Record<string, number>;
  selectedModelName: string;
}

export interface CharacterizeRunStartStatePatch {
  activeTab: "characterize";
  error: null;
  isSimulating: true;
  progressMsg: string;
  progressPct: number;
}

export interface CharacterizeProgressStatePatch {
  progressMsg: string;
  progressPct: number;
}

export interface CharacterizeCompleteStatePatch {
  charResult: CharacterizeResponse;
  isSimulating: false;
  progressMsg: "";
  progressPct: 100;
}

export interface CharacterizeFailureStatePatch {
  error: string;
  isSimulating: false;
  progressMsg: "";
  progressPct: 0;
}

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

export function characterizeRunStartState(): CharacterizeRunStartStatePatch {
  return {
    activeTab: "characterize",
    error: null,
    isSimulating: true,
    progressMsg: "Starting characterisation...",
    progressPct: 0,
  };
}

export function characterizeProgressState(
  message: ProgressMessage,
): CharacterizeProgressStatePatch {
  return {
    progressMsg: typeof message.msg === "string" ? message.msg : "",
    progressPct: percentValue(message.pct),
  };
}

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

function percentValue(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
}

function errorMessage(error: unknown, fallbackMessage: string): string {
  if (error instanceof Error && error.message.length > 0) return error.message;
  return typeof error === "string" && error.length > 0 ? error : fallbackMessage;
}

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

function isPattern(value: unknown): boolean {
  const record = recordValue(value);
  return typeof record.pattern === "string" && typeof record.description === "string";
}

function isFICurve(value: unknown): boolean {
  const record = recordValue(value);
  return numberArray(record.currents) && numberArray(record.rates);
}

function isStateRanges(value: unknown): boolean {
  const ranges = recordValue(value);
  return Object.values(ranges).every((item) => {
    const range = recordValue(item);
    return finiteNumber(range.min) && finiteNumber(range.max) && finiteNumber(range.mean);
  });
}

function isTopSensitivities(value: unknown): boolean {
  return Array.isArray(value) && value.every((item) => {
    const sensitivity = recordValue(item);
    return typeof sensitivity.param === "string"
      && finiteNumber(sensitivity.rate_change);
  });
}

function isSpikeStats(value: unknown): value is SpikeStats {
  const stats = recordValue(value);
  return finiteNumber(stats.rate_hz)
    && nullableFiniteNumber(stats.isi_mean_ms)
    && nullableFiniteNumber(stats.isi_cv)
    && (stats.isi_histogram === null || isISIHistogram(stats.isi_histogram));
}

function isISIHistogram(value: unknown): boolean {
  const histogram = recordValue(value);
  return numberArray(histogram.counts) && numberArray(histogram.edges);
}

function nullableFiniteNumber(value: unknown): boolean {
  return value === null || finiteNumber(value);
}

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function numberArray(value: unknown): boolean {
  return Array.isArray(value) && value.every(finiteNumber);
}

function recordValue(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}
