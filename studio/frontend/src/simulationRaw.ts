// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio raw-trace accessors

import type { SimulateResponse } from "./api/client";

/**
 * Whether the result carries the full-resolution raw block.
 */
export function hasRawTraces(result: SimulateResponse): boolean {
  return result.raw?.included === true && result.raw.states !== undefined;
}

/**
 * Full-resolution trace of one scalar state.
 *
 * Uses the ``raw`` block when the result carries it; a legacy result without
 * raw custody falls back to its (display) ``states`` field.
 */
export function fullStateTrace(result: SimulateResponse, name: string): number[] | undefined {
  if (hasRawTraces(result)) {
    return result.raw?.states?.[name];
  }
  return result.states[name];
}

/**
 * Names of the scalar states that have a full-resolution trace.
 */
export function fullStateNames(result: SimulateResponse): string[] {
  if (hasRawTraces(result)) {
    return Object.keys(result.raw?.states ?? {});
  }
  return Object.keys(result.states);
}

/**
 * Full-resolution drive samples (raw when present, else the display trace).
 */
export function fullDriveTrace(result: SimulateResponse): number[] {
  if (hasRawTraces(result) && result.raw?.drive !== undefined) {
    return result.raw.drive;
  }
  return result.current_trace;
}

/**
 * Post-step sample times of the full-resolution traces: ``(index + 1) * dt``.
 */
export function fullSampleTimes(result: SimulateResponse): number[] {
  if (hasRawTraces(result)) {
    return Array.from({ length: result.raw?.n_steps ?? 0 }, (_, index) => (index + 1) * result.dt);
  }
  return result.time;
}

/**
 * Raw step index of the display point nearest to a sample time.
 *
 * Display points carry their raw step in ``display.sample_index``; a legacy
 * result maps the time back through ``dt`` and the post-step clock.
 */
export function rawStepAtTime(result: SimulateResponse, timeMs: number): number {
  const times = result.time;
  if (times.length === 0) {
    return 0;
  }
  let lo = 0;
  let hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < timeMs) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const candidate = lo > 0 && Math.abs(times[lo - 1] - timeMs) <= Math.abs(times[lo] - timeMs) ? lo - 1 : lo;
  const sampleIndex = result.display?.sample_index;
  if (sampleIndex !== undefined && sampleIndex.length === times.length) {
    return sampleIndex[candidate];
  }
  return Math.min(Math.max(Math.round(timeMs / result.dt) - 1, 0), result.n_steps - 1);
}

/**
 * Display array position of a sample time (for reading display arrays).
 */
export function displayPositionAtTime(result: SimulateResponse, timeMs: number): number {
  const times = result.time;
  if (times.length === 0) {
    return 0;
  }
  let lo = 0;
  let hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < timeMs) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo > 0 && Math.abs(times[lo - 1] - timeMs) <= Math.abs(times[lo] - timeMs) ? lo - 1 : lo;
}
