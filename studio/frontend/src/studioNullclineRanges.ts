// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Nullcline domain provenance

import { studioExperimentKey } from "./studioExperimentKey";
import { studioSimulationConfigInput } from "./studioSimulationConfigInput";
import type { StudioState } from "./stores/studioTypes";

/**
 * Resolve the existing nullcline domain policy using only a current trace.
 *
 * Without a current simulation use the established fixed domains. A current
 * but malformed trace is an error, not permission to hide it with a default.
 * Extrema are scanned without spreading unbounded traces onto the call stack.
 *
 * @param state - Captured experiment and its last simulation.
 * @param first - First state variable.
 * @param second - Second state variable.
 * @returns Domains in the variables' native units with existing padding.
 * @throws {Error} When a current trace lacks finite, nonempty variable samples.
 */
export function studioNullclineRanges(state: StudioState, first: string, second: string): Record<string, [number, number]> {
  if (state.result === null || state.resultExperimentKey === null
    || state.resultExperimentKey !== studioExperimentKey(studioSimulationConfigInput(state))) {
    return { [first]: [-80, 40], [second]: [-2, 2] };
  }
  const ranges: Record<string, [number, number]> = {};
  for (const [variable, padding] of [[first, 10], [second, 0.5]] as const) {
    const values = state.result.states[variable];
    if (!values?.length) throw new Error(`Current simulation has no samples for ${variable}`);
    let min = Infinity, max = -Infinity;
    for (const value of values) {
      if (!Number.isFinite(value)) throw new Error(`Current simulation has non-finite samples for ${variable}`);
      min = Math.min(min, value); max = Math.max(max, value);
    }
    const lower = min - padding, upper = max + padding;
    if (!Number.isFinite(lower) || !Number.isFinite(upper) || lower >= upper) {
      throw new Error(`Current simulation cannot define a finite domain for ${variable}`);
    }
    ranges[variable] = [lower, upper];
  }
  return ranges;
}
