// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — say when a network's rate is set by the time step, not the model

/**
 * Populations of a graph run that fire on most steps.
 *
 * A population whose neurons spike on half the steps or more is not showing
 * the model's dynamics: the time step caps its rate at 1000 / dt Hz, and the
 * number moves with dt. The canvas's defaults once ran at 1290 Hz per neuron
 * at dt 0.5 ms and 6340 Hz at 0.1 ms, and the summary printed both as rates.
 */

/** Fraction of steps above which a population is called saturated. */
export const SATURATED_STEP_FRACTION = 0.5;

/** One saturated population. */
export interface SaturatedPopulation {
  label: string;
  /** Share of steps on which an average neuron spiked, 0–1. */
  stepFraction: number;
}

/**
 * Find the populations whose neurons spike on most steps.
 *
 * @param populations - Each population's label and mean rate in Hz.
 * @param dtMs - The step the run used, in ms.
 * @returns The saturated populations, in order.
 */
export function saturatedPopulations(
  populations: readonly { label: string; mean_rate_hz: number }[],
  dtMs: number,
): SaturatedPopulation[] {
  if (!(dtMs > 0)) return [];
  return populations
    .map((population) => ({ label: population.label, stepFraction: (population.mean_rate_hz * dtMs) / 1000 }))
    .filter((population) => population.stepFraction >= SATURATED_STEP_FRACTION);
}

/**
 * Say which populations saturated, and what that means, in one sentence.
 *
 * @param saturated - From {@link saturatedPopulations}.
 * @returns The sentence, or `null` when none saturated.
 */
export function saturationWarning(saturated: readonly SaturatedPopulation[]): string | null {
  if (saturated.length === 0) return null;
  const which = saturated
    .map((population) => `${population.label} spikes on ${Math.round(population.stepFraction * 100)} % of steps`)
    .join("; ");
  return `${which}: the time step sets ${saturated.length === 1 ? "this rate" : "these rates"}, ` +
    "not the model. The network has run away; weaken excitation or strengthen inhibition.";
}
