// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Conversion run result reading

/**
 * What a finished conversion run reports about the network it converted to.
 *
 * A conversion run's `val_accuracy` is the converted network's; the source
 * ANN's accuracy and the difference travel beside it in the same final
 * metrics. They are read together or not at all, so a spiking run, or a
 * status this build cannot read, shows no conversion result.
 */

/** The converted network's validation accuracy beside its source's. */
export interface TrainingConversionResult {
  val_accuracy: number;
  source_val_accuracy: number;
  conversion_accuracy_drop: number;
  /** Accuracy with coefficients rounded for the run's target profile, when one was named. */
  target_accuracy?: number;
}

/**
 * Say whether a value is a fraction in [lower, 1].
 *
 * @param value - Any value.
 * @param lower - The smallest accepted value.
 * @returns Whether it is a finite number in range.
 */
function fraction(value: unknown, lower: number): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= lower && value <= 1;
}

/**
 * Read a finished run's conversion result from its final metrics.
 *
 * @param finalMetrics - The `final_metrics` of a training status.
 * @returns The result, or `null` for a spiking run or unreadable metrics.
 */
export function readTrainingConversionResult(finalMetrics: unknown): TrainingConversionResult | null {
  if (typeof finalMetrics !== "object" || finalMetrics === null || Array.isArray(finalMetrics)) return null;
  const metrics = finalMetrics as Record<string, unknown>;
  const { val_accuracy: converted, source_val_accuracy: source, conversion_accuracy_drop: drop } = metrics;
  if (!fraction(converted, 0) || !fraction(source, 0) || !fraction(drop, -1)) return null;
  const target = metrics.target_accuracy;
  return {
    val_accuracy: converted, source_val_accuracy: source, conversion_accuracy_drop: drop,
    ...(fraction(target, 0) ? { target_accuracy: target } : {}),
  };
}
