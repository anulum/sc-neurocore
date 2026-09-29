// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Preregistered training criterion and verdict reading

/**
 * The acceptance criterion a training run declares before it starts.
 *
 * The rules here mirror the server's contract so the form can say what is
 * wrong before a request is sent; the server still decides. Stored criteria
 * and verdicts are read field by field, and anything this build cannot read
 * is treated as absent rather than trusted.
 */

import type {
  TrainingModelKind,
  TrainingPreregistration,
  TrainingPreregistrationMetric,
  TrainingPreregistrationVerdict,
} from "./api/client";

/** Longest rationale the server accepts. */
export const PREREGISTRATION_RATIONALE_MAX = 500;

/** Each judged metric and the direction in which it passes. */
export const PREREGISTRATION_DIRECTIONS: Record<TrainingPreregistrationMetric, "at_least" | "at_most"> = {
  val_accuracy: "at_least",
  val_loss: "at_most",
  conversion_accuracy_drop: "at_most",
};

/**
 * Say whether a value names a judged metric.
 *
 * @param value - Any value.
 * @returns Whether it is `val_accuracy` or `val_loss`.
 */
function isMetric(value: unknown): value is TrainingPreregistrationMetric {
  return value === "val_accuracy" || value === "val_loss" || value === "conversion_accuracy_drop";
}

/**
 * Read a plain object, or say the value is not one.
 *
 * @param value - Any value.
 * @returns The object, or `null`.
 */
function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown> : null;
}

/**
 * Say what is wrong with a criterion, or nothing when the server will accept it.
 *
 * @param criterion - The criterion as edited.
 * @param modelKind - The run's kind; the accuracy-drop criterion needs a conversion run.
 * @returns A sentence naming the problem, or `null`.
 */
export function preregistrationProblem(
  criterion: TrainingPreregistration,
  modelKind?: TrainingModelKind,
): string | null {
  if (!isMetric(criterion.metric)) return "Choose validation accuracy or validation loss.";
  if (criterion.metric === "conversion_accuracy_drop" && modelKind !== "qcfs_conversion") {
    return "The accuracy-drop criterion judges a conversion run only.";
  }
  const bound = criterion.threshold;
  const fraction = criterion.metric !== "val_loss";
  if (typeof bound !== "number" || !Number.isFinite(bound) || bound < 0 || (fraction && bound > 1)) {
    if (!fraction) return "A loss threshold is a finite number at or above 0.";
    return criterion.metric === "val_accuracy"
      ? "An accuracy threshold lies between 0 and 1."
      : "An accuracy-drop threshold lies between 0 and 1.";
  }
  if (criterion.rationale.length > PREREGISTRATION_RATIONALE_MAX) {
    return `State the rationale in at most ${PREREGISTRATION_RATIONALE_MAX} characters.`;
  }
  return null;
}

/**
 * Read a stored criterion, keeping only what the form edits.
 *
 * @param value - A criterion from a saved workspace or a retained job configuration.
 * @returns The criterion, or `undefined` when there is none this build can read.
 */
export function readTrainingPreregistration(value: unknown): TrainingPreregistration | undefined {
  const data = record(value);
  if (data === null || !isMetric(data.metric) || typeof data.threshold !== "number") return undefined;
  const criterion: TrainingPreregistration = {
    metric: data.metric,
    threshold: data.threshold,
    rationale: typeof data.rationale === "string" ? data.rationale : "",
  };
  // The run's kind is checked where the criterion is used, not where it is read.
  return preregistrationProblem(criterion, "qcfs_conversion") === null ? criterion : undefined;
}

/**
 * Read a finished run's verdict, or say there is none this build can trust.
 *
 * @param value - The `preregistration_verdict` of a training status.
 * @returns The verdict, or `null`.
 */
export function readTrainingPreregistrationVerdict(value: unknown): TrainingPreregistrationVerdict | null {
  const data = record(value);
  if (data?.schema_version !== "studio.training-preregistration.v1"
    || !isMetric(data.metric)
    || data.direction !== PREREGISTRATION_DIRECTIONS[data.metric]
    || typeof data.threshold !== "number" || !Number.isFinite(data.threshold)
    || !(data.observed === null || (typeof data.observed === "number" && Number.isFinite(data.observed)))
    || typeof data.passed !== "boolean"
    || typeof data.preregistration_sha256 !== "string"
    || !/^[0-9a-f]{64}$/.test(data.preregistration_sha256)) {
    return null;
  }
  return data as unknown as TrainingPreregistrationVerdict;
}
