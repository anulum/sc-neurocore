// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Training request body per model kind

/**
 * What a training request sends for each kind of run.
 *
 * A conversion run builds no spiking cells, and the server refuses a request
 * that configures them, so the form's surrogate and cell flags are left out of
 * a conversion request rather than sent and silently ignored. A spiking request
 * sends no kind at all, which is what it always sent.
 */

import type { TrainingConfig, TrainingModelKind, TrainingTargetProfile } from "./api/client";
import type { StudioProjectTrainingConfig } from "./studioProjectState";

/** Static datasets the conversion route encodes as rates in [0, 1]. */
export const CONVERSION_DATASETS: readonly string[] = ["synthetic", "mnist"];

/** Largest QCFS step budget the server accepts. */
export const QCFS_STEP_LIMIT = 2 ** 32 - 1;

/** The server's cell settings, which a conversion run leaves at their unused defaults. */
export const CELL_DEFAULTS = { surrogate: "atan_surrogate", learn_beta: false, learn_threshold: false } as const;

/**
 * Read a stored model kind, keeping only a kind other than the default.
 *
 * @param value - A kind from a saved workspace or a retained job configuration.
 * @returns `qcfs_conversion`, or `undefined` for a spiking run.
 */
export function readTrainingModelKind(value: unknown): TrainingModelKind | undefined {
  return value === "qcfs_conversion" ? "qcfs_conversion" : undefined;
}

/**
 * Read a stored target profile name, keeping only a non-empty name.
 *
 * @param value - A name from a saved workspace or a retained job configuration.
 * @returns The name, or `undefined`.
 */
export function readTrainingTargetProfile(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

/**
 * Read the listed target profiles, keeping only entries with a name and format.
 *
 * @param value - The `/api/training/target-profiles` response.
 * @returns The readable profiles in server order.
 */
export function readTrainingTargetProfiles(value: unknown): TrainingTargetProfile[] {
  if (!Array.isArray(value)) return [];
  return value.filter((entry: unknown): entry is TrainingTargetProfile => {
    if (typeof entry !== "object" || entry === null) return false;
    const data = entry as Record<string, unknown>;
    return readTrainingTargetProfile(data.name) !== undefined && typeof data.q_format === "string";
  });
}

/**
 * Say what the conversion route cannot run in these settings, or nothing.
 *
 * @param config - The settings as edited.
 * @returns A sentence naming the problem, or `null` for a spiking run or a runnable conversion.
 */
export function conversionProblem(config: StudioProjectTrainingConfig): string | null {
  if (config.model_kind !== "qcfs_conversion") return null;
  if (!CONVERSION_DATASETS.includes(config.dataset)) {
    return "A conversion run trains on the synthetic or MNIST dataset.";
  }
  if (config.timesteps > QCFS_STEP_LIMIT) {
    return `A conversion run's timestep budget is at most ${String(QCFS_STEP_LIMIT)}.`;
  }
  return null;
}

/**
 * Build the body `POST /api/training/start` receives for these settings.
 *
 * @param config - The settings as edited.
 * @returns The settings, without the cell settings on a conversion run and without a kind or
 *   target on a spiking one.
 */
export function trainingRequestBody(config: StudioProjectTrainingConfig): Partial<TrainingConfig> {
  const body: Partial<TrainingConfig> = { ...config, hidden: [...config.hidden] };
  if (config.model_kind === "qcfs_conversion") {
    delete body.surrogate;
    delete body.learn_beta;
    delete body.learn_threshold;
  } else {
    delete body.model_kind;
    delete body.target_profile;
  }
  return body;
}
