// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import type { TrainingConfig, TrainingEpochMetrics } from "./api/client";

/** What the evidence strip states about a training run. */
export interface TrainingEvidenceModel {
  actionKind: string;
  classification: string;
  configSummary: string;
  evidenceArtifact: string;
  jobId: string;
  latestEpoch: string;
  replayRoute: string;
  status: string;
  statusArtifact: string;
}

/**
 * Describe a training run for its evidence strip.
 *
 * A run that has not been submitted shows `not submitted` and `pending`
 * rather than blanks, so the strip distinguishes a run that has not
 * started from one whose evidence failed to load.
 *
 * @param jobId - The job, once there is one.
 * @param status - Where the run is.
 * @param config - The parts of the configuration worth stating.
 * @param latestEpoch - The most recent epoch's metrics, if any.
 * @returns What the strip should state.
 */
export function buildTrainingEvidenceModel(
  jobId: string | null,
  status: string,
  config: Pick<TrainingConfig, "dataset" | "epochs" | "surrogate" | "timesteps">,
  latestEpoch: TrainingEpochMetrics | null,
): TrainingEvidenceModel {
  const submitted = jobId !== null && jobId.length > 0;
  return {
    actionKind: "studio.training.run",
    classification: "training",
    configSummary:
      `${config.dataset}, ${config.epochs} epochs, ${config.surrogate}, ${config.timesteps} steps`,
    evidenceArtifact: submitted ? "training/evidence.json" : "pending",
    jobId: submitted ? jobId : "not submitted",
    latestEpoch: latestEpoch === null ? "none" : String(latestEpoch.epoch),
    replayRoute: "POST /api/training/start",
    status,
    statusArtifact: submitted ? "training/status.json" : "pending",
  };
}
