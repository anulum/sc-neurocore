// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Studio API: training endpoints.
import { post, get } from "./http";
import type {
  SurrogateInfo,
  CellTypeInfo,
  TrainingConfig,
  TrainingJobStatus,
  TrainingCheckpointPayload,
  TrainingWeightRestoreResult,
  TrainingWeightAttachResult,
  TrainingWeightLiveAttachResult,
  TrainingCheckpointImportResponse,
  TrainingJobSummary,
} from "./types";

/**
 * List the surrogate gradients training can use.
 *
 * A spike is not differentiable, so training needs a stand-in derivative; this
 * is the set the server implements.
 *
 * @returns Each surrogate with its parameters.
 */
export const fetchSurrogates = () => get<SurrogateInfo[]>("/training/surrogates");

/**
 * List the cell types a trained network may be built from.
 *
 * @returns Each cell type with what it supports.
 */
export const fetchCellTypes = () => get<CellTypeInfo[]>("/training/cell-types");

/**
 * Start a training run.
 *
 * The config is partial: the server fills what is left out from its own
 * defaults and reports the resolved config back on the job.
 *
 * @param config - Whatever the caller wants to set.
 * @returns The job's identifier and its initial status.
 */
export const startTraining = (config: Partial<TrainingConfig>) =>
  post<{ job_id: string; status: string }>("/training/start", config);

/**
 * Ask a training run to stop.
 *
 * @param jobId - The job to stop.
 * @returns The job and the status it moved to.
 */
export const stopTraining = (jobId: string) =>
  post<{ job_id: string; status: string }>("/training/stop", { job_id: jobId });

/**
 * Read one training run's status and its epochs so far.
 *
 * @param jobId - The job to read.
 * @returns Its status, its config, and every epoch it has recorded.
 */
export const fetchTrainingStatus = (jobId: string) =>
  get<TrainingJobStatus>(`/training/status/${jobId}`);

/**
 * List the training runs the server knows about.
 *
 * @returns One summary per job.
 */
export const fetchTrainingJobs = () => get<TrainingJobSummary[]>("/training/jobs");

/**
 * Export a run's weights as a checkpoint document.
 *
 * @param jobId - The job to export.
 * @returns The checkpoint, with the config digest it was trained under.
 */
export const exportTrainingCheckpoint = (jobId: string) =>
  get<TrainingCheckpointPayload>(`/training/checkpoint/${jobId}`);

/**
 * Import a checkpoint document the server did not produce.
 *
 * @param checkpoint - The checkpoint to import.
 * @returns What the server made of it, including anything it refused.
 */
export const importTrainingCheckpoint = (checkpoint: TrainingCheckpointPayload) =>
  post<TrainingCheckpointImportResponse>("/training/checkpoint/import", checkpoint);

/**
 * Restore weights from an earlier run into a new one.
 *
 * `expectedConfigSha256` is how a caller says which configuration it believes
 * the weights were trained under; the server refuses rather than restoring
 * weights into a network they do not fit. Omitting it skips that check, which
 * is why it is offered at every call rather than assumed.
 *
 * @param sourceJobId - The run whose weights to take.
 * @param expectedConfigSha256 - The config digest the caller expects.
 * @returns What was restored, or the mismatch that stopped it.
 */
export const restoreTrainingWeights = (
  sourceJobId: string,
  expectedConfigSha256?: string,
) =>
  post<TrainingWeightRestoreResult>("/studio/training/weight-restore", {
    source_job_id: sourceJobId,
    ...(expectedConfigSha256 ? { expected_config_sha256: expectedConfigSha256 } : {}),
  });

/**
 * Start a run with weights from an earlier one already attached.
 *
 * @param sourceJobId - The run whose weights to take.
 * @param config - The configuration for the new run.
 * @param expectedConfigSha256 - The config digest the caller expects.
 * @returns The new run, or the mismatch that stopped it.
 */
export const attachTrainingWeights = (
  sourceJobId: string,
  config: Partial<TrainingConfig>,
  expectedConfigSha256?: string,
) =>
  post<TrainingWeightAttachResult>("/studio/training/weight-restore/attach", {
    source_job_id: sourceJobId,
    config,
    ...(expectedConfigSha256 ? { expected_config_sha256: expectedConfigSha256 } : {}),
  });

/**
 * Attach weights to a run that is already going.
 *
 * @param targetJobId - The running job to attach to.
 * @param sourceJobId - The run whose weights to take.
 * @param expectedConfigSha256 - The config digest the caller expects.
 * @returns What was attached, or the mismatch that stopped it.
 */
export const attachTrainingWeightsLive = (
  targetJobId: string,
  sourceJobId: string,
  expectedConfigSha256?: string,
) =>
  post<TrainingWeightLiveAttachResult>("/studio/training/weight-restore/attach/live", {
    target_job_id: targetJobId,
    source_job_id: sourceJobId,
    ...(expectedConfigSha256 ? { expected_config_sha256: expectedConfigSha256 } : {}),
  });
