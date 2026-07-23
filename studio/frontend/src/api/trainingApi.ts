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

export const fetchSurrogates = () => get<SurrogateInfo[]>("/training/surrogates");

export const fetchCellTypes = () => get<CellTypeInfo[]>("/training/cell-types");

export const startTraining = (config: Partial<TrainingConfig>) =>
  post<{ job_id: string; status: string }>("/training/start", config);

export const stopTraining = (jobId: string) =>
  post<{ job_id: string; status: string }>("/training/stop", { job_id: jobId });

export const fetchTrainingStatus = (jobId: string) =>
  get<TrainingJobStatus>(`/training/status/${jobId}`);

export const fetchTrainingJobs = () => get<TrainingJobSummary[]>("/training/jobs");

export const exportTrainingCheckpoint = (jobId: string) =>
  get<TrainingCheckpointPayload>(`/training/checkpoint/${jobId}`);

export const importTrainingCheckpoint = (checkpoint: TrainingCheckpointPayload) =>
  post<TrainingCheckpointImportResponse>("/training/checkpoint/import", checkpoint);

export const restoreTrainingWeights = (
  sourceJobId: string,
  expectedConfigSha256?: string,
) =>
  post<TrainingWeightRestoreResult>("/studio/training/weight-restore", {
    source_job_id: sourceJobId,
    ...(expectedConfigSha256 ? { expected_config_sha256: expectedConfigSha256 } : {}),
  });

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

