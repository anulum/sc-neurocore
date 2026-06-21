// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio training store state helpers
import type {
  SurrogateInfo,
  TrainingCheckpointImportResponse,
  TrainingEpochMetrics,
  TrainingWeightRestorePlan,
  TrainingWeightRestoreResult,
} from "./api/client";
import type { StudioProjectTrainingConfig } from "./studioProjectState";
import type { StudioTrainingTerminalStatus } from "./studioTrainingStream";
import type { TrainingWeightRestoreVerification } from "./trainingRestore";

export interface TrainingSurrogatesLoadedStatePatch {
  trainingSurrogates: SurrogateInfo[];
}

export interface TrainingStartStatePatch {
  activeTab: "train";
  error: null;
  trainingEpochs: [];
  trainingStatus: "starting";
  trainingWeightRestorePlan: null;
  trainingWeightRestoreVerification: null;
}

export interface TrainingStartedStatePatch {
  trainingJobId: string;
  trainingStatus: "running";
}

export interface TrainingEpochAppendedStatePatch {
  trainingEpochs: TrainingEpochMetrics[];
}

export interface TrainingTerminalStatePatch {
  trainingStatus: StudioTrainingTerminalStatus;
}

export interface TrainingStreamErrorStatePatch {
  error: string;
  trainingStatus: "failed";
}

export interface TrainingStreamDisconnectedStatePatch {
  trainingStatus: "disconnected";
}

export interface TrainingStoppingStatePatch {
  trainingStatus: "stopping";
}

export interface TrainingCheckpointImportedStatePatch {
  activeTab: "train";
  error: null;
  trainingConfig: StudioProjectTrainingConfig;
  trainingEpochs: [];
  trainingJobId: string;
  trainingStatus: string;
  trainingWeightRestorePlan: TrainingWeightRestorePlan | null;
  trainingWeightRestoreVerification: null;
}

export interface TrainingConfigUpdatedStatePatch {
  trainingConfig: StudioProjectTrainingConfig;
}

export interface TrainingWeightRestoreVerificationStartStatePatch {
  error: null;
  trainingWeightRestoreVerification: null;
}

export interface TrainingWeightRestoreVerificationLoadedStatePatch {
  trainingWeightRestoreVerification: TrainingWeightRestoreVerification;
}

export interface TrainingWeightMaterializationLoadedStatePatch {
  error: null;
  trainingWeightMaterialization: TrainingWeightRestoreResult;
}

export interface TrainingFailureStatePatch {
  error: string;
  trainingStatus?: "failed";
}

export interface TrainingPreconditionErrorStatePatch {
  error: string;
}

export interface TrainingExportSuccessStatePatch {
  error: null;
}

export function trainingSurrogatesLoadedState(
  trainingSurrogates: SurrogateInfo[],
): TrainingSurrogatesLoadedStatePatch {
  return { trainingSurrogates };
}

export function trainingStartState(): TrainingStartStatePatch {
  return {
    activeTab: "train",
    error: null,
    trainingEpochs: [],
    trainingStatus: "starting",
    trainingWeightRestorePlan: null,
    trainingWeightRestoreVerification: null,
  };
}

export function trainingStartedState(jobId: string): TrainingStartedStatePatch {
  return {
    trainingJobId: jobId,
    trainingStatus: "running",
  };
}

export function trainingEpochAppendedState(
  currentEpochs: TrainingEpochMetrics[],
  metrics: TrainingEpochMetrics,
): TrainingEpochAppendedStatePatch {
  return {
    trainingEpochs: [...currentEpochs, metrics],
  };
}

export function trainingTerminalState(
  status: StudioTrainingTerminalStatus,
): TrainingTerminalStatePatch {
  return { trainingStatus: status };
}

export function trainingStreamErrorState(message: string): TrainingStreamErrorStatePatch {
  return {
    error: message,
    trainingStatus: "failed",
  };
}

export function trainingStreamDisconnectedState(): TrainingStreamDisconnectedStatePatch {
  return { trainingStatus: "disconnected" };
}

export function trainingStoppingState(): TrainingStoppingStatePatch {
  return { trainingStatus: "stopping" };
}

export function trainingCheckpointImportedState(
  currentConfig: StudioProjectTrainingConfig,
  imported: TrainingCheckpointImportResponse,
): TrainingCheckpointImportedStatePatch {
  return {
    activeTab: "train",
    error: null,
    trainingConfig: { ...currentConfig, ...imported.config },
    trainingEpochs: [],
    trainingJobId: imported.source_job_id,
    trainingStatus: `checkpoint:${imported.source_status}`,
    trainingWeightRestorePlan: imported.weight_restore_plan,
    trainingWeightRestoreVerification: null,
  };
}

export function trainingConfigUpdatedState<K extends keyof StudioProjectTrainingConfig>(
  currentConfig: StudioProjectTrainingConfig,
  key: K,
  value: StudioProjectTrainingConfig[K],
): TrainingConfigUpdatedStatePatch {
  return {
    trainingConfig: {
      ...currentConfig,
      [key]: value,
    },
  };
}

export function trainingWeightRestoreVerificationStartState():
TrainingWeightRestoreVerificationStartStatePatch {
  return {
    error: null,
    trainingWeightRestoreVerification: null,
  };
}

export function trainingWeightRestoreVerificationLoadedState(
  trainingWeightRestoreVerification: TrainingWeightRestoreVerification,
): TrainingWeightRestoreVerificationLoadedStatePatch {
  return { trainingWeightRestoreVerification };
}

export function trainingWeightMaterializationLoadedState(
  trainingWeightMaterialization: TrainingWeightRestoreResult,
): TrainingWeightMaterializationLoadedStatePatch {
  return { error: null, trainingWeightMaterialization };
}

export function trainingFailureState(
  error: unknown,
  fallbackMessage: string,
  options: { markFailed?: boolean } = {},
): TrainingFailureStatePatch {
  return {
    error: error instanceof Error && error.message.length > 0
      ? error.message
      : fallbackMessage,
    ...(options.markFailed ? { trainingStatus: "failed" as const } : {}),
  };
}

export function trainingPreconditionErrorState(
  message: string,
): TrainingPreconditionErrorStatePatch {
  return { error: message };
}

export function trainingExportSuccessState(): TrainingExportSuccessStatePatch {
  return { error: null };
}
