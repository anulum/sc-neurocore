// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio training store state helpers

/**
 * The state patches a training run moves through.
 *
 * A run is long and its updates arrive on a socket, so the transitions here
 * are finer-grained than elsewhere: starting, started, each epoch, a stream
 * error, a disconnection, stopping, and each of the four ways weights can be
 * carried between runs. A stream error and a disconnection are separate
 * because one says the run reported a problem and the other says the
 * connection went away while the run may still be going.
 */
import type {
  SurrogateInfo,
  TrainingCheckpointImportResponse,
  TrainingEpochMetrics,
  TrainingWeightAttachResult,
  TrainingWeightLiveAttachResult,
  TrainingWeightRestorePlan,
  TrainingWeightRestoreResult,
  TrainingJobSummary,
} from "./api/client";
import type { StudioProjectTrainingConfig } from "./studioProjectState";
import type { StudioTrainingTerminalStatus } from "./studioTrainingStream";
import type { TrainingStopStatus } from "./studioTrainingRecovery";
import type { TrainingWeightRestoreVerification } from "./trainingRestore";

/** The surrogate gradients and cell types arrived. */
export interface TrainingSurrogatesLoadedStatePatch {
  trainingSurrogates: SurrogateInfo[];
}

/** A run is being started: the previous run's epochs and errors clear. */
export interface TrainingStartStatePatch {
  activeTab: "train";
  error: null;
  trainingEpochs: [];
  trainingJobId: null;
  trainingObservedConfig: null;
  trainingStatus: "starting";
  trainingWeightRestorePlan: null;
  trainingWeightRestoreVerification: null;
  trainingWeightMaterialization: null;
  trainingWeightAttach: null;
  trainingWeightLiveAttach: null;
}

/** The server accepted the run and named it. */
export interface TrainingStartedStatePatch {
  trainingJobId: string;
  trainingStatus: "running";
}

/** A durable run selected for observation, without changing project settings. */
export interface TrainingRecoveredStatePatch {
  error: null;
  trainingJobId: string;
  trainingStatus: string;
  trainingEpochs: [];
  trainingObservedConfig: StudioProjectTrainingConfig | null;
  trainingExperimentKey: null;
  trainingWeightRestorePlan: null;
  trainingWeightRestoreVerification: null;
  trainingWeightMaterialization: null;
  trainingWeightAttach: null;
  trainingWeightLiveAttach: null;
}

/** The durable list was loaded from the training route. */
export interface TrainingJobsLoadedStatePatch {
  trainingJobs: TrainingJobSummary[];
  trainingJobsLoading: false;
  trainingJobsError: null;
}

/** One epoch's metrics arrived, appended to the ones before it. */
export interface TrainingEpochAppendedStatePatch {
  trainingEpochs: TrainingEpochMetrics[];
}

/** The run reached a terminal status. */
export interface TrainingTerminalStatePatch {
  trainingStatus: StudioTrainingTerminalStatus;
}

/** The run reported a problem over its stream. */
export interface TrainingStreamErrorStatePatch {
  error: string;
  trainingStatus: "failed";
}

/** The stream went away; the run itself may or may not still be going. */
export interface TrainingStreamDisconnectedStatePatch {
  trainingStatus: "disconnected";
}

/** A stop was asked for and has not taken effect yet. */
export interface TrainingStopResultStatePatch {
  trainingStatus: TrainingStopStatus;
}

/** An imported checkpoint's configuration replaced the current one. */
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

/** One training setting changed, the rest carried over. */
export interface TrainingConfigUpdatedStatePatch {
  trainingConfig: StudioProjectTrainingConfig;
}

/** A weight restore is being verified before it is carried out. */
export interface TrainingWeightRestoreVerificationStartStatePatch {
  error: null;
  trainingWeightRestoreVerification: null;
}

/** The restore plan arrived: where the weights are and how to check them. */
export interface TrainingWeightRestoreVerificationLoadedStatePatch {
  trainingWeightRestoreVerification: TrainingWeightRestoreVerification;
}

/** A restore finished, with what it actually loaded. */
export interface TrainingWeightMaterializationLoadedStatePatch {
  error: null;
  trainingWeightMaterialization: TrainingWeightRestoreResult;
}

/** A new run started with another run's weights attached. */
export interface TrainingWeightAttachLoadedStatePatch {
  error: null;
  trainingWeightAttach: TrainingWeightAttachResult;
}

/** Weights were attached to a run already going. */
export interface TrainingWeightLiveAttachLoadedStatePatch {
  error: null;
  trainingWeightLiveAttach: TrainingWeightLiveAttachResult;
}

/** A training operation failed, with the message to show. */
export interface TrainingFailureStatePatch {
  error: string;
  trainingStatus?: "failed";
}

/** Something was asked for that cannot be done yet, said without ending the run. */
export interface TrainingPreconditionErrorStatePatch {
  error: string;
}

/** An export succeeded and the previous error, if any, is cleared. */
export interface TrainingExportSuccessStatePatch {
  error: null;
}

/**
 * Take the surrogates and cell types into the store.
 *
 * @param trainingSurrogates - What the server offers.
 * @returns The patch.
 */
export function trainingSurrogatesLoadedState(
  trainingSurrogates: SurrogateInfo[],
): TrainingSurrogatesLoadedStatePatch {
  return { trainingSurrogates };
}

/**
 * Clear the previous run and mark a new one as starting.
 *
 * @returns The patch.
 */
export function trainingStartState(): TrainingStartStatePatch {
  return {
    activeTab: "train",
    error: null,
    trainingEpochs: [],
    trainingJobId: null,
    trainingObservedConfig: null,
    trainingStatus: "starting",
    trainingWeightRestorePlan: null,
    trainingWeightRestoreVerification: null,
    trainingWeightMaterialization: null,
    trainingWeightAttach: null,
    trainingWeightLiveAttach: null,
  };
}

/**
 * Record the job the server started.
 *
 * @param jobId - The job's identifier.
 * @returns The patch.
 */
export function trainingStartedState(jobId: string): TrainingStartedStatePatch {
  return {
    trainingJobId: jobId,
    trainingStatus: "running",
  };
}

/**
 * Observe a retained job while preserving the editable project configuration.
 *
 * @param jobId - The retained job ID.
 * @param status - Fresh status from the job's own endpoint.
 * @param observedConfig - Verified configuration, or null for an old row.
 * @returns The selected observation state.
 */
export function trainingRecoveredState(
  jobId: string,
  status: string,
  observedConfig: StudioProjectTrainingConfig | null,
): TrainingRecoveredStatePatch {
  return {
    error: null,
    trainingJobId: jobId,
    trainingStatus: status,
    trainingEpochs: [],
    trainingObservedConfig: observedConfig,
    trainingExperimentKey: null,
    trainingWeightRestorePlan: null,
    trainingWeightRestoreVerification: null,
    trainingWeightMaterialization: null,
    trainingWeightAttach: null,
    trainingWeightLiveAttach: null,
  };
}

/**
 * Record the verified retained list in the store.
 *
 * @param jobs - Jobs decoded from the durable route.
 * @returns The list state patch.
 */
export function trainingJobsLoadedState(jobs: TrainingJobSummary[]): TrainingJobsLoadedStatePatch {
  return { trainingJobs: jobs, trainingJobsLoading: false, trainingJobsError: null };
}

/**
 * Append one epoch's metrics.
 *
 * The epochs so far are passed in rather than read from the store, so an
 * update cannot lose one that arrived while it was being applied.
 *
 * @param currentEpochs - The epochs recorded so far.
 * @param metrics - The epoch that just finished.
 * @returns The patch.
 */
export function trainingEpochAppendedState(
  currentEpochs: TrainingEpochMetrics[],
  metrics: TrainingEpochMetrics,
): TrainingEpochAppendedStatePatch {
  return {
    trainingEpochs: [...currentEpochs, metrics],
  };
}

/**
 * Record that the run reached a terminal status.
 *
 * @param status - The status it ended in.
 * @returns The patch.
 */
export function trainingTerminalState(
  status: StudioTrainingTerminalStatus,
): TrainingTerminalStatePatch {
  return { trainingStatus: status };
}

/**
 * Report a problem the run itself sent.
 *
 * @param message - What it said.
 * @returns The patch.
 */
export function trainingStreamErrorState(message: string): TrainingStreamErrorStatePatch {
  return {
    error: message,
    trainingStatus: "failed",
  };
}

/**
 * Record that the stream went away.
 *
 * Distinct from a stream error: the run may still be going, and saying it
 * failed would be a claim this cannot support.
 *
 * @returns The patch.
 */
export function trainingStreamDisconnectedState(): TrainingStreamDisconnectedStatePatch {
  return { trainingStatus: "disconnected" };
}

/**
 * Record the Stop API's verified pending or terminal outcome.
 *
 * @param status - The server's current outcome.
 * @returns The patch.
 */
export function trainingStopResultState(status: TrainingStopStatus): TrainingStopResultStatePatch {
  return { trainingStatus: status };
}

/**
 * Adopt an imported checkpoint's configuration.
 *
 * @param currentConfig - The configuration as it stands.
 * @param imported - What the import returned.
 * @returns The patch.
 */
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

/**
 * Change one training setting, keeping the rest.
 *
 * @param currentConfig - The configuration as it stands.
 * @param key - The setting to change.
 * @param value - Its new value.
 * @returns The patch.
 */
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

/**
 * Mark a weight restore as being verified.
 *
 * @returns The patch.
 */
export function trainingWeightRestoreVerificationStartState():
TrainingWeightRestoreVerificationStartStatePatch {
  return {
    error: null,
    trainingWeightRestoreVerification: null,
  };
}

/**
 * Take a restore plan into the store.
 *
 * @param trainingWeightRestoreVerification - The plan.
 * @returns The patch.
 */
export function trainingWeightRestoreVerificationLoadedState(
  trainingWeightRestoreVerification: TrainingWeightRestoreVerification,
): TrainingWeightRestoreVerificationLoadedStatePatch {
  return { trainingWeightRestoreVerification };
}

/**
 * Take what a restore actually loaded into the store.
 *
 * @param trainingWeightMaterialization - The materialisation.
 * @returns The patch.
 */
export function trainingWeightMaterializationLoadedState(
  trainingWeightMaterialization: TrainingWeightRestoreResult,
): TrainingWeightMaterializationLoadedStatePatch {
  return { error: null, trainingWeightMaterialization };
}

/**
 * Take an attachment to a new run into the store.
 *
 * @param trainingWeightAttach - The attachment.
 * @returns The patch.
 */
export function trainingWeightAttachLoadedState(
  trainingWeightAttach: TrainingWeightAttachResult,
): TrainingWeightAttachLoadedStatePatch {
  return { error: null, trainingWeightAttach };
}

/**
 * Take an attachment to a running job into the store.
 *
 * @param trainingWeightLiveAttach - The attachment.
 * @returns The patch.
 */
export function trainingWeightLiveAttachLoadedState(
  trainingWeightLiveAttach: TrainingWeightLiveAttachResult,
): TrainingWeightLiveAttachLoadedStatePatch {
  return { error: null, trainingWeightLiveAttach };
}

/**
 * Report a failed training operation.
 *
 * @param error - Whatever was thrown or rejected.
 * @param fallbackMessage - What to show when the error carries no message.
 * @param options - Whether the failure also ends the run.
 * @returns The patch.
 */
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

/**
 * Say that something cannot be done yet, without ending the run.
 *
 * @param message - What is missing.
 * @returns The patch.
 */
export function trainingPreconditionErrorState(
  message: string,
): TrainingPreconditionErrorStatePatch {
  return { error: message };
}

/**
 * Record a successful export and clear the previous error.
 *
 * @returns The patch.
 */
export function trainingExportSuccessState(): TrainingExportSuccessStatePatch {
  return { error: null };
}
