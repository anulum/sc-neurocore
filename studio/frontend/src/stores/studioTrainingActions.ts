// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio training actions

import {
  fetchStudioJobArtifact,
  fetchSurrogates as apiFetchSurrogates,
  fetchTargetProfiles as apiFetchTargetProfiles,
  fetchTrainingJobs as apiFetchTrainingJobs,
  fetchTrainingStatus as apiFetchTrainingStatus,
  startTraining as apiStartTraining,
  stopTraining as apiStopTraining,
  exportTrainingCheckpoint as apiExportTrainingCheckpoint,
  importTrainingCheckpoint as apiImportTrainingCheckpoint,
  restoreTrainingWeights as apiRestoreTrainingWeights,
  attachTrainingWeights as apiAttachTrainingWeights,
  attachTrainingWeightsLive as apiAttachTrainingWeightsLive,
} from "../api/client";
import { parseTrainingCheckpointPayload } from "../trainingCheckpoint";
import { verifyTrainingWeightArtifactBlob } from "../trainingRestore";
import {
  connectStudioTrainingEventSource,
  type StudioTrainingStreamEventSource,
} from "../studioTrainingStream";
import {
  decodeTrainingJobSummaries,
  decodeTrainingRecoveryStatus,
  decodeTrainingStopResult,
  isTrainingTerminalStatus,
  observedTrainingConfig,
} from "../studioTrainingRecovery";
import { readTrainingPreregistrationVerdict } from "../trainingPreregistration";
import { readTrainingTargetProfiles, trainingRequestBody } from "../trainingRequest";
import { readTrainingConversionResult } from "../trainingConversion";
import {
  trainingCheckpointExportPlan,
  trainingWeightRestoreVerificationExportPlan,
} from "../trainingExports";
import {
  trainingCheckpointImportedState,
  trainingJobsLoadedState,
  trainingRecoveredState,
  trainingConfigUpdatedState,
  trainingEpochAppendedState,
  trainingExportSuccessState,
  trainingFailureState,
  trainingPreconditionErrorState,
  trainingStartedState,
  trainingStartState,
  trainingStopResultState,
  trainingStreamDisconnectedState,
  trainingStreamErrorState,
  trainingSurrogatesLoadedState,
  trainingTerminalState,
  trainingWeightRestoreVerificationLoadedState,
  trainingWeightMaterializationLoadedState,
  trainingWeightAttachLoadedState,
  trainingWeightLiveAttachLoadedState,
  trainingWeightRestoreVerificationStartState,
} from "../trainingStoreState";
import { studioTrainingKey } from "../studioExperimentKey";
import type { StudioState, StudioStoreActions } from "./studioTypes";

/** How often, and how many times, a finished run's status is read for its verdict. */
const TRAINING_VERDICT_READ_ATTEMPTS = 40;

const TRAINING_VERDICT_READ_INTERVAL_MS = 250;

/** The training actions: surrogates, target profiles, training jobs, their checkpoints and their weights. */
export type StudioTrainingActions = Pick<StudioStoreActions,
  "loadSurrogates" | "loadTargetProfiles" | "loadTrainingJobs" | "selectTrainingJob" | "startTraining" | "stopTraining" | "exportTrainingCheckpoint" | "importTrainingCheckpointText" | "verifyTrainingWeightRestoreArtifact" | "exportTrainingWeightRestoreVerification" | "materializeTrainingWeights" | "attachTrainingWeights" | "liveAttachTrainingWeights" | "setTrainingConfig">;

/** The training actions, and the hook that forgets the training job a session was following. */
export interface StudioTrainingActionBinding {
  /** The actions the store exposes. */
  readonly actions: StudioTrainingActions;
  /**
   * Stop following the selected training job: close its event stream and outdate every
   * selection and list reply still on its way, so none of them lands after sign-out.
   */
  readonly releaseTrainingSession: () => void;
}

/**
 * Bind the training actions to the Studio store.
 *
 * Training jobs are listed, selected, started and stopped here; a selected job's event
 * stream is followed and its sealed verdict read once the stream reports the end; and
 * checkpoints and trained weights are exported, imported, verified and attached. The
 * stream and the version counters that outdate late replies live in this closure, which
 * is why signing out reaches them through `releaseTrainingSession`.
 *
 * @param set - Writes a patch into the store.
 * @param get - Reads the store as it stands.
 * @returns The actions and the release hook.
 */
export function createStudioTrainingActions(
  set: (partial: Partial<StudioState> | ((state: StudioState) => Partial<StudioState>)) => void,
  get: () => StudioState,
): StudioTrainingActionBinding {
  let trainingStream: StudioTrainingStreamEventSource | null = null;
  let trainingSelectionVersion = 0;
  let trainingListVersion = 0;

  const observeTrainingStream = (jobId: string): void => {
    trainingStream?.close();
    const version = trainingSelectionVersion;
    const stillSelected = (): boolean =>
      version === trainingSelectionVersion && get().trainingJobId === jobId;
    trainingStream = connectStudioTrainingEventSource(jobId, {
      onDisconnected: () => {
        if (stillSelected() && !isTrainingTerminalStatus(get().trainingStatus)) {
          set(trainingStreamDisconnectedState());
        }
      },
      onEpoch: (metrics) => {
        if (stillSelected()) {
          set((prev) => trainingEpochAppendedState(prev.trainingEpochs, metrics));
        }
      },
      onError: (message) => {
        if (!stillSelected()) return;
        if (!isTrainingTerminalStatus(get().trainingStatus)) {
          set(trainingStreamErrorState(message));
        }
        void get().loadTrainingJobs();
      },
      onTerminal: (status) => {
        if (!stillSelected()) return;
        if (!isTrainingTerminalStatus(get().trainingStatus)) {
          set(trainingTerminalState(status));
        }
        void readSealedTrainingJob(jobId, stillSelected, status === "completed");
      },
    });
  };

  // The verdict is part of the finished job's status, not of the event stream.
  // The worker writes its terminal event before the job manager seals the
  // job record, so the status route can still say "running" when the event
  // arrives: it is read again until the record is terminal. The run list is
  // reloaded only then; reloading it on the event listed the finished run as
  // "running" until the reader pressed Refresh.
  // The run list is a second record: it can still say "running" after the
  // status route has turned terminal, so it is read until it agrees.
  const listUntilTerminal = async (jobId: string, stillSelected: () => boolean): Promise<void> => {
    for (let attempt = 0; attempt < TRAINING_VERDICT_READ_ATTEMPTS && stillSelected(); attempt += 1) {
      await get().loadTrainingJobs();
      const listed = get().trainingJobs.find((job) => job.job_id === jobId);
      if (listed === undefined || isTrainingTerminalStatus(listed.status)) return;
      await new Promise((resolve) => { setTimeout(resolve, TRAINING_VERDICT_READ_INTERVAL_MS); });
    }
  };

  const readSealedTrainingJob = async (
    jobId: string,
    stillSelected: () => boolean,
    completed: boolean,
  ): Promise<void> => {
    try {
      for (let attempt = 0; attempt < TRAINING_VERDICT_READ_ATTEMPTS && stillSelected(); attempt += 1) {
        const status = await apiFetchTrainingStatus(jobId);
        if (isTrainingTerminalStatus(status.status)) {
          if (stillSelected() && completed) {
            set({
              trainingPreregistrationVerdict: readTrainingPreregistrationVerdict(status.preregistration_verdict),
              trainingConversionResult: readTrainingConversionResult(status.final_metrics),
            });
          }
          await listUntilTerminal(jobId, stillSelected);
          return;
        }
        await new Promise((resolve) => { setTimeout(resolve, TRAINING_VERDICT_READ_INTERVAL_MS); });
      }
      // Never sealed within the reads: list what the server says now.
      if (stillSelected()) void get().loadTrainingJobs();
    } catch (error) {
      if (stillSelected()) set({ trainingJobsError: String(error) });
    }
  };

  const actions: StudioTrainingActions = {
  loadSurrogates: async () => {
    try {
      const trainingSurrogates = await apiFetchSurrogates();
      set(trainingSurrogatesLoadedState(trainingSurrogates));
    } catch { /* non-critical */ }
  },

  loadTargetProfiles: async () => {
    try {
      set({ trainingTargetProfiles: readTrainingTargetProfiles(await apiFetchTargetProfiles()) });
    } catch { /* non-critical: the form still offers no target */ }
  },

  loadTrainingJobs: async () => {
    const version = ++trainingListVersion;
    set({ trainingJobsLoading: true, trainingJobsError: null });
    try {
      const jobs = decodeTrainingJobSummaries(await apiFetchTrainingJobs());
      if (version !== trainingListVersion) return;
      set(trainingJobsLoadedState(jobs));
      const latest = jobs.at(-1);
      const currentJobId = get().trainingJobId;
      if (get().trainingStatus === "disconnected"
        && currentJobId !== null
        && jobs.some((job) => job.job_id === currentJobId)) {
        await get().selectTrainingJob(currentJobId);
      } else if (currentJobId === null && get().trainingStatus === "idle" && latest) {
        await get().selectTrainingJob(latest.job_id);
      }
    } catch (error) {
      if (version === trainingListVersion) {
        set({ trainingJobsLoading: false, trainingJobsError: String(error) });
      }
    }
  },

  selectTrainingJob: async (jobId) => {
    const selected = get().trainingJobs.find((job) => job.job_id === jobId);
    if (!selected) {
      set({ trainingJobsError: "Training job is not in the retained list" });
      return;
    }
    const version = ++trainingSelectionVersion;
    try {
      const retained = await apiFetchTrainingStatus(jobId);
      const status = decodeTrainingRecoveryStatus(retained, jobId);
      if (version !== trainingSelectionVersion) return;
      const observedConfig = observedTrainingConfig(selected.config);
      const verdict = readTrainingPreregistrationVerdict(retained.preregistration_verdict);
      const conversion = readTrainingConversionResult(retained.final_metrics);
      set({
        ...trainingRecoveredState(jobId, status, observedConfig, verdict, conversion),
        trainingJobsError: null,
      });
      observeTrainingStream(jobId);
    } catch (error) {
      if (version === trainingSelectionVersion) set({ trainingJobsError: String(error) });
    }
  },

  startTraining: async () => {
    const s = get();
    if (s.trainingStatus === "running") return;
    // Recorded at the start rather than at the end: what makes a finished run
    // stale is a change to what was trained, and the reader can change that
    // while the run is going.
    const version = ++trainingSelectionVersion;
    trainingStream?.close();
    trainingStream = null;
    set({ ...trainingStartState(), trainingExperimentKey: studioTrainingKey(s.trainingConfig) });
    try {
      const result = await apiStartTraining(trainingRequestBody(s.trainingConfig));
      if (version !== trainingSelectionVersion) {
        void get().loadTrainingJobs();
        return;
      }
      set({ ...trainingStartedState(result.job_id), trainingObservedConfig: s.trainingConfig });
      observeTrainingStream(result.job_id);
      void get().loadTrainingJobs();
    } catch (e) {
      if (version === trainingSelectionVersion) {
        set(trainingFailureState(e, "Training start failed", { markFailed: true }));
      }
    }
  },

  stopTraining: async () => {
    const s = get();
    if (!s.trainingJobId) return;
    const jobId = s.trainingJobId;
    const version = trainingSelectionVersion;
    try {
      const result = decodeTrainingStopResult(await apiStopTraining(jobId), jobId);
      if (version !== trainingSelectionVersion || get().trainingJobId !== jobId) return;
      if (!isTrainingTerminalStatus(get().trainingStatus)) {
        set(trainingStopResultState(result));
      }
    } catch (e) {
      if (version === trainingSelectionVersion && get().trainingJobId === jobId) {
        set(trainingFailureState(e, "Training stop failed"));
      }
    }
  },

  exportTrainingCheckpoint: async () => {
    const s = get();
    if (!s.trainingJobId) return;
    try {
      const checkpoint = await apiExportTrainingCheckpoint(s.trainingJobId);
      trainingCheckpointExportPlan(checkpoint).writeExport();
    } catch (e) {
      set(trainingFailureState(e, "Training checkpoint export failed"));
    }
  },

  importTrainingCheckpointText: async (checkpointJson) => {
    try {
      const parsed = parseTrainingCheckpointPayload(checkpointJson);
      const imported = await apiImportTrainingCheckpoint(parsed);
      set((s) => trainingCheckpointImportedState(s.trainingConfig, imported));
    } catch (e) {
      set(trainingFailureState(e, "Training checkpoint import failed"));
    }
  },

  verifyTrainingWeightRestoreArtifact: async () => {
    const restorePlan = get().trainingWeightRestorePlan;
    if (restorePlan === null) {
      set(trainingPreconditionErrorState("No training weight restore plan is available."));
      return;
    }
    set(trainingWeightRestoreVerificationStartState());
    try {
      const payload = await fetchStudioJobArtifact(
        restorePlan.source_job_id,
        restorePlan.weights_artifact.relative_path,
      );
      const verification = await verifyTrainingWeightArtifactBlob(restorePlan, payload);
      set(trainingWeightRestoreVerificationLoadedState(verification));
    } catch (error: unknown) {
      set(trainingFailureState(error, "Training weight artifact verification failed"));
    }
  },

  exportTrainingWeightRestoreVerification: () => {
    const { trainingWeightRestorePlan, trainingWeightRestoreVerification } = get();
    const plan = trainingWeightRestoreVerificationExportPlan(
      trainingWeightRestorePlan,
      trainingWeightRestoreVerification,
    );
    if (!plan.available) {
      set(trainingPreconditionErrorState(plan.message));
      return;
    }
    try {
      plan.writeExport();
      set(trainingExportSuccessState());
    } catch (error: unknown) {
      set(trainingFailureState(error, "Training weight restore verification export failed"));
    }
  },

  materializeTrainingWeights: async () => {
    const s = get();
    if (!s.trainingJobId) {
      set(trainingPreconditionErrorState("No completed training job is available."));
      return;
    }
    try {
      const materialization = await apiRestoreTrainingWeights(s.trainingJobId);
      set(trainingWeightMaterializationLoadedState(materialization));
    } catch (error: unknown) {
      set(trainingFailureState(error, "Training weight materialization failed"));
    }
  },

  attachTrainingWeights: async (mode = "warm_start") => {
    const s = get();
    if (!s.trainingJobId) {
      set(trainingPreconditionErrorState("No completed training job is available."));
      return;
    }
    if (["starting", "running", "stopping", "unknown", "disconnected"].includes(s.trainingStatus)) return;
    const version = ++trainingSelectionVersion;
    trainingStream?.close();
    trainingStream = null;
    set({ trainingStatus: "starting", error: null });
    try {
      const attach = await apiAttachTrainingWeights(
        s.trainingJobId, trainingRequestBody(s.trainingConfig), undefined, mode,
      );
      if (version !== trainingSelectionVersion) {
        void get().loadTrainingJobs();
        return;
      }
      if (attach.source_job_id !== s.trainingJobId || typeof attach.job_id !== "string"
        || attach.job_id.length === 0 || attach.status !== "running"
        || ((mode === "exact_resume" || attach.mode !== undefined) && attach.mode !== mode)) {
        throw new Error("Training attachment response does not identify the requested operation");
      }
      set({
        ...trainingStartState(), ...trainingStartedState(attach.job_id),
        ...trainingWeightAttachLoadedState(attach),
        trainingExperimentKey: studioTrainingKey(s.trainingConfig),
        trainingObservedConfig: s.trainingConfig,
      });
      observeTrainingStream(attach.job_id);
      void get().loadTrainingJobs();
    } catch (error: unknown) {
      if (version === trainingSelectionVersion) {
        set({ ...trainingFailureState(error, "Training weight attach failed"), trainingStatus: s.trainingStatus });
      }
    }
  },

  liveAttachTrainingWeights: async () => {
    const s = get();
    const sourceJobId = s.trainingWeightMaterialization?.source_job_id;
    if (!s.trainingJobId || !sourceJobId) {
      set(
        trainingPreconditionErrorState(
          "A running target job and a verified source are required.",
        ),
      );
      return;
    }
    try {
      const attach = await apiAttachTrainingWeightsLive(s.trainingJobId, sourceJobId);
      set(trainingWeightLiveAttachLoadedState(attach));
    } catch (error: unknown) {
      set(trainingFailureState(error, "Training weight live attach failed"));
    }
  },

  setTrainingConfig: (key, value) => {
    set((s) => trainingConfigUpdatedState(s.trainingConfig, key, value));
  },
  };

  return {
    actions,
    releaseTrainingSession: () => {
      trainingSelectionVersion += 1;
      trainingListVersion += 1;
      trainingStream?.close();
      trainingStream = null;
    },
  };
}
