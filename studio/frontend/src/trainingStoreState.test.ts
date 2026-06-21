// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio training store state helper tests
import { describe, expect, it } from "vitest";

import type {
  TrainingCheckpointImportResponse,
  TrainingEpochMetrics,
  TrainingWeightRestorePlan,
} from "./api/client";
import type { StudioProjectTrainingConfig } from "./studioProjectState";
import {
  trainingConfigUpdatedState,
  trainingCheckpointImportedState,
  trainingEpochAppendedState,
  trainingExportSuccessState,
  trainingFailureState,
  trainingStartedState,
  trainingStartState,
  trainingStoppingState,
  trainingStreamDisconnectedState,
  trainingStreamErrorState,
  trainingSurrogatesLoadedState,
  trainingTerminalState,
  trainingWeightRestoreVerificationLoadedState,
  trainingWeightRestoreVerificationStartState,
} from "./trainingStoreState";
import type { TrainingWeightRestoreVerification } from "./trainingRestore";

const trainingConfig: StudioProjectTrainingConfig = {
  batch_size: 64,
  dataset: "synthetic",
  epochs: 10,
  hidden: [128],
  learn_beta: false,
  learn_threshold: false,
  lr: 0.001,
  surrogate: "atan_surrogate",
  timesteps: 25,
};

const metrics: TrainingEpochMetrics = {
  epoch: 1,
  layer_spike_rates: { hidden: 0.2 },
  param_snapshot: { beta: 0.9 },
  train_accuracy: 0.8,
  train_loss: 0.4,
  val_accuracy: 0.75,
  val_loss: 0.5,
};

function restorePlan(): TrainingWeightRestorePlan {
  return {
    architecture: "snn",
    artifact_route_template: "/api/studio/jobs/{job_id}/artifacts/{artifact_path}",
    config_sha256: "c".repeat(64),
    format: "torch-state-dict",
    framework: "torch",
    loader_policy: "download_from_authenticated_artifact_route_and_verify_sha256",
    metadata_artifact: {
      relative_path: "training/model_state.json",
      sha256: "a".repeat(64),
      size_bytes: 64,
    },
    parameter_count: 128,
    restore_ready: true,
    schema_version: "studio.training.weight-restore-plan.v1",
    source_job_id: "sj_training",
    source_status: "completed",
    weights_artifact: {
      relative_path: "training/model_state.pt",
      sha256: "b".repeat(64),
      size_bytes: 256,
    },
  };
}

function importedCheckpoint(): TrainingCheckpointImportResponse {
  return {
    config: { epochs: 5, lr: 0.002 },
    config_sha256: "c".repeat(64),
    imported_schema_version: "studio.training.checkpoint.v1",
    source_job_id: "sj_training",
    source_status: "completed",
    source_weight_checkpoint: null,
    weight_restore_plan: restorePlan(),
  };
}

describe("training store state helpers", () => {
  it("builds surrogate and training lifecycle patches", () => {
    expect(trainingSurrogatesLoadedState([{ available: true, name: "atan" }])).toEqual({
      trainingSurrogates: [{ available: true, name: "atan" }],
    });
    expect(trainingStartState()).toEqual({
      activeTab: "train",
      error: null,
      trainingEpochs: [],
      trainingStatus: "starting",
      trainingWeightRestorePlan: null,
      trainingWeightRestoreVerification: null,
    });
    expect(trainingStartedState("sj_training")).toEqual({
      trainingJobId: "sj_training",
      trainingStatus: "running",
    });
    expect(trainingStoppingState()).toEqual({ trainingStatus: "stopping" });
  });

  it("builds stream update patches", () => {
    expect(trainingEpochAppendedState([], metrics)).toEqual({ trainingEpochs: [metrics] });
    expect(trainingTerminalState("completed")).toEqual({ trainingStatus: "completed" });
    expect(trainingStreamErrorState("diverged")).toEqual({
      error: "diverged",
      trainingStatus: "failed",
    });
    expect(trainingStreamDisconnectedState()).toEqual({ trainingStatus: "disconnected" });
  });

  it("builds checkpoint import and restore-verification patches", () => {
    const imported = importedCheckpoint();
    const verification: TrainingWeightRestoreVerification = {
      actual_sha256: "b".repeat(64),
      expected_sha256: "b".repeat(64),
      relative_path: "training/model_state.pt",
      size_bytes: 256,
      source_job_id: "sj_training",
      status: "verified",
      verified_at_utc: "2026-06-21T09:30:00Z",
    };

    expect(trainingCheckpointImportedState(trainingConfig, imported)).toEqual({
      activeTab: "train",
      error: null,
      trainingConfig: { ...trainingConfig, epochs: 5, lr: 0.002 },
      trainingEpochs: [],
      trainingJobId: "sj_training",
      trainingStatus: "checkpoint:completed",
      trainingWeightRestorePlan: imported.weight_restore_plan,
      trainingWeightRestoreVerification: null,
    });
    expect(trainingWeightRestoreVerificationStartState()).toEqual({
      error: null,
      trainingWeightRestoreVerification: null,
    });
    expect(trainingWeightRestoreVerificationLoadedState(verification)).toEqual({
      trainingWeightRestoreVerification: verification,
    });
  });

  it("updates training config through typed field patches", () => {
    expect(trainingConfigUpdatedState(trainingConfig, "epochs", 20)).toEqual({
      trainingConfig: { ...trainingConfig, epochs: 20 },
    });
    expect(trainingConfigUpdatedState(trainingConfig, "learn_beta", true)).toEqual({
      trainingConfig: { ...trainingConfig, learn_beta: true },
    });
    expect(trainingConfigUpdatedState(trainingConfig, "hidden", [256, 128])).toEqual({
      trainingConfig: { ...trainingConfig, hidden: [256, 128] },
    });
  });

  it("builds failure and export-success patches", () => {
    expect(trainingFailureState(new Error("training offline"), "fallback")).toEqual({
      error: "training offline",
    });
    expect(trainingFailureState("bad", "fallback", { markFailed: true })).toEqual({
      error: "fallback",
      trainingStatus: "failed",
    });
    expect(trainingExportSuccessState()).toEqual({ error: null });
  });
});
