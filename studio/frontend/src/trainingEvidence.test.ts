import { describe, expect, it } from "vitest";

import type { TrainingConfig, TrainingEpochMetrics } from "./api/client";
import { buildTrainingEvidenceModel } from "./trainingEvidence";

const config: Pick<TrainingConfig, "dataset" | "epochs" | "surrogate" | "timesteps"> = {
  dataset: "synthetic",
  epochs: 12,
  surrogate: "atan_surrogate",
  timesteps: 25,
};

const latestEpoch: TrainingEpochMetrics = {
  epoch: 3,
  layer_spike_rates: { hidden: 0.12 },
  param_snapshot: { beta: 0.9 },
  train_accuracy: 0.8,
  train_loss: 0.2,
  val_accuracy: 0.75,
  val_loss: 0.3,
};

describe("training evidence model", () => {
  it("describes the pending training action-evidence contract before submission", () => {
    expect(buildTrainingEvidenceModel(null, "idle", config, null)).toEqual({
      actionKind: "studio.training.run",
      classification: "training",
      configSummary: "synthetic, 12 epochs, atan_surrogate, 25 steps",
      evidenceArtifact: "pending",
      jobId: "not submitted",
      latestEpoch: "none",
      replayRoute: "POST /api/training/start",
      status: "idle",
      statusArtifact: "pending",
    });
  });

  it("describes submitted training job evidence artifacts and latest epoch", () => {
    expect(buildTrainingEvidenceModel("sj_training", "completed", config, latestEpoch)).toEqual({
      actionKind: "studio.training.run",
      classification: "training",
      configSummary: "synthetic, 12 epochs, atan_surrogate, 25 steps",
      evidenceArtifact: "training/evidence.json",
      jobId: "sj_training",
      latestEpoch: "3",
      replayRoute: "POST /api/training/start",
      status: "completed",
      statusArtifact: "training/status.json",
    });
  });
});
