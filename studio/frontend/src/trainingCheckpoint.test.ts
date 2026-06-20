import { describe, expect, it } from "vitest";

import {
  parseTrainingCheckpointPayload,
  validateTrainingCheckpointPayload,
} from "./trainingCheckpoint";

const digest = "a".repeat(64);

function checkpointPayload(): Record<string, unknown> {
  return {
    checkpoint_sha256: digest,
    config: {
      batch_size: 32,
      dataset: "synthetic",
      epochs: 4,
      hidden: [128],
      learn_beta: false,
      learn_threshold: true,
      lr: 0.001,
      max_grad_norm: 1,
      surrogate: "atan_surrogate",
      timesteps: 25,
    },
    config_sha256: "b".repeat(64),
    evidence_summary: { action_kind: "studio.training.run" },
    final_metrics: { train_accuracy: 0.75 },
    generated_at_utc: "2026-06-20T12:00:00Z",
    job_id: "sj_training",
    schema_version: "studio.training.checkpoint.v1",
    status: "completed",
    weight_checkpoint: {
      architecture: "64->128->10",
      config_sha256: "b".repeat(64),
      final_metrics: { train_accuracy: 0.75 },
      format: "torch_state_dict",
      framework: "pytorch",
      metadata_artifact: {
        relative_path: "training/model_state.json",
        sha256: "c".repeat(64),
        size_bytes: 512,
      },
      parameter_count: 9610,
      schema_version: "studio.training.weight-checkpoint.v1",
      weights_artifact: {
        relative_path: "training/model_state.pt",
        sha256: "d".repeat(64),
        size_bytes: 4096,
      },
    },
  };
}

describe("training checkpoint import validation", () => {
  it("parses a portable checkpoint payload with weight metadata", () => {
    const parsed = parseTrainingCheckpointPayload(JSON.stringify(checkpointPayload()));

    expect(parsed.schema_version).toBe("studio.training.checkpoint.v1");
    expect(parsed.job_id).toBe("sj_training");
    expect(parsed.config.dataset).toBe("synthetic");
    expect(parsed.weight_checkpoint?.weights_artifact).toEqual({
      relative_path: "training/model_state.pt",
      sha256: "d".repeat(64),
      size_bytes: 4096,
    });
  });

  it("rejects invalid checkpoint JSON before API import", () => {
    expect(() => parseTrainingCheckpointPayload("{not json")).toThrow(
      "Training checkpoint JSON is invalid",
    );
  });

  it("rejects unsupported checkpoint schema versions", () => {
    const payload = checkpointPayload();
    payload.schema_version = "studio.training.checkpoint.v0";

    expect(() => validateTrainingCheckpointPayload(payload)).toThrow(
      "schema_version must be studio.training.checkpoint.v1",
    );
  });

  it("rejects non-finite training config values", () => {
    const payload = checkpointPayload();
    payload.config = { dataset: "synthetic", epochs: Number.NaN };

    expect(() => validateTrainingCheckpointPayload(payload)).toThrow(
      "config.epochs must be a finite number",
    );
  });

  it("rejects forged weight artifact paths", () => {
    const payload = checkpointPayload();
    const weightCheckpoint = payload.weight_checkpoint as Record<string, unknown>;
    weightCheckpoint.weights_artifact = {
      relative_path: "../model_state.pt",
      sha256: "d".repeat(64),
      size_bytes: 4096,
    };

    expect(() => validateTrainingCheckpointPayload(payload)).toThrow(
      "weight_checkpoint.weights_artifact.relative_path must be training/model_state.pt",
    );
  });

  it("rejects uppercase checkpoint digests", () => {
    const payload = checkpointPayload();
    payload.checkpoint_sha256 = "A".repeat(64);

    expect(() => validateTrainingCheckpointPayload(payload)).toThrow(
      "checkpoint_sha256 must be a lowercase SHA-256 digest",
    );
  });
});
