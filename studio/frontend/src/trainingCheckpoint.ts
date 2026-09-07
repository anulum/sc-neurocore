// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * Reading a training checkpoint someone else's installation wrote.
 *
 * A checkpoint is an untrusted document: it arrives from a file the reader
 * chose, and every field it claims is checked before any of it reaches the
 * store. The validators are separate and named after the shape they enforce,
 * so a refusal says which field was wrong rather than that the document was.
 *
 * The digests matter most. A checkpoint whose weights do not match the
 * configuration they were trained under will load into a network they do not
 * fit, and the failure appears later as bad numbers rather than as a refused
 * import — which is why the format is validated here and the digests are
 * checked again when the weights are actually restored.
 */

import type {
  TrainingCheckpointPayload,
  TrainingConfig,
  TrainingWeightArtifact,
  TrainingWeightCheckpoint,
} from "./api/client";

/** The only checkpoint schema this build reads. */
const CHECKPOINT_SCHEMA_VERSION = "studio.training.checkpoint.v1";
/** The only weight-checkpoint schema this build reads. */
const WEIGHT_CHECKPOINT_SCHEMA_VERSION = "studio.training.weight-checkpoint.v1";
/** Where a run's weights are stored within its job. */
const TRAINING_WEIGHT_ARTIFACT_PATH = "training/model_state.pt";
/** Where the weights' metadata is stored within its job. */
const TRAINING_WEIGHT_METADATA_ARTIFACT_PATH = "training/model_state.json";
/** A lowercase hexadecimal SHA-256 digest, and nothing else. */
const SHA256_PATTERN = /^[0-9a-f]{64}$/;

/**
 * Read a checkpoint from text, refusing anything that is not one.
 *
 * @param text - The document's text.
 * @returns The validated checkpoint.
 * @throws {Error} When the text is not JSON, or is JSON that is not a
 *   checkpoint. The message names what was wrong.
 */
export function parseTrainingCheckpointPayload(text: string): TrainingCheckpointPayload {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text) as unknown;
  } catch (error: unknown) {
    // The original is attached, not merely quoted: its message says what was
    // wrong and its stack says where, and a caller that logs only the symptom
    // would otherwise lose the second half.
    throw new Error(
      error instanceof Error
        ? `Training checkpoint JSON is invalid: ${error.message}`
        : "Training checkpoint JSON is invalid.",
      { cause: error },
    );
  }
  return validateTrainingCheckpointPayload(parsed);
}

/**
 * Check an already-parsed value is a checkpoint.
 *
 * @param value - The parsed document.
 * @returns The validated checkpoint.
 * @throws {Error} When any field is missing or the wrong shape.
 */
export function validateTrainingCheckpointPayload(
  value: unknown,
): TrainingCheckpointPayload {
  const checkpoint = requireObject(value, "Training checkpoint must be a JSON object.");
  const payload: TrainingCheckpointPayload = {
    checkpoint_sha256: requireSha256(checkpoint.checkpoint_sha256, "checkpoint_sha256"),
    config: validateTrainingConfig(checkpoint.config),
    config_sha256: requireSha256(checkpoint.config_sha256, "config_sha256"),
    evidence_summary: validateOptionalObject(checkpoint.evidence_summary, "evidence_summary"),
    final_metrics: validateOptionalNumberRecord(checkpoint.final_metrics, "final_metrics"),
    generated_at_utc: requireString(checkpoint.generated_at_utc, "generated_at_utc"),
    job_id: requireString(checkpoint.job_id, "job_id"),
    schema_version: requireLiteral(
      checkpoint.schema_version,
      CHECKPOINT_SCHEMA_VERSION,
      "schema_version",
    ),
    status: requireString(checkpoint.status, "status"),
  };
  const weightCheckpoint = checkpoint.weight_checkpoint;
  if (weightCheckpoint !== undefined) {
    payload.weight_checkpoint = weightCheckpoint === null
      ? null
      : validateTrainingWeightCheckpoint(weightCheckpoint);
  }
  return payload;
}

/**
 * Check the training configuration a checkpoint carries.
 *
 * @param value - The configuration.
 * @returns It, validated.
 * @throws {Error} When a field is present and the wrong shape.
 */
function validateTrainingConfig(value: unknown): Partial<TrainingConfig> {
  const config = requireObject(value, "Training checkpoint config must be a JSON object.");
  validateOptionalString(config.dataset, "config.dataset");
  validateOptionalString(config.surrogate, "config.surrogate");
  validateOptionalNumber(config.epochs, "config.epochs");
  validateOptionalNumber(config.batch_size, "config.batch_size");
  validateOptionalNumber(config.lr, "config.lr");
  validateOptionalNumber(config.timesteps, "config.timesteps");
  validateOptionalBoolean(config.learn_beta, "config.learn_beta");
  validateOptionalBoolean(config.learn_threshold, "config.learn_threshold");
  if (config.hidden !== undefined) {
    if (!Array.isArray(config.hidden) || !config.hidden.every((item) => isFiniteNumber(item))) {
      throw new Error("Training checkpoint config.hidden must be a numeric array.");
    }
  }
  if (config.max_grad_norm !== undefined) {
    validateOptionalNumber(config.max_grad_norm, "config.max_grad_norm");
  }
  return config;
}

/**
 * Check the weight checkpoint a document carries.
 *
 * @param value - The weight checkpoint.
 * @returns It, validated.
 * @throws {Error} When a field is missing or the wrong shape.
 */
function validateTrainingWeightCheckpoint(value: unknown): TrainingWeightCheckpoint {
  const checkpoint = requireObject(value, "Training weight checkpoint must be a JSON object.");
  const payload: TrainingWeightCheckpoint = {
    schema_version: requireLiteral(
      checkpoint.schema_version,
      WEIGHT_CHECKPOINT_SCHEMA_VERSION,
      "weight_checkpoint.schema_version",
    ),
    weights_artifact: validateArtifact(
      checkpoint.weights_artifact,
      TRAINING_WEIGHT_ARTIFACT_PATH,
      "weight_checkpoint.weights_artifact",
    ),
  };
  if (checkpoint.metadata_artifact !== undefined) {
    payload.metadata_artifact = validateArtifact(
      checkpoint.metadata_artifact,
      TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
      "weight_checkpoint.metadata_artifact",
    );
  }
  if (checkpoint.architecture !== undefined) {
    payload.architecture = requireString(checkpoint.architecture, "weight_checkpoint.architecture");
  }
  if (checkpoint.config_sha256 !== undefined) {
    payload.config_sha256 = requireSha256(
      checkpoint.config_sha256,
      "weight_checkpoint.config_sha256",
    );
  }
  if (checkpoint.final_metrics !== undefined) {
    payload.final_metrics = validateOptionalObject(
      checkpoint.final_metrics,
      "weight_checkpoint.final_metrics",
    );
  }
  if (checkpoint.format !== undefined) {
    payload.format = requireString(checkpoint.format, "weight_checkpoint.format");
  }
  if (checkpoint.framework !== undefined) {
    payload.framework = requireString(checkpoint.framework, "weight_checkpoint.framework");
  }
  if (checkpoint.parameter_count !== undefined) {
    payload.parameter_count = requireNonNegativeInteger(
      checkpoint.parameter_count,
      "weight_checkpoint.parameter_count",
    );
  }
  return payload;
}

/**
 * Check one artefact reference, including that it is the expected path.
 *
 * The path is pinned because a checkpoint naming a different file would
 * have the reader fetch something the digest does not describe.
 *
 * @param value - The artefact reference.
 * @param expectedPath - The path it must name.
 * @param fieldName - The field being checked, for the message.
 * @returns It, validated.
 * @throws {Error} When it is not that artefact.
 */
function validateArtifact(
  value: unknown,
  expectedPath: string,
  fieldName: string,
): TrainingWeightArtifact {
  const artifact = requireObject(value, `${fieldName} must be a JSON object.`);
  const relativePath = requireString(artifact.relative_path, `${fieldName}.relative_path`);
  if (relativePath !== expectedPath) {
    throw new Error(`${fieldName}.relative_path must be ${expectedPath}.`);
  }
  return {
    relative_path: relativePath,
    sha256: requireSha256(artifact.sha256, `${fieldName}.sha256`),
    size_bytes: requirePositiveInteger(artifact.size_bytes, `${fieldName}.size_bytes`),
  };
}

/**
 * Require a plain object.
 *
 * @param value - The value.
 * @param message - What to say when it is not one.
 * @returns The object.
 * @throws {Error} When it is not one.
 */
function requireObject(value: unknown, message: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(message);
  }
  return value as Record<string, unknown>;
}

/**
 * Check an optional object field.
 *
 * @param value - The value.
 * @param fieldName - The field, for the message.
 * @returns The object, or `null`.
 * @throws {Error} When present and not an object.
 */
function validateOptionalObject(
  value: unknown,
  fieldName: string,
): Record<string, unknown> | null {
  if (value === null) return null;
  if (value === undefined) {
    throw new Error(`Training checkpoint requires ${fieldName}.`);
  }
  return requireObject(value, `Training checkpoint ${fieldName} must be an object or null.`);
}

/**
 * Check an optional record of numbers.
 *
 * @param value - The value.
 * @param fieldName - The field, for the message.
 * @returns The record, or `null`.
 * @throws {Error} When present and not one.
 */
function validateOptionalNumberRecord(
  value: unknown,
  fieldName: string,
): Record<string, number> | null {
  if (value === null) return null;
  const record = requireObject(value, `Training checkpoint ${fieldName} must be an object or null.`);
  for (const [key, item] of Object.entries(record)) {
    if (!isFiniteNumber(item)) {
      throw new Error(`Training checkpoint ${fieldName}.${key} must be a finite number.`);
    }
  }
  return record as Record<string, number>;
}

/**
 * Require a non-empty string.
 *
 * @param value - The value.
 * @param fieldName - The field, for the message.
 * @returns The string.
 * @throws {Error} When it is not one.
 */
function requireString(value: unknown, fieldName: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`Training checkpoint requires ${fieldName}.`);
  }
  return value;
}

/**
 * Require a lowercase hexadecimal SHA-256 digest.
 *
 * The shape is checked, not the content: this cannot say a digest is the
 * right one, only that it is a digest at all.
 *
 * @param value - The value.
 * @param fieldName - The field, for the message.
 * @returns The digest.
 * @throws {Error} When it is not one.
 */
function requireSha256(value: unknown, fieldName: string): string {
  const digest = requireString(value, fieldName);
  if (!SHA256_PATTERN.test(digest)) {
    throw new Error(`Training checkpoint ${fieldName} must be a lowercase SHA-256 digest.`);
  }
  return digest;
}

/**
 * Require an exact value, such as a schema version.
 *
 * @param value - The value.
 * @param expected - What it must equal.
 * @param fieldName - The field, for the message.
 * @returns The value.
 * @throws {Error} When it differs.
 */
function requireLiteral<T extends string>(value: unknown, expected: T, fieldName: string): T {
  if (value !== expected) {
    throw new Error(`Training checkpoint ${fieldName} must be ${expected}.`);
  }
  return expected;
}

/**
 * Check an optional string field.
 *
 * @param value - The value.
 * @param fieldName - The field, for the message.
 * @throws {Error} When present and not a string.
 */
function validateOptionalString(value: unknown, fieldName: string): void {
  if (value !== undefined && typeof value !== "string") {
    throw new Error(`Training checkpoint ${fieldName} must be a string.`);
  }
}

/**
 * Check an optional boolean field.
 *
 * @param value - The value.
 * @param fieldName - The field, for the message.
 * @throws {Error} When present and not a boolean.
 */
function validateOptionalBoolean(value: unknown, fieldName: string): void {
  if (value !== undefined && typeof value !== "boolean") {
    throw new Error(`Training checkpoint ${fieldName} must be a boolean.`);
  }
}

/**
 * Check an optional finite number field.
 *
 * @param value - The value.
 * @param fieldName - The field, for the message.
 * @throws {Error} When present and not a finite number.
 */
function validateOptionalNumber(value: unknown, fieldName: string): void {
  if (value !== undefined && !isFiniteNumber(value)) {
    throw new Error(`Training checkpoint ${fieldName} must be a finite number.`);
  }
}

/**
 * Require a whole number above zero.
 *
 * @param value - The value.
 * @param fieldName - The field, for the message.
 * @returns The number.
 * @throws {Error} When it is not one.
 */
function requirePositiveInteger(value: unknown, fieldName: string): number {
  if (!Number.isInteger(value) || typeof value !== "number" || value <= 0) {
    throw new Error(`Training checkpoint ${fieldName} must be a positive integer.`);
  }
  return value;
}

/**
 * Require a whole number of zero or more.
 *
 * @param value - The value.
 * @param fieldName - The field, for the message.
 * @returns The number.
 * @throws {Error} When it is not one.
 */
function requireNonNegativeInteger(value: unknown, fieldName: string): number {
  if (!Number.isInteger(value) || typeof value !== "number" || value < 0) {
    throw new Error(`Training checkpoint ${fieldName} must be a non-negative integer.`);
  }
  return value;
}

/**
 * Whether a value is a finite number.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}
