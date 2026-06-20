import type {
  TrainingCheckpointPayload,
  TrainingConfig,
  TrainingWeightArtifact,
  TrainingWeightCheckpoint,
} from "./api/client";

const CHECKPOINT_SCHEMA_VERSION = "studio.training.checkpoint.v1";
const WEIGHT_CHECKPOINT_SCHEMA_VERSION = "studio.training.weight-checkpoint.v1";
const TRAINING_WEIGHT_ARTIFACT_PATH = "training/model_state.pt";
const TRAINING_WEIGHT_METADATA_ARTIFACT_PATH = "training/model_state.json";
const SHA256_PATTERN = /^[0-9a-f]{64}$/;

export function parseTrainingCheckpointPayload(text: string): TrainingCheckpointPayload {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text) as unknown;
  } catch (error: unknown) {
    throw new Error(
      error instanceof Error
        ? `Training checkpoint JSON is invalid: ${error.message}`
        : "Training checkpoint JSON is invalid.",
    );
  }
  return validateTrainingCheckpointPayload(parsed);
}

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
  return config as Partial<TrainingConfig>;
}

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

function requireObject(value: unknown, message: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(message);
  }
  return value as Record<string, unknown>;
}

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

function requireString(value: unknown, fieldName: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`Training checkpoint requires ${fieldName}.`);
  }
  return value;
}

function requireSha256(value: unknown, fieldName: string): string {
  const digest = requireString(value, fieldName);
  if (!SHA256_PATTERN.test(digest)) {
    throw new Error(`Training checkpoint ${fieldName} must be a lowercase SHA-256 digest.`);
  }
  return digest;
}

function requireLiteral<T extends string>(value: unknown, expected: T, fieldName: string): T {
  if (value !== expected) {
    throw new Error(`Training checkpoint ${fieldName} must be ${expected}.`);
  }
  return expected;
}

function validateOptionalString(value: unknown, fieldName: string): void {
  if (value !== undefined && typeof value !== "string") {
    throw new Error(`Training checkpoint ${fieldName} must be a string.`);
  }
}

function validateOptionalBoolean(value: unknown, fieldName: string): void {
  if (value !== undefined && typeof value !== "boolean") {
    throw new Error(`Training checkpoint ${fieldName} must be a boolean.`);
  }
}

function validateOptionalNumber(value: unknown, fieldName: string): void {
  if (value !== undefined && !isFiniteNumber(value)) {
    throw new Error(`Training checkpoint ${fieldName} must be a finite number.`);
  }
}

function requirePositiveInteger(value: unknown, fieldName: string): number {
  if (!Number.isInteger(value) || typeof value !== "number" || value <= 0) {
    throw new Error(`Training checkpoint ${fieldName} must be a positive integer.`);
  }
  return value;
}

function requireNonNegativeInteger(value: unknown, fieldName: string): number {
  if (!Number.isInteger(value) || typeof value !== "number" || value < 0) {
    throw new Error(`Training checkpoint ${fieldName} must be a non-negative integer.`);
  }
  return value;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}
