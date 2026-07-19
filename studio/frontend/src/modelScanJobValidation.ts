// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fail-closed runtime guards for model-scan job payloads

/**
 * Nested type guards for model-scan job receipts, poll records, and completed
 * ``studio.model-scan.v1`` results. No unchecked casts of untrusted JSON.
 */

import type {
  ModelBehavior,
  ModelScanFailure,
  ModelScanJobReceipt,
  ModelScanMetadata,
  ModelScanResponse,
  StudioJobRecord,
} from "./api/client";

export type ValidationResult<T> =
  | { ok: true; value: T }
  | { ok: false; error: string };

const HEX64 = /^[0-9a-fA-F]{64}$/;

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function isSha256Hex(value: unknown): value is string {
  return typeof value === "string" && HEX64.test(value);
}

/**
 * Guard one model behaviour entry from a scan result.
 */
export function parseModelBehavior(value: unknown): ValidationResult<ModelBehavior> {
  if (!isRecord(value)) {
    return { ok: false, error: "model_scan_model_not_object" };
  }
  if (!isNonEmptyString(value.name)) {
    return { ok: false, error: "model_scan_model_name_invalid" };
  }
  if (!isNonEmptyString(value.category)) {
    return { ok: false, error: "model_scan_model_category_invalid" };
  }
  if (!isNonEmptyString(value.pattern)) {
    return { ok: false, error: "model_scan_model_pattern_invalid" };
  }
  if (typeof value.description !== "string") {
    return { ok: false, error: "model_scan_model_description_invalid" };
  }
  if (!isFiniteNumber(value.rate_hz)) {
    return { ok: false, error: "model_scan_model_rate_hz_invalid" };
  }
  if (!isFiniteNumber(value.spike_count)) {
    return { ok: false, error: "model_scan_model_spike_count_invalid" };
  }
  if (value.error_type !== undefined && typeof value.error_type !== "string") {
    return { ok: false, error: "model_scan_model_error_type_invalid" };
  }
  const model: ModelBehavior = {
    name: value.name,
    category: value.category,
    pattern: value.pattern,
    description: value.description,
    rate_hz: value.rate_hz,
    spike_count: value.spike_count,
  };
  if (typeof value.error_type === "string") {
    model.error_type = value.error_type;
  }
  return { ok: true, value: model };
}

/**
 * Guard one scan failure entry.
 */
export function parseModelScanFailure(value: unknown): ValidationResult<ModelScanFailure> {
  if (!isRecord(value)) {
    return { ok: false, error: "model_scan_failure_not_object" };
  }
  if (!isNonEmptyString(value.name)) {
    return { ok: false, error: "model_scan_failure_name_invalid" };
  }
  if (!isNonEmptyString(value.category)) {
    return { ok: false, error: "model_scan_failure_category_invalid" };
  }
  if (!isNonEmptyString(value.error_type)) {
    return { ok: false, error: "model_scan_failure_error_type_invalid" };
  }
  if (typeof value.error_message !== "string") {
    return { ok: false, error: "model_scan_failure_error_message_invalid" };
  }
  return {
    ok: true,
    value: {
      name: value.name,
      category: value.category,
      error_type: value.error_type,
      error_message: value.error_message,
    },
  };
}

/**
 * Guard ``pattern_counts`` as a string-key to finite-number record.
 */
export function parsePatternCounts(
  value: unknown,
): ValidationResult<Record<string, number>> {
  if (!isRecord(value)) {
    return { ok: false, error: "model_scan_pattern_counts_invalid" };
  }
  const counts: Record<string, number> = {};
  for (const [key, count] of Object.entries(value)) {
    if (!isNonEmptyString(key) || !isFiniteNumber(count)) {
      return { ok: false, error: "model_scan_pattern_counts_entry_invalid" };
    }
    counts[key] = count;
  }
  return { ok: true, value: counts };
}

/**
 * Guard scan metadata including digests, counts, and nested failures.
 */
export function parseModelScanMetadata(value: unknown): ValidationResult<ModelScanMetadata> {
  if (!isRecord(value)) {
    return { ok: false, error: "model_scan_metadata_missing" };
  }
  if (value.schema_version !== "studio.model-scan.v1") {
    return { ok: false, error: "model_scan_metadata_schema_mismatch" };
  }
  if (value.evidence_classification !== "analysis") {
    return { ok: false, error: "model_scan_evidence_class_invalid" };
  }
  if (value.status !== "completed") {
    return { ok: false, error: "model_scan_metadata_status_invalid" };
  }
  if (!isFiniteNumber(value.current)) {
    return { ok: false, error: "model_scan_metadata_current_invalid" };
  }
  if (!isFiniteNumber(value.duration)) {
    return { ok: false, error: "model_scan_metadata_duration_invalid" };
  }
  if (!isFiniteNumber(value.error_count)) {
    return { ok: false, error: "model_scan_metadata_error_count_invalid" };
  }
  if (!isFiniteNumber(value.model_count)) {
    return { ok: false, error: "model_scan_metadata_model_count_invalid" };
  }
  if (!isSha256Hex(value.input_sha256)) {
    return { ok: false, error: "model_scan_metadata_input_sha256_invalid" };
  }
  if (!isSha256Hex(value.result_sha256)) {
    return { ok: false, error: "model_scan_metadata_result_sha256_invalid" };
  }
  if (!Array.isArray(value.failed_models)) {
    return { ok: false, error: "model_scan_failed_models_invalid" };
  }
  const failedModels: ModelScanFailure[] = [];
  for (const entry of value.failed_models) {
    const parsed = parseModelScanFailure(entry);
    if (!parsed.ok) {
      return parsed;
    }
    failedModels.push(parsed.value);
  }
  const patterns = parsePatternCounts(value.pattern_counts);
  if (!patterns.ok) {
    return patterns;
  }
  return {
    ok: true,
    value: {
      current: value.current,
      duration: value.duration,
      error_count: value.error_count,
      evidence_classification: "analysis",
      failed_models: failedModels,
      input_sha256: value.input_sha256,
      model_count: value.model_count,
      pattern_counts: patterns.value,
      result_sha256: value.result_sha256,
      schema_version: "studio.model-scan.v1",
      status: "completed",
    },
  };
}

/**
 * Validate a completed job ``result`` as full ``studio.model-scan.v1`` evidence.
 * Fails closed on any invalid nested entry (does not drop invalids silently).
 */
export function validateModelScanJobResult(
  result: unknown,
): ValidationResult<ModelScanResponse> {
  if (!isRecord(result)) {
    return { ok: false, error: "model_scan_result_not_object" };
  }
  if (result.schema_version !== "studio.model-scan.v1") {
    return { ok: false, error: "model_scan_schema_mismatch" };
  }
  if (!Array.isArray(result.models)) {
    return { ok: false, error: "model_scan_models_missing" };
  }
  const models: ModelBehavior[] = [];
  for (const entry of result.models) {
    const parsed = parseModelBehavior(entry);
    if (!parsed.ok) {
      return parsed;
    }
    models.push(parsed.value);
  }
  const metadata = parseModelScanMetadata(result.scan_metadata);
  if (!metadata.ok) {
    return metadata;
  }
  return {
    ok: true,
    value: {
      models,
      scan_metadata: metadata.value,
      schema_version: "studio.model-scan.v1",
    },
  };
}

/**
 * Validate a model-scan job submit receipt and bind ``job.job_id`` / kind.
 */
export function validateModelScanJobReceipt(
  receipt: unknown,
): ValidationResult<ModelScanJobReceipt> {
  if (!isRecord(receipt)) {
    return { ok: false, error: "model_scan_job_receipt_invalid" };
  }
  if (receipt.schema_version !== "studio.model-scan.job.v1") {
    return { ok: false, error: "model_scan_job_receipt_schema_invalid" };
  }
  if (receipt.execution_mode !== "async_job") {
    return { ok: false, error: "model_scan_job_receipt_mode_invalid" };
  }
  if (!isNonEmptyString(receipt.job_id)) {
    return { ok: false, error: "model_scan_job_receipt_job_id_invalid" };
  }
  if (!isNonEmptyString(receipt.status_route)) {
    return { ok: false, error: "model_scan_job_receipt_status_route_invalid" };
  }
  if (!isRecord(receipt.job)) {
    return { ok: false, error: "model_scan_job_receipt_job_missing" };
  }
  if (receipt.job.kind !== "model_scan") {
    return { ok: false, error: "model_scan_job_receipt_kind_mismatch" };
  }
  if (receipt.job.job_id !== receipt.job_id) {
    return { ok: false, error: "model_scan_job_receipt_job_id_mismatch" };
  }
  const jobStatus = receipt.job.status;
  if (
    jobStatus !== "pending"
    && jobStatus !== "running"
    && jobStatus !== "completed"
    && jobStatus !== "failed"
    && jobStatus !== "cancelling"
    && jobStatus !== "cancelled"
    && jobStatus !== "timed_out"
  ) {
    return { ok: false, error: "model_scan_job_receipt_status_invalid" };
  }
  // Reconstruct a minimal typed receipt; nested job fields beyond binding are
  // re-validated on each poll via validateModelScanPollRecord.
  const job: StudioJobRecord = {
    artifacts: Array.isArray(receipt.job.artifacts)
      ? (receipt.job.artifacts as StudioJobRecord["artifacts"])
      : [],
    created_at_utc:
      typeof receipt.job.created_at_utc === "string"
        ? receipt.job.created_at_utc
        : "",
    error: typeof receipt.job.error === "string" ? receipt.job.error : null,
    execution_model:
      receipt.job.execution_model === "process" ? "process" : "thread",
    finished_at_utc:
      typeof receipt.job.finished_at_utc === "string"
        ? receipt.job.finished_at_utc
        : null,
    job_id: receipt.job_id,
    kind: "model_scan",
    owner: typeof receipt.job.owner === "string" ? receipt.job.owner : "studio",
    request_id:
      typeof receipt.job.request_id === "string" ? receipt.job.request_id : null,
    result: isRecord(receipt.job.result) ? receipt.job.result : null,
    started_at_utc:
      typeof receipt.job.started_at_utc === "string"
        ? receipt.job.started_at_utc
        : null,
    status: jobStatus,
  };
  return {
    ok: true,
    value: {
      execution_mode: "async_job",
      job,
      job_id: receipt.job_id,
      schema_version: "studio.model-scan.job.v1",
      status_route: receipt.status_route,
    },
  };
}

const JOB_STATUSES = new Set([
  "pending",
  "running",
  "completed",
  "failed",
  "cancelling",
  "cancelled",
  "timed_out",
]);

/**
 * Bind a polled job record to the retained session job id and model_scan kind.
 */
export function validateModelScanPollRecord(
  record: unknown,
  expectedJobId: string | null,
): ValidationResult<StudioJobRecord> {
  if (!isRecord(record)) {
    return { ok: false, error: "model_scan_poll_record_invalid" };
  }
  if (!isNonEmptyString(record.job_id)) {
    return { ok: false, error: "model_scan_poll_job_id_invalid" };
  }
  if (record.kind !== "model_scan") {
    return { ok: false, error: "model_scan_poll_kind_mismatch" };
  }
  if (expectedJobId === null || record.job_id !== expectedJobId) {
    return { ok: false, error: "model_scan_poll_job_id_mismatch" };
  }
  if (typeof record.status !== "string" || !JOB_STATUSES.has(record.status)) {
    return { ok: false, error: "model_scan_poll_status_invalid" };
  }
  const status = record.status as StudioJobRecord["status"];
  return {
    ok: true,
    value: {
      artifacts: Array.isArray(record.artifacts)
        ? (record.artifacts as StudioJobRecord["artifacts"])
        : [],
      created_at_utc:
        typeof record.created_at_utc === "string" ? record.created_at_utc : "",
      error: typeof record.error === "string" ? record.error : null,
      execution_model: record.execution_model === "process" ? "process" : "thread",
      finished_at_utc:
        typeof record.finished_at_utc === "string" ? record.finished_at_utc : null,
      job_id: record.job_id,
      kind: "model_scan",
      owner: typeof record.owner === "string" ? record.owner : "studio",
      request_id: typeof record.request_id === "string" ? record.request_id : null,
      result: isRecord(record.result) ? record.result : null,
      started_at_utc:
        typeof record.started_at_utc === "string" ? record.started_at_utc : null,
      status,
    },
  };
}
