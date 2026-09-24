// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fail-closed runtime guards for model-scan job payloads

/**
 * Reading a catalogue scan's answer, field by field.
 *
 * No untrusted JSON is cast here. Every entry of every nested list is checked
 * and a bad one refuses the whole payload rather than being dropped: a scan
 * result is what the reader will cite about a catalogue of models, and one
 * silently missing model makes it a different claim.
 *
 * Refusals are stable identifiers naming the field, so a panel can say what
 * was wrong rather than only that something was.
 */

import { parseStudioJobArtifact } from "./analysisJobRecordValidation";
import type {
  ModelBehavior,
  ModelScanFailure,
  ModelScanJobReceipt,
  ModelScanMetadata,
  ModelScanResponse,
  StudioJobRecord,
} from "./api/client";

/**
 * Either a value that has been checked, or the identifier of what was wrong.
 * Never both, and never a value that has only been asserted.
 */
export type ValidationResult<T> =
  | { ok: true; value: T }
  | { ok: false; error: string };

/** A 64-character hexadecimal digest, in either case. */
const HEX64 = /^[0-9a-fA-F]{64}$/;

/**
 * Whether a value is a plain object.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/**
 * Whether a value is a number that is actually a number.
 *
 * @param value - The value.
 * @returns Whether it is finite.
 */
function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * Whether a value is a string with something in it.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

/**
 * Whether a value looks like a SHA-256 digest.
 *
 * @param value - The value.
 * @returns Whether it has that shape.
 */
function isSha256Hex(value: unknown): value is string {
  return typeof value === "string" && HEX64.test(value);
}

/**
 * Read a job's artefact list.
 *
 * This module's contract is that untrusted JSON is never cast, and this list
 * used to be the exception: an array of anything was asserted to be an array
 * of artefacts, so a malformed entry reached callers with a digest field that
 * might be any type at all. Each entry is checked now, and a bad one refuses
 * the whole record rather than being dropped -- a job whose artefact list is
 * partly unreadable is a job whose evidence cannot be trusted.
 *
 * @param value - The list, as it arrived. A missing list is read as no
 *   artefacts, which is what a job that has produced none sends.
 * @returns The artefacts, or the identifier of the first bad entry.
 */
function parseJobArtifacts(value: unknown): ValidationResult<StudioJobRecord["artifacts"]> {
  if (value === undefined || value === null) {
    return { ok: true, value: [] };
  }
  if (!Array.isArray(value)) {
    return { ok: false, error: "model_scan_job_artifacts_invalid" };
  }
  const artifacts: StudioJobRecord["artifacts"] = [];
  for (const entry of value) {
    const parsed = parseStudioJobArtifact(entry);
    if (!parsed.ok) {
      return { ok: false, error: `model_scan_${parsed.error}` };
    }
    artifacts.push(parsed.value);
  }
  return { ok: true, value: artifacts };
}

/**
 * Read one model's reported behaviour.
 *
 * @param value - The entry, as it arrived.
 * @returns The behaviour, or the identifier of the field that was wrong.
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
 * Read one model the scan could not run, and why.
 *
 * These are not job failures: a model that would not run is reported inside a
 * completed result, so one broken model does not cost the reader the rest.
 *
 * @param value - The entry, as it arrived.
 * @returns The failure, or the identifier of the field that was wrong.
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
 * Read the firing-pattern histogram.
 *
 * @param value - The counts, as they arrived.
 * @returns The counts, or the identifier of what was wrong. A single bad entry
 *   refuses the whole histogram: a count that is partly readable is a
 *   histogram that does not add up.
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
 * Read a scan's metadata: its inputs, its digests, and what it could not run.
 *
 * The status must be `completed`. Metadata is only read from a completed job,
 * so metadata claiming anything else is a result that disagrees with the job
 * carrying it.
 *
 * @param value - The metadata, as it arrived.
 * @returns The metadata, or the identifier of what was wrong.
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
 * Read a completed scan's result.
 *
 * @param result - The result, as it arrived.
 * @returns The scan, or the identifier of the first thing that was wrong. One
 *   unreadable model refuses the whole result rather than being dropped -- a
 *   catalogue scan missing a model it does not mention is a different claim
 *   from the one the reader asked for.
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
 * Read a submit receipt, and bind it to the scan that was asked for.
 *
 * The receipt's own job identifier and the nested job's must agree, and the
 * kind must be a model scan. A receipt for another job would set the session
 * polling something whose result it cannot read.
 *
 * The job is reconstructed field by field rather than taken whole, because
 * only the binding matters here; every later poll re-validates the record.
 *
 * @param receipt - The receipt, as it arrived.
 * @returns The receipt, or the identifier of what was wrong.
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
    && jobStatus !== "interrupted"
    && jobStatus !== "unknown"
  ) {
    return { ok: false, error: "model_scan_job_receipt_status_invalid" };
  }
  // Reconstruct a minimal typed receipt; nested job fields beyond binding are
  // re-validated on each poll via validateModelScanPollRecord.
  const receiptArtifacts = parseJobArtifacts(receipt.job.artifacts);
  if (!receiptArtifacts.ok) {
    return receiptArtifacts;
  }
  const job: StudioJobRecord = {
    artifacts: receiptArtifacts.value,
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
    training_config: null,
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

/**
 * Every status the job contract defines.
 *
 * Typed as the contract's own union, so a status added to the contract without
 * being added here is a type error rather than a record refused at runtime,
 * and so the check below narrows instead of needing a cast.
 */
const JOB_STATUSES: readonly StudioJobRecord["status"][] = [
  "pending",
  "running",
  "completed",
  "failed",
  "cancelling",
  "cancelled",
  "timed_out",
  "interrupted",
  "unknown",
];

/**
 * Whether a value is one of the contract's job statuses.
 *
 * @param value - The value, as it arrived.
 * @returns Whether it is a status this build knows.
 */
function isJobStatus(value: unknown): value is StudioJobRecord["status"] {
  return typeof value === "string" && JOB_STATUSES.some((status) => status === value);
}

/**
 * Read a poll response, and bind it to the scan being polled.
 *
 * @param record - The response, as it arrived.
 * @param expectedJobId - The scan being polled. A `null` refuses every record,
 *   which is what a caller holding no job identifier wants.
 * @returns The record, or the identifier of what was wrong.
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
  if (!isJobStatus(record.status)) {
    return { ok: false, error: "model_scan_poll_status_invalid" };
  }
  const status = record.status;
  const artifacts = parseJobArtifacts(record.artifacts);
  if (!artifacts.ok) {
    return artifacts;
  }
  return {
    ok: true,
    value: {
      artifacts: artifacts.value,
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
      training_config: null,
    },
  };
}
