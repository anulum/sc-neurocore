// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fail-closed runtime guards for analysis job payloads

/**
 * Nested type guards for analysis-job receipts, poll records, and completed
 * results for fi_curve, bifurcation, heatmap, and sensitivity.
 */

import type {
  AnalysisJobKind,
  AnalysisJobReceipt,
  AnalysisJobResult,
  AnalysisResultMetadata,
  BifurcationResponse,
  FICurveResponse,
  HeatmapResponse,
  SensitivityResponse,
  StudioJobRecord,
} from "./api/client";

export type ValidationResult<T> =
  | { ok: true; value: T }
  | { ok: false; error: string };

const HEX64 = /^[0-9a-fA-F]{64}$/;
const JOB_STATUSES = new Set([
  "pending",
  "running",
  "completed",
  "failed",
  "cancelling",
  "cancelled",
  "timed_out",
]);
const ANALYSIS_KINDS = new Set<AnalysisJobKind>([
  "fi_curve",
  "bifurcation",
  "heatmap",
  "sensitivity",
]);
const ANALYSIS_SOURCES = new Set(["ode", "model", "mixed", "unknown"]);

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

function parseFiniteNumberArray(value: unknown, error: string): ValidationResult<number[]> {
  if (!Array.isArray(value)) {
    return { ok: false, error };
  }
  const out: number[] = [];
  for (const entry of value) {
    if (!isFiniteNumber(entry)) {
      return { ok: false, error };
    }
    out.push(entry);
  }
  return { ok: true, value: out };
}

/**
 * Guard analysis_metadata for a completed analysis evidence object.
 */
export function parseAnalysisResultMetadata(
  value: unknown,
  expectedType: AnalysisJobKind,
): ValidationResult<AnalysisResultMetadata> {
  if (!isRecord(value)) {
    return { ok: false, error: "analysis_metadata_missing" };
  }
  if (value.schema_version !== "studio.analysis-result.v1") {
    return { ok: false, error: "analysis_metadata_schema_mismatch" };
  }
  if (value.evidence_classification !== "analysis") {
    return { ok: false, error: "analysis_metadata_class_invalid" };
  }
  if (value.status !== "completed") {
    return { ok: false, error: "analysis_metadata_status_invalid" };
  }
  if (value.analysis_type !== expectedType) {
    return { ok: false, error: "analysis_metadata_type_mismatch" };
  }
  if (typeof value.source !== "string" || !ANALYSIS_SOURCES.has(value.source)) {
    return { ok: false, error: "analysis_metadata_source_invalid" };
  }
  if (!isSha256Hex(value.input_sha256)) {
    return { ok: false, error: "analysis_metadata_input_sha256_invalid" };
  }
  if (!isSha256Hex(value.result_sha256)) {
    return { ok: false, error: "analysis_metadata_result_sha256_invalid" };
  }
  if (!Array.isArray(value.output_keys)) {
    return { ok: false, error: "analysis_metadata_output_keys_invalid" };
  }
  const outputKeys: string[] = [];
  for (const key of value.output_keys) {
    if (!isNonEmptyString(key)) {
      return { ok: false, error: "analysis_metadata_output_keys_invalid" };
    }
    outputKeys.push(key);
  }
  return {
    ok: true,
    value: {
      analysis_type: expectedType,
      evidence_classification: "analysis",
      input_sha256: value.input_sha256,
      output_keys: outputKeys,
      result_sha256: value.result_sha256,
      schema_version: "studio.analysis-result.v1",
      source: value.source as AnalysisResultMetadata["source"],
      status: "completed",
    },
  };
}

function parseFiCurve(
  body: Record<string, unknown>,
  metadata: AnalysisResultMetadata,
): ValidationResult<FICurveResponse> {
  const currents = parseFiniteNumberArray(body.currents, "fi_curve_currents_invalid");
  if (!currents.ok) {
    return currents;
  }
  const rates = parseFiniteNumberArray(body.rates, "fi_curve_rates_invalid");
  if (!rates.ok) {
    return rates;
  }
  if (currents.value.length !== rates.value.length) {
    return { ok: false, error: "fi_curve_length_mismatch" };
  }
  return {
    ok: true,
    value: {
      analysis_metadata: metadata,
      currents: currents.value,
      rates: rates.value,
    },
  };
}

function parseBifurcation(
  body: Record<string, unknown>,
  metadata: AnalysisResultMetadata,
): ValidationResult<BifurcationResponse> {
  if (!isNonEmptyString(body.param_name)) {
    return { ok: false, error: "bifurcation_param_name_invalid" };
  }
  const paramValues = parseFiniteNumberArray(
    body.param_values,
    "bifurcation_param_values_invalid",
  );
  if (!paramValues.ok) {
    return paramValues;
  }
  if (!Array.isArray(body.attractors)) {
    return { ok: false, error: "bifurcation_attractors_invalid" };
  }
  const attractors: number[][] = [];
  for (const row of body.attractors) {
    const parsed = parseFiniteNumberArray(row, "bifurcation_attractors_invalid");
    if (!parsed.ok) {
      return parsed;
    }
    attractors.push(parsed.value);
  }
  if (attractors.length !== paramValues.value.length) {
    return { ok: false, error: "bifurcation_length_mismatch" };
  }
  return {
    ok: true,
    value: {
      analysis_metadata: metadata,
      attractors,
      param_name: body.param_name,
      param_values: paramValues.value,
    },
  };
}

function parseHeatmap(
  body: Record<string, unknown>,
  metadata: AnalysisResultMetadata,
): ValidationResult<HeatmapResponse> {
  if (!isNonEmptyString(body.param_x) || !isNonEmptyString(body.param_y)) {
    return { ok: false, error: "heatmap_param_names_invalid" };
  }
  const xValues = parseFiniteNumberArray(body.x_values, "heatmap_x_values_invalid");
  if (!xValues.ok) {
    return xValues;
  }
  const yValues = parseFiniteNumberArray(body.y_values, "heatmap_y_values_invalid");
  if (!yValues.ok) {
    return yValues;
  }
  if (!isFiniteNumber(body.rate_min) || !isFiniteNumber(body.rate_max)) {
    return { ok: false, error: "heatmap_rate_bounds_invalid" };
  }
  if (!Array.isArray(body.rates)) {
    return { ok: false, error: "heatmap_rates_invalid" };
  }
  const rates: number[][] = [];
  for (const row of body.rates) {
    const parsed = parseFiniteNumberArray(row, "heatmap_rates_invalid");
    if (!parsed.ok) {
      return parsed;
    }
    if (parsed.value.length !== xValues.value.length) {
      return { ok: false, error: "heatmap_rates_width_mismatch" };
    }
    rates.push(parsed.value);
  }
  if (rates.length !== yValues.value.length) {
    return { ok: false, error: "heatmap_rates_height_mismatch" };
  }
  return {
    ok: true,
    value: {
      analysis_metadata: metadata,
      param_x: body.param_x,
      param_y: body.param_y,
      rate_max: body.rate_max,
      rate_min: body.rate_min,
      rates,
      x_values: xValues.value,
      y_values: yValues.value,
    },
  };
}

function parseSensitivity(
  body: Record<string, unknown>,
  metadata: AnalysisResultMetadata,
): ValidationResult<SensitivityResponse> {
  if (!isFiniteNumber(body.base_rate)) {
    return { ok: false, error: "sensitivity_base_rate_invalid" };
  }
  if (!Array.isArray(body.sensitivities)) {
    return { ok: false, error: "sensitivity_list_invalid" };
  }
  const sensitivities: SensitivityResponse["sensitivities"] = [];
  for (const entry of body.sensitivities) {
    if (!isRecord(entry)) {
      return { ok: false, error: "sensitivity_entry_invalid" };
    }
    if (!isNonEmptyString(entry.param)) {
      return { ok: false, error: "sensitivity_param_invalid" };
    }
    if (!isFiniteNumber(entry.sensitivity)) {
      return { ok: false, error: "sensitivity_value_invalid" };
    }
    if (!isFiniteNumber(entry.rate_minus) || !isFiniteNumber(entry.rate_plus)) {
      return { ok: false, error: "sensitivity_rates_invalid" };
    }
    sensitivities.push({
      param: entry.param,
      rate_minus: entry.rate_minus,
      rate_plus: entry.rate_plus,
      sensitivity: entry.sensitivity,
    });
  }
  return {
    ok: true,
    value: {
      analysis_metadata: metadata,
      base_rate: body.base_rate,
      sensitivities,
    },
  };
}

/**
 * Validate a completed analysis job result for the selected analysis kind.
 */
export function validateAnalysisJobResult(
  result: unknown,
  expectedKind: AnalysisJobKind,
): ValidationResult<AnalysisJobResult> {
  if (!isRecord(result)) {
    return { ok: false, error: "analysis_result_not_object" };
  }
  const metadata = parseAnalysisResultMetadata(result.analysis_metadata, expectedKind);
  if (!metadata.ok) {
    return metadata;
  }
  switch (expectedKind) {
    case "fi_curve":
      return parseFiCurve(result, metadata.value);
    case "bifurcation":
      return parseBifurcation(result, metadata.value);
    case "heatmap":
      return parseHeatmap(result, metadata.value);
    case "sensitivity":
      return parseSensitivity(result, metadata.value);
    default: {
      const _exhaustive: never = expectedKind;
      return { ok: false, error: `analysis_kind_unsupported:${_exhaustive}` };
    }
  }
}

/**
 * Validate an analysis job submit receipt and bind kind/id.
 */
export function validateAnalysisJobReceipt(
  receipt: unknown,
  expectedKind: AnalysisJobKind,
): ValidationResult<AnalysisJobReceipt> {
  if (!isRecord(receipt)) {
    return { ok: false, error: "analysis_job_receipt_invalid" };
  }
  if (receipt.schema_version !== "studio.analysis.job.v1") {
    return { ok: false, error: "analysis_job_receipt_schema_invalid" };
  }
  if (receipt.execution_mode !== "async_job") {
    return { ok: false, error: "analysis_job_receipt_mode_invalid" };
  }
  if (receipt.analysis !== expectedKind || !ANALYSIS_KINDS.has(receipt.analysis as AnalysisJobKind)) {
    return { ok: false, error: "analysis_job_receipt_analysis_mismatch" };
  }
  if (!isNonEmptyString(receipt.job_id)) {
    return { ok: false, error: "analysis_job_receipt_job_id_invalid" };
  }
  if (!isNonEmptyString(receipt.status_route)) {
    return { ok: false, error: "analysis_job_receipt_status_route_invalid" };
  }
  if (!isRecord(receipt.job)) {
    return { ok: false, error: "analysis_job_receipt_job_missing" };
  }
  if (receipt.job.kind !== "analysis") {
    return { ok: false, error: "analysis_job_receipt_kind_mismatch" };
  }
  if (receipt.job.job_id !== receipt.job_id) {
    return { ok: false, error: "analysis_job_receipt_job_id_mismatch" };
  }
  const jobStatus = receipt.job.status;
  if (typeof jobStatus !== "string" || !JOB_STATUSES.has(jobStatus)) {
    return { ok: false, error: "analysis_job_receipt_status_invalid" };
  }
  const job: StudioJobRecord = {
    artifacts: Array.isArray(receipt.job.artifacts)
      ? (receipt.job.artifacts as StudioJobRecord["artifacts"])
      : [],
    created_at_utc:
      typeof receipt.job.created_at_utc === "string" ? receipt.job.created_at_utc : "",
    error: typeof receipt.job.error === "string" ? receipt.job.error : null,
    execution_model: receipt.job.execution_model === "process" ? "process" : "thread",
    finished_at_utc:
      typeof receipt.job.finished_at_utc === "string" ? receipt.job.finished_at_utc : null,
    job_id: receipt.job_id,
    kind: "analysis",
    owner: typeof receipt.job.owner === "string" ? receipt.job.owner : "studio",
    request_id:
      typeof receipt.job.request_id === "string" ? receipt.job.request_id : null,
    result: isRecord(receipt.job.result) ? receipt.job.result : null,
    started_at_utc:
      typeof receipt.job.started_at_utc === "string" ? receipt.job.started_at_utc : null,
    status: jobStatus as StudioJobRecord["status"],
  };
  const value: AnalysisJobReceipt = {
    analysis: expectedKind,
    execution_mode: "async_job",
    job,
    job_id: receipt.job_id,
    schema_version: "studio.analysis.job.v1",
    status_route: receipt.status_route,
  };
  if (isFiniteNumber(receipt.projected_simulations)) {
    value.projected_simulations = receipt.projected_simulations;
  }
  if (isFiniteNumber(receipt.duration_ms)) {
    value.duration_ms = receipt.duration_ms;
  }
  if (isFiniteNumber(receipt.dt_ms)) {
    value.dt_ms = receipt.dt_ms;
  }
  return { ok: true, value };
}

/**
 * Bind a polled job record to the retained analysis job id and kind.
 */
export function validateAnalysisPollRecord(
  record: unknown,
  expectedJobId: string | null,
): ValidationResult<StudioJobRecord> {
  if (!isRecord(record)) {
    return { ok: false, error: "analysis_poll_record_invalid" };
  }
  if (!isNonEmptyString(record.job_id)) {
    return { ok: false, error: "analysis_poll_job_id_invalid" };
  }
  if (record.kind !== "analysis") {
    return { ok: false, error: "analysis_poll_kind_mismatch" };
  }
  if (expectedJobId === null || record.job_id !== expectedJobId) {
    return { ok: false, error: "analysis_poll_job_id_mismatch" };
  }
  if (typeof record.status !== "string" || !JOB_STATUSES.has(record.status)) {
    return { ok: false, error: "analysis_poll_status_invalid" };
  }
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
      kind: "analysis",
      owner: typeof record.owner === "string" ? record.owner : "studio",
      request_id: typeof record.request_id === "string" ? record.request_id : null,
      result: isRecord(record.result) ? record.result : null,
      started_at_utc:
        typeof record.started_at_utc === "string" ? record.started_at_utc : null,
      status: record.status as StudioJobRecord["status"],
    },
  };
}
