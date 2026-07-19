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
  AnalysisJobResult,
  AnalysisResultMetadata,
  BifurcationResponse,
  FICurveResponse,
  HeatmapResponse,
  SensitivityResponse,
} from "./api/client";
import type { ValidationResult } from "./analysisJobRecordValidation";

export type { ValidationResult } from "./analysisJobRecordValidation";
export {
  validateAnalysisJobReceipt,
  validateAnalysisPollRecord,
} from "./analysisJobRecordValidation";

const HEX64 = /^[0-9a-fA-F]{64}$/;
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
