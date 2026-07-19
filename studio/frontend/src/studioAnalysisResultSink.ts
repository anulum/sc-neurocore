// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Pure async analysis result → store state sink

/**
 * Maps a completed analysis-job result into the same store patches used by the
 * legacy sync runners. No network, store imports, or second full validators.
 *
 * Fail-closed and assert-free: structural type predicates only (no `as` / `any`
 * / non-null assertions on the product path).
 */

import type {
  AnalysisJobKind,
  AnalysisJobResult,
  BifurcationResponse,
  FICurveResponse,
  HeatmapResponse,
  SensitivityResponse,
} from "./api/client";
import {
  studioBifurcationResultState,
  studioFICurveResultState,
  studioHeatmapResultState,
  studioSensitivityResultState,
  type StudioBifurcationResultStatePatch,
  type StudioFICurveResultStatePatch,
  type StudioHeatmapResultStatePatch,
  type StudioSensitivityResultStatePatch,
} from "./studioAnalysisState";

export type StudioAnalysisResultViewTab =
  | "fi-curve"
  | "bifurcation"
  | "heatmap"
  | "sensitivity";

export type StudioAnalysisResultSinkPatch =
  | (StudioFICurveResultStatePatch & { activeTab: "fi-curve"; error: null })
  | (StudioBifurcationResultStatePatch & { activeTab: "bifurcation"; error: null })
  | (StudioHeatmapResultStatePatch & { activeTab: "heatmap"; error: null })
  | (StudioSensitivityResultStatePatch & { activeTab: "sensitivity"; error: null });

export type StudioAnalysisResultSinkResult =
  | { ok: true; patch: StudioAnalysisResultSinkPatch }
  | { ok: false; error: string };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNumberArray(value: unknown): value is number[] {
  return Array.isArray(value) && value.every((item) => typeof item === "number");
}

function isNestedNumberArray(value: unknown): value is number[][] {
  return Array.isArray(value) && value.every((row) => isNumberArray(row));
}

function metadataType(result: AnalysisJobResult): string | null {
  if (!isRecord(result)) {
    return null;
  }
  const metadata = result.analysis_metadata;
  if (!isRecord(metadata)) {
    return null;
  }
  const analysisType = metadata.analysis_type;
  return typeof analysisType === "string" && analysisType.length > 0
    ? analysisType
    : null;
}

function isFICurveResponse(result: AnalysisJobResult): result is FICurveResponse {
  return (
    isRecord(result)
    && "currents" in result
    && isNumberArray(result.currents)
    && "rates" in result
    && isNumberArray(result.rates)
  );
}

function isBifurcationResponse(
  result: AnalysisJobResult,
): result is BifurcationResponse {
  return (
    isRecord(result)
    && typeof result.param_name === "string"
    && result.param_name.length > 0
    && isNumberArray(result.param_values)
    && isNestedNumberArray(result.attractors)
  );
}

function isHeatmapResponse(result: AnalysisJobResult): result is HeatmapResponse {
  return (
    isRecord(result)
    && typeof result.param_x === "string"
    && result.param_x.length > 0
    && typeof result.param_y === "string"
    && result.param_y.length > 0
    && isNumberArray(result.x_values)
    && isNumberArray(result.y_values)
    && isNestedNumberArray(result.rates)
  );
}

function isSensitivityResponse(
  result: AnalysisJobResult,
): result is SensitivityResponse {
  if (!isRecord(result) || typeof result.base_rate !== "number") {
    return false;
  }
  if (!Array.isArray(result.sensitivities)) {
    return false;
  }
  return result.sensitivities.every(
    (entry) =>
      isRecord(entry)
      && typeof entry.param === "string"
      && typeof entry.sensitivity === "number"
      && typeof entry.rate_minus === "number"
      && typeof entry.rate_plus === "number",
  );
}

/**
 * Map analysis job kind to the Studio view tab used by sync runners.
 */
export function studioAnalysisResultViewTab(
  kind: AnalysisJobKind,
): StudioAnalysisResultViewTab {
  switch (kind) {
    case "fi_curve":
      return "fi-curve";
    case "bifurcation":
      return "bifurcation";
    case "heatmap":
      return "heatmap";
    case "sensitivity":
      return "sensitivity";
    default: {
      const _exhaustive: never = kind;
      return _exhaustive;
    }
  }
}

/**
 * Build the store patch for a completed async analysis job of the given kind.
 */
export function studioAnalysisResultSink(
  kind: AnalysisJobKind,
  result: AnalysisJobResult,
): StudioAnalysisResultSinkResult {
  const declared = metadataType(result);
  if (declared === null) {
    return { ok: false, error: "analysis_result_sink_metadata_missing" };
  }
  if (declared !== kind) {
    return {
      ok: false,
      error: `analysis_result_sink_kind_mismatch:${kind}:${declared}`,
    };
  }

  switch (kind) {
    case "fi_curve": {
      if (!isFICurveResponse(result)) {
        return { ok: false, error: "analysis_result_sink_fi_curve_shape_invalid" };
      }
      return {
        ok: true,
        patch: {
          ...studioFICurveResultState(result),
          activeTab: "fi-curve",
          error: null,
        },
      };
    }
    case "bifurcation": {
      if (!isBifurcationResponse(result)) {
        return { ok: false, error: "analysis_result_sink_bifurcation_shape_invalid" };
      }
      return {
        ok: true,
        patch: {
          ...studioBifurcationResultState(result),
          activeTab: "bifurcation",
          error: null,
        },
      };
    }
    case "heatmap": {
      if (!isHeatmapResponse(result)) {
        return { ok: false, error: "analysis_result_sink_heatmap_shape_invalid" };
      }
      return {
        ok: true,
        patch: {
          ...studioHeatmapResultState(result),
          activeTab: "heatmap",
          error: null,
        },
      };
    }
    case "sensitivity": {
      if (!isSensitivityResponse(result)) {
        return { ok: false, error: "analysis_result_sink_sensitivity_shape_invalid" };
      }
      return {
        ok: true,
        patch: {
          ...studioSensitivityResultState(result),
          activeTab: "sensitivity",
          error: null,
        },
      };
    }
    default: {
      const _exhaustive: never = kind;
      return {
        ok: false,
        error: `analysis_result_sink_kind_unsupported:${String(_exhaustive)}`,
      };
    }
  }
}
