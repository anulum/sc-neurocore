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

/** The Studio tabs an analysis result can be shown in. */
export type StudioAnalysisResultViewTab =
  | "fi-curve"
  | "bifurcation"
  | "heatmap"
  | "sensitivity";

/**
 * A patch that stores one analysis result and opens its tab.
 *
 * Each member clears `error`, because a patch is only built from a result that
 * has already passed every check.
 */
export type StudioAnalysisResultSinkPatch =
  | (StudioFICurveResultStatePatch & { activeTab: "fi-curve"; error: null })
  | (StudioBifurcationResultStatePatch & { activeTab: "bifurcation"; error: null })
  | (StudioHeatmapResultStatePatch & { activeTab: "heatmap"; error: null })
  | (StudioSensitivityResultStatePatch & { activeTab: "sensitivity"; error: null });

/** Either a patch to apply, or the identifier of the reason there is none. */
export type StudioAnalysisResultSinkResult =
  | { ok: true; patch: StudioAnalysisResultSinkPatch }
  | { ok: false; error: string };

/**
 * Whether a value is a plain object.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Whether a value is an array of numbers.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isNumberArray(value: unknown): value is number[] {
  return Array.isArray(value) && value.every((item) => typeof item === "number");
}

/**
 * Whether a value is an array of arrays of numbers.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isNestedNumberArray(value: unknown): value is number[][] {
  return Array.isArray(value) && value.every((row) => isNumberArray(row));
}

/**
 * Read the analysis type a result declares about itself.
 *
 * @param result - The job's result.
 * @returns The declared type, or `null` when the result declares none.
 */
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

/**
 * Whether a result has the shape of an f-I curve.
 *
 * @param result - The job's result.
 * @returns Whether it is one.
 */
function isFICurveResponse(result: AnalysisJobResult): result is FICurveResponse {
  return (
    isRecord(result)
    && "currents" in result
    && isNumberArray(result.currents)
    && "rates" in result
    && isNumberArray(result.rates)
  );
}

/**
 * Whether a result has the shape of a bifurcation sweep.
 *
 * @param result - The job's result.
 * @returns Whether it is one.
 */
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

/**
 * Whether a result has the shape of a two-parameter heatmap.
 *
 * @param result - The job's result.
 * @returns Whether it is one.
 */
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

/**
 * Whether a result has the shape of a sensitivity analysis.
 *
 * @param result - The job's result.
 * @returns Whether it is one.
 */
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
 * Name the view tab that shows a given kind of analysis.
 *
 * @param kind - The job's kind.
 * @returns The tab that displays its result.
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
 * Turn a finished analysis job's result into the patch that displays it.
 *
 * The result is checked twice over: the metadata must say it is the kind of
 * analysis that was asked for, and the body must have that kind's shape. A
 * result that passes the first check and fails the second is a server that
 * answered the wrong question, which is worth refusing rather than plotting.
 *
 * @param kind - The kind of analysis the job ran.
 * @param result - The job's result, as it arrived.
 * @returns The patch, or the reason the result was refused. Each reason is a
 *   stable identifier rather than a sentence, so callers can branch on it.
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
