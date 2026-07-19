// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fail-closed analysis-job request policy

/**
 * Converts Studio simulation/sweep inputs into AnalysisJobRequestBody for
 * fi_curve, bifurcation, heatmap, and sensitivity. Reuses existing request
 * builders; does not invent API, evidence, or progress state.
 */

import type { AnalysisJobKind, AnalysisJobRequestBody } from "./api/client";
import {
  studioBifurcationRequest,
  studioFICurveRequest,
  studioHeatmapRequest,
  studioSimulationConfig,
  type StudioBifurcationSweepInput,
  type StudioHeatmapSweepInput,
  type StudioSimulationConfigInput,
} from "./studioSimulationConfig";

export type AnalysisJobSelection =
  | { analysis: "fi_curve" }
  | { analysis: "sensitivity" }
  | { analysis: "bifurcation"; sweep: StudioBifurcationSweepInput }
  | { analysis: "heatmap"; sweep: StudioHeatmapSweepInput };

export type AnalysisJobRequestBuildResult =
  | { ok: true; value: AnalysisJobRequestBody }
  | { ok: false; error: string };

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function recordHasNonFinite(
  values: Record<string, number>,
  error: string,
): string | null {
  for (const value of Object.values(values)) {
    if (!isFiniteNumber(value)) {
      return error;
    }
  }
  return null;
}

/**
 * Validate core simulation numerics shared by every analysis kind.
 */
export function validateStudioSimulationCoreNumerics(
  input: StudioSimulationConfigInput,
): string | null {
  if (!isFiniteNumber(input.dt)) {
    return "analysis_request_dt_invalid";
  }
  if (!isFiniteNumber(input.duration)) {
    return "analysis_request_duration_invalid";
  }
  if (!isFiniteNumber(input.current)) {
    return "analysis_request_current_invalid";
  }
  const modelParamsError = recordHasNonFinite(
    input.modelParams,
    "analysis_request_model_params_invalid",
  );
  if (modelParamsError !== null) {
    return modelParamsError;
  }
  const odeParamsError = recordHasNonFinite(
    input.odeParams,
    "analysis_request_ode_params_invalid",
  );
  if (odeParamsError !== null) {
    return odeParamsError;
  }
  const odeInitError = recordHasNonFinite(
    input.odeInit,
    "analysis_request_ode_init_invalid",
  );
  if (odeInitError !== null) {
    return odeInitError;
  }
  return null;
}

function validateBifurcationSweep(sweep: StudioBifurcationSweepInput): string | null {
  if (sweep.sweepParam.trim().length === 0) {
    return "analysis_request_sweep_param_blank";
  }
  if (!isFiniteNumber(sweep.parameterValue)) {
    return "analysis_request_sweep_value_invalid";
  }
  return null;
}

function validateHeatmapSweep(sweep: StudioHeatmapSweepInput): string | null {
  const x = sweep.sweepParamX.trim();
  const y = sweep.sweepParamY.trim();
  if (x.length === 0 || y.length === 0) {
    return "analysis_request_heatmap_param_blank";
  }
  if (x === y) {
    return "analysis_request_heatmap_axes_identical";
  }
  if (!isFiniteNumber(sweep.parameterValueX) || !isFiniteNumber(sweep.parameterValueY)) {
    return "analysis_request_heatmap_value_invalid";
  }
  return null;
}

/**
 * Build a typed async analysis job request from Studio simulation inputs.
 *
 * Returns a typed success/failure result; never throws for policy violations.
 */
export function buildAnalysisJobRequest(
  input: StudioSimulationConfigInput,
  selection: AnalysisJobSelection,
): AnalysisJobRequestBuildResult {
  const coreError = validateStudioSimulationCoreNumerics(input);
  if (coreError !== null) {
    return { ok: false, error: coreError };
  }

  const base = studioSimulationConfig(input);
  let analysis: AnalysisJobKind;
  let payload: Record<string, unknown>;

  switch (selection.analysis) {
    case "fi_curve": {
      analysis = "fi_curve";
      payload = studioFICurveRequest(base, input.current);
      break;
    }
    case "sensitivity": {
      analysis = "sensitivity";
      payload = { ...base };
      break;
    }
    case "bifurcation": {
      const sweepError = validateBifurcationSweep(selection.sweep);
      if (sweepError !== null) {
        return { ok: false, error: sweepError };
      }
      analysis = "bifurcation";
      payload = studioBifurcationRequest(base, {
        sweepParam: selection.sweep.sweepParam.trim(),
        parameterValue: selection.sweep.parameterValue,
      });
      break;
    }
    case "heatmap": {
      const sweepError = validateHeatmapSweep(selection.sweep);
      if (sweepError !== null) {
        return { ok: false, error: sweepError };
      }
      analysis = "heatmap";
      payload = studioHeatmapRequest(base, {
        sweepParamX: selection.sweep.sweepParamX.trim(),
        parameterValueX: selection.sweep.parameterValueX,
        sweepParamY: selection.sweep.sweepParamY.trim(),
        parameterValueY: selection.sweep.parameterValueY,
      });
      break;
    }
    default: {
      const _exhaustive: never = selection;
      return { ok: false, error: `analysis_request_kind_unsupported:${_exhaustive}` };
    }
  }

  return {
    ok: true,
    value: {
      analysis,
      payload,
    },
  };
}
