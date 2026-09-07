// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fail-closed analysis-job request policy

/**
 * Turning what the reader filled in into a request the server will accept.
 *
 * Every number is checked here rather than at the server, and a request that
 * would not make sense is refused before it is sent. That is not politeness:
 * an analysis job runs for minutes, and a sweep whose two axes are the same
 * parameter, or whose step is `NaN`, costs that time before failing. Refusing
 * it here costs nothing and says which field was wrong.
 *
 * The payloads themselves are built by `studioSimulationConfig.ts`, which the
 * synchronous panels use too. This module adds the job wrapper and the checks,
 * and invents nothing about the request that those builders do not already
 * express.
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

/**
 * Which analysis to run. The two sweeps carry their own inputs, so a selection
 * cannot name a sweep analysis without the sweep it needs.
 */
export type AnalysisJobSelection =
  | { analysis: "fi_curve" }
  | { analysis: "sensitivity" }
  | { analysis: "bifurcation"; sweep: StudioBifurcationSweepInput }
  | { analysis: "heatmap"; sweep: StudioHeatmapSweepInput };

/** The request to send, or the identifier of what stopped it being built. */
export type AnalysisJobRequestBuildResult =
  | { ok: true; value: AnalysisJobRequestBody }
  | { ok: false; error: string };

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
 * Find the first parameter in a map that is not a real number.
 *
 * @param values - The parameters.
 * @param error - The identifier to refuse the whole map with.
 * @returns The refusal, or `null` when every value is finite.
 */
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
 * Check the numbers every analysis needs, whatever it is.
 *
 * @param input - The simulation configuration the reader filled in.
 * @returns The identifier of the first field that was wrong, or `null` when
 *   all of them are numbers.
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

/**
 * Check a one-parameter sweep.
 *
 * @param sweep - The sweep the reader filled in.
 * @returns The identifier of what was wrong, or `null`.
 */
function validateBifurcationSweep(sweep: StudioBifurcationSweepInput): string | null {
  if (sweep.sweepParam.trim().length === 0) {
    return "analysis_request_sweep_param_blank";
  }
  if (!isFiniteNumber(sweep.parameterValue)) {
    return "analysis_request_sweep_value_invalid";
  }
  return null;
}

/**
 * Check a two-parameter sweep.
 *
 * Identical axes are refused: the run would be valid and the heatmap
 * meaningless, which is worse than a refusal because it looks like a
 * result.
 *
 * @param sweep - The sweep the reader filled in.
 * @returns The identifier of what was wrong, or `null`.
 */
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
 * Build the request for one analysis.
 *
 * A refused request is a returned result, never a thrown error: the caller is
 * a form handler, and a rejected form is an ordinary outcome rather than an
 * exceptional one.
 *
 * @param input - The simulation configuration.
 * @param selection - Which analysis to run, with the sweep it needs.
 * @returns The request body, or the identifier of the field that was wrong.
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
      return { ok: false, error: `analysis_request_kind_unsupported:${String(_exhaustive)}` };
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
