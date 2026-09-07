// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Pure UI/store analysis selection resolver

/**
 * Turning what the panel has selected into a typed analysis selection.
 *
 * The work is resolving a sweep *by name*: the reader picks a parameter from
 * whichever set is active -- the model's or the ODE's -- and this checks that
 * the name exists there and holds a real number. A name that is not in the
 * active set is refused rather than defaulted, because a sweep over a
 * parameter the run does not have would produce a flat, meaningless result
 * instead of an error.
 *
 * No defaults are invented and nothing is sent. The payload is built later, by
 * `analysisJobRequest.ts`.
 */

import type { AnalysisJobKind } from "./api/client";
import type { AnalysisJobSelection } from "./analysisJobRequest";
import type { StudioSimulationSourceMode } from "./studioSimulationConfig";

/**
 * The resolved selection with the label to show, or the identifier of what
 * stopped it resolving.
 */
export type StudioAnalysisJobSelectionResult =
  | { ok: true; selection: AnalysisJobSelection; label: string }
  | { ok: false; error: string };

/** What the panel has: the analysis, the active parameters, the sweep names. */
export interface StudioAnalysisJobSelectionInput {
  analysis: AnalysisJobKind;
  sourceMode: StudioSimulationSourceMode;
  modelParams: Record<string, number>;
  odeParams: Record<string, number>;
  sweepParam: string;
  sweepParamY: string;
}

/** What each analysis is called where the reader can see it. */
const LABELS: Readonly<Record<AnalysisJobKind, string>> = {
  fi_curve: "f-I curve",
  sensitivity: "sensitivity",
  bifurcation: "bifurcation",
  heatmap: "heatmap",
};

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
 * Choose the parameter set the reader is sweeping over.
 *
 * @param sourceMode - Whether the run is driven by a model or by an ODE.
 * @param modelParams - The model's parameters.
 * @param odeParams - The ODE's parameters.
 * @returns The set that is in force.
 */
function activeParams(
  sourceMode: StudioSimulationSourceMode,
  modelParams: Record<string, number>,
  odeParams: Record<string, number>,
): Record<string, number> {
  return sourceMode === "model" ? modelParams : odeParams;
}

/**
 * Resolve one sweep parameter by name.
 *
 * Presence is checked with `hasOwnProperty` rather than by reading the
 * value, so a parameter that legitimately holds `0` is found and one
 * inherited from the prototype chain is not.
 *
 * @param params - The active parameter set.
 * @param rawName - The name as the reader typed it.
 * @param errors - The identifiers to refuse with: blank, missing, or not a
 *   finite number.
 * @returns The trimmed name and its value, or the refusal.
 */
function resolveNamedParam(
  params: Record<string, number>,
  rawName: string,
  errors: { blank: string; missing: string; nonFinite: string },
): { ok: true; name: string; value: number } | { ok: false; error: string } {
  const name = rawName.trim();
  if (name.length === 0) return { ok: false, error: errors.blank };
  if (!Object.prototype.hasOwnProperty.call(params, name)) {
    return { ok: false, error: errors.missing };
  }
  const value = params[name];
  if (!isFiniteNumber(value)) return { ok: false, error: errors.nonFinite };
  return { ok: true, name, value };
}

/**
 * Resolve the panel's selection.
 *
 * @param input - The analysis chosen, which parameter set is active, and the
 *   sweep names as they were typed.
 * @returns The selection with its label, or the identifier of what was wrong.
 */
export function buildStudioAnalysisJobSelection(
  input: StudioAnalysisJobSelectionInput,
): StudioAnalysisJobSelectionResult {
  const label = LABELS[input.analysis];
  const params = activeParams(input.sourceMode, input.modelParams, input.odeParams);

  switch (input.analysis) {
    case "fi_curve":
      return { ok: true, selection: { analysis: "fi_curve" }, label };
    case "sensitivity":
      return { ok: true, selection: { analysis: "sensitivity" }, label };
    case "bifurcation": {
      const resolved = resolveNamedParam(params, input.sweepParam, {
        blank: "analysis_selection_sweep_param_blank",
        missing: "analysis_selection_sweep_param_missing",
        nonFinite: "analysis_selection_sweep_value_invalid",
      });
      if (!resolved.ok) return resolved;
      return {
        ok: true,
        label,
        selection: {
          analysis: "bifurcation",
          sweep: { sweepParam: resolved.name, parameterValue: resolved.value },
        },
      };
    }
    case "heatmap": {
      const x = resolveNamedParam(params, input.sweepParam, {
        blank: "analysis_selection_heatmap_param_x_blank",
        missing: "analysis_selection_heatmap_param_x_missing",
        nonFinite: "analysis_selection_heatmap_value_x_invalid",
      });
      if (!x.ok) return x;
      const y = resolveNamedParam(params, input.sweepParamY, {
        blank: "analysis_selection_heatmap_param_y_blank",
        missing: "analysis_selection_heatmap_param_y_missing",
        nonFinite: "analysis_selection_heatmap_value_y_invalid",
      });
      if (!y.ok) return y;
      if (x.name === y.name) {
        return { ok: false, error: "analysis_selection_heatmap_axes_identical" };
      }
      return {
        ok: true,
        label,
        selection: {
          analysis: "heatmap",
          sweep: {
            sweepParamX: x.name,
            parameterValueX: x.value,
            sweepParamY: y.name,
            parameterValueY: y.value,
          },
        },
      };
    }
    default: {
      const _exhaustive: never = input.analysis;
      return {
        ok: false,
        error: `analysis_selection_kind_unsupported:${String(_exhaustive)}`,
      };
    }
  }
}
