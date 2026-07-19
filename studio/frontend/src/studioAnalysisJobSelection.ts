// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Pure UI/store analysis selection resolver

/**
 * Resolves UI/store selection into W09 {@link AnalysisJobSelection}.
 * Does not build payloads, invent sweep defaults, or call the network.
 */

import type { AnalysisJobKind } from "./api/client";
import type { AnalysisJobSelection } from "./analysisJobRequest";
import type { StudioSimulationSourceMode } from "./studioSimulationConfig";

export type StudioAnalysisJobSelectionResult =
  | { ok: true; selection: AnalysisJobSelection; label: string }
  | { ok: false; error: string };

export interface StudioAnalysisJobSelectionInput {
  analysis: AnalysisJobKind;
  sourceMode: StudioSimulationSourceMode;
  modelParams: Record<string, number>;
  odeParams: Record<string, number>;
  sweepParam: string;
  sweepParamY: string;
}

const LABELS: Readonly<Record<AnalysisJobKind, string>> = {
  fi_curve: "f-I curve",
  sensitivity: "sensitivity",
  bifurcation: "bifurcation",
  heatmap: "heatmap",
};

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function activeParams(
  sourceMode: StudioSimulationSourceMode,
  modelParams: Record<string, number>,
  odeParams: Record<string, number>,
): Record<string, number> {
  return sourceMode === "model" ? modelParams : odeParams;
}

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
 * Resolve analysis kind + sweep names into a typed W09 selection.
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
