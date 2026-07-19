// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — studioAnalysisJobSelection pure resolver tests
import { describe, expect, it } from "vitest";

import { buildAnalysisJobRequest } from "./analysisJobRequest";
import {
  buildStudioAnalysisJobSelection,
  type StudioAnalysisJobSelectionInput,
} from "./studioAnalysisJobSelection";
import { studioSimulationConfigInput } from "./studioSimulationConfigInput";

const modelParams = { tau: 10, capacitance: 1, e_l: -65 };
const odeParams = { tau: 20, e_l: -65, C: 1 };
const base: Omit<StudioAnalysisJobSelectionInput, "analysis"> = {
  sourceMode: "model",
  modelParams,
  odeParams,
  sweepParam: "tau",
  sweepParamY: "capacitance",
};
const simInput = studioSimulationConfigInput({
  sourceMode: "model",
  selectedModelName: "lif",
  modelParams,
  equations: ["dv/dt = 0"],
  threshold: "v > -50",
  reset: "v = -65",
  odeParams,
  odeInit: { v: -65 },
  dt: 0.1,
  duration: 100,
  current: 12,
  protocol: "constant",
});

describe("buildStudioAnalysisJobSelection success", () => {
  it("resolves fi_curve and sensitivity in both modes without sweeps", () => {
    for (const sourceMode of ["model", "ode"] as const) {
      for (const analysis of ["fi_curve", "sensitivity"] as const) {
        const result = buildStudioAnalysisJobSelection({
          ...base,
          sourceMode,
          analysis,
          sweepParam: "",
          sweepParamY: "",
        });
        expect(result).toEqual({
          ok: true,
          label: analysis === "fi_curve" ? "f-I curve" : "sensitivity",
          selection: { analysis },
        });
        if (result.ok) {
          expect(buildAnalysisJobRequest(simInput, result.selection).ok).toBe(true);
        }
      }
    }
  });

  it("resolves bifurcation and heatmap from active mode with trimmed names", () => {
    expect(
      buildStudioAnalysisJobSelection({
        ...base,
        analysis: "bifurcation",
        sweepParam: "  tau  ",
      }),
    ).toEqual({
      ok: true,
      label: "bifurcation",
      selection: {
        analysis: "bifurcation",
        sweep: { sweepParam: "tau", parameterValue: 10 },
      },
    });
    expect(
      buildStudioAnalysisJobSelection({
        ...base,
        sourceMode: "ode",
        analysis: "bifurcation",
        sweepParam: "tau",
      }),
    ).toEqual({
      ok: true,
      label: "bifurcation",
      selection: {
        analysis: "bifurcation",
        sweep: { sweepParam: "tau", parameterValue: 20 },
      },
    });

    const heatModel = buildStudioAnalysisJobSelection({
      ...base,
      analysis: "heatmap",
      sweepParam: " tau ",
      sweepParamY: " capacitance ",
    });
    expect(heatModel).toEqual({
      ok: true,
      label: "heatmap",
      selection: {
        analysis: "heatmap",
        sweep: {
          sweepParamX: "tau",
          parameterValueX: 10,
          sweepParamY: "capacitance",
          parameterValueY: 1,
        },
      },
    });
    const heatOde = buildStudioAnalysisJobSelection({
      ...base,
      sourceMode: "ode",
      analysis: "heatmap",
      sweepParam: "tau",
      sweepParamY: "C",
    });
    expect(heatOde.ok).toBe(true);
    if (heatOde.ok) {
      expect(heatOde.selection).toEqual({
        analysis: "heatmap",
        sweep: {
          sweepParamX: "tau",
          parameterValueX: 20,
          sweepParamY: "C",
          parameterValueY: 1,
        },
      });
      expect(buildAnalysisJobRequest(simInput, heatOde.selection).ok).toBe(true);
    }
  });
});

describe("buildStudioAnalysisJobSelection fail-closed", () => {
  it("rejects bifurcation blank, missing, and non-finite values", () => {
    expect(
      buildStudioAnalysisJobSelection({
        ...base,
        analysis: "bifurcation",
        sweepParam: "   ",
      }),
    ).toEqual({ ok: false, error: "analysis_selection_sweep_param_blank" });
    expect(
      buildStudioAnalysisJobSelection({
        ...base,
        analysis: "bifurcation",
        sweepParam: "missing",
      }),
    ).toEqual({ ok: false, error: "analysis_selection_sweep_param_missing" });
    expect(
      buildStudioAnalysisJobSelection({
        ...base,
        analysis: "bifurcation",
        modelParams: { tau: Number.NaN },
        sweepParam: "tau",
      }),
    ).toEqual({ ok: false, error: "analysis_selection_sweep_value_invalid" });
    expect(
      buildStudioAnalysisJobSelection({
        ...base,
        sourceMode: "ode",
        analysis: "bifurcation",
        odeParams: { tau: Number.POSITIVE_INFINITY },
        sweepParam: "tau",
      }),
    ).toEqual({ ok: false, error: "analysis_selection_sweep_value_invalid" });
  });

  it("rejects heatmap blank/missing/non-finite/identical and wrong mode maps", () => {
    expect(
      buildStudioAnalysisJobSelection({
        ...base,
        analysis: "heatmap",
        sweepParam: "",
        sweepParamY: "capacitance",
      }),
    ).toEqual({ ok: false, error: "analysis_selection_heatmap_param_x_blank" });
    expect(
      buildStudioAnalysisJobSelection({
        ...base,
        analysis: "heatmap",
        sweepParam: "tau",
        sweepParamY: "  ",
      }),
    ).toEqual({ ok: false, error: "analysis_selection_heatmap_param_y_blank" });
    expect(
      buildStudioAnalysisJobSelection({
        ...base,
        analysis: "heatmap",
        sweepParam: "nope",
        sweepParamY: "capacitance",
      }),
    ).toEqual({ ok: false, error: "analysis_selection_heatmap_param_x_missing" });
    expect(
      buildStudioAnalysisJobSelection({
        ...base,
        analysis: "heatmap",
        sweepParam: "tau",
        sweepParamY: "nope",
      }),
    ).toEqual({ ok: false, error: "analysis_selection_heatmap_param_y_missing" });
    expect(
      buildStudioAnalysisJobSelection({
        ...base,
        analysis: "heatmap",
        modelParams: { tau: Number.NaN, capacitance: 1 },
        sweepParam: "tau",
        sweepParamY: "capacitance",
      }),
    ).toEqual({ ok: false, error: "analysis_selection_heatmap_value_x_invalid" });
    expect(
      buildStudioAnalysisJobSelection({
        ...base,
        analysis: "heatmap",
        modelParams: { tau: 10, capacitance: Number.NaN },
        sweepParam: "tau",
        sweepParamY: "capacitance",
      }),
    ).toEqual({ ok: false, error: "analysis_selection_heatmap_value_y_invalid" });
    expect(
      buildStudioAnalysisJobSelection({
        ...base,
        analysis: "heatmap",
        sweepParam: "tau",
        sweepParamY: "tau",
      }),
    ).toEqual({ ok: false, error: "analysis_selection_heatmap_axes_identical" });
    expect(
      buildStudioAnalysisJobSelection({
        sourceMode: "model",
        analysis: "bifurcation",
        modelParams: { only_model: 1 },
        odeParams: { tau: 99 },
        sweepParam: "tau",
        sweepParamY: "",
      }),
    ).toEqual({ ok: false, error: "analysis_selection_sweep_param_missing" });
    expect(
      buildStudioAnalysisJobSelection({
        sourceMode: "ode",
        analysis: "bifurcation",
        modelParams: { tau: 99 },
        odeParams: { only_ode: 2 },
        sweepParam: "tau",
        sweepParamY: "",
      }),
    ).toEqual({ ok: false, error: "analysis_selection_sweep_param_missing" });
  });
});
