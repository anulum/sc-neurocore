// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — analysis job request policy tests
import { describe, expect, it } from "vitest";

import {
  buildAnalysisJobRequest,
  validateStudioSimulationCoreNumerics,
} from "./analysisJobRequest";
import {
  studioBifurcationRequest,
  studioFICurveRequest,
  studioHeatmapRequest,
  studioSimulationConfig,
  type StudioSimulationConfigInput,
} from "./studioSimulationConfig";

const modelInput: StudioSimulationConfigInput = {
  sourceMode: "model",
  selectedModelName: "lif",
  modelParams: { tau: 10, capacitance: 1 },
  equations: ["dv/dt = -(v - e_l) / tau + i"],
  threshold: "v > -50",
  reset: "v = -65",
  odeParams: { tau: 20, e_l: -65 },
  odeInit: { v: -65 },
  dt: 0.1,
  duration: 100,
  current: 12,
  protocol: "constant",
};

const odeInput: StudioSimulationConfigInput = {
  ...modelInput,
  sourceMode: "ode",
  selectedModelName: "",
};

describe("buildAnalysisJobRequest valid shapes", () => {
  it("builds fi_curve for model and ODE using Studio FI defaults", () => {
    const model = buildAnalysisJobRequest(modelInput, { analysis: "fi_curve" });
    expect(model.ok).toBe(true);
    if (!model.ok) {
      return;
    }
    expect(model.value.analysis).toBe("fi_curve");
    expect(model.value.payload).toEqual(
      studioFICurveRequest(studioSimulationConfig(modelInput), modelInput.current),
    );

    const ode = buildAnalysisJobRequest(odeInput, { analysis: "fi_curve" });
    expect(ode.ok).toBe(true);
    if (!ode.ok) {
      return;
    }
    expect(ode.value.payload).toEqual(
      studioFICurveRequest(studioSimulationConfig(odeInput), odeInput.current),
    );
  });

  it("builds sensitivity as the base simulation payload only", () => {
    const model = buildAnalysisJobRequest(modelInput, { analysis: "sensitivity" });
    expect(model.ok).toBe(true);
    if (!model.ok) {
      return;
    }
    expect(model.value).toEqual({
      analysis: "sensitivity",
      payload: studioSimulationConfig(modelInput),
    });

    const ode = buildAnalysisJobRequest(odeInput, { analysis: "sensitivity" });
    expect(ode.ok).toBe(true);
    if (!ode.ok) {
      return;
    }
    expect(ode.value.payload).toEqual(studioSimulationConfig(odeInput));
  });

  it("builds bifurcation with retained Studio sweep defaults", () => {
    const sweep = { sweepParam: "tau", parameterValue: 10 };
    const model = buildAnalysisJobRequest(modelInput, {
      analysis: "bifurcation",
      sweep,
    });
    expect(model.ok).toBe(true);
    if (!model.ok) {
      return;
    }
    expect(model.value.analysis).toBe("bifurcation");
    expect(model.value.payload).toEqual(
      studioBifurcationRequest(studioSimulationConfig(modelInput), sweep),
    );
  });

  it("builds heatmap with retained Studio two-axis defaults", () => {
    const sweep = {
      sweepParamX: "tau",
      parameterValueX: 10,
      sweepParamY: "capacitance",
      parameterValueY: 2,
    };
    const model = buildAnalysisJobRequest(modelInput, {
      analysis: "heatmap",
      sweep,
    });
    expect(model.ok).toBe(true);
    if (!model.ok) {
      return;
    }
    expect(model.value.analysis).toBe("heatmap");
    expect(model.value.payload).toEqual(
      studioHeatmapRequest(studioSimulationConfig(modelInput), sweep),
    );
  });
});

describe("buildAnalysisJobRequest fail-closed conditions", () => {
  it("rejects non-finite core numerics and parameter maps", () => {
    expect(
      buildAnalysisJobRequest({ ...modelInput, dt: Number.NaN }, { analysis: "fi_curve" }),
    ).toEqual({ ok: false, error: "analysis_request_dt_invalid" });
    expect(
      buildAnalysisJobRequest(
        { ...modelInput, duration: Number.POSITIVE_INFINITY },
        { analysis: "fi_curve" },
      ),
    ).toEqual({ ok: false, error: "analysis_request_duration_invalid" });
    expect(
      buildAnalysisJobRequest({ ...modelInput, current: Number.NaN }, { analysis: "fi_curve" }),
    ).toEqual({ ok: false, error: "analysis_request_current_invalid" });
    expect(
      buildAnalysisJobRequest(
        { ...modelInput, modelParams: { tau: Number.NaN } },
        { analysis: "sensitivity" },
      ),
    ).toEqual({ ok: false, error: "analysis_request_model_params_invalid" });
    expect(validateStudioSimulationCoreNumerics(modelInput)).toBeNull();
  });

  it("rejects blank sweep names and non-finite sweep values", () => {
    expect(
      buildAnalysisJobRequest(modelInput, {
        analysis: "bifurcation",
        sweep: { sweepParam: "   ", parameterValue: 10 },
      }),
    ).toEqual({ ok: false, error: "analysis_request_sweep_param_blank" });
    expect(
      buildAnalysisJobRequest(modelInput, {
        analysis: "bifurcation",
        sweep: { sweepParam: "tau", parameterValue: Number.NaN },
      }),
    ).toEqual({ ok: false, error: "analysis_request_sweep_value_invalid" });
    expect(
      buildAnalysisJobRequest(modelInput, {
        analysis: "heatmap",
        sweep: {
          sweepParamX: "",
          parameterValueX: 1,
          sweepParamY: "c",
          parameterValueY: 2,
        },
      }),
    ).toEqual({ ok: false, error: "analysis_request_heatmap_param_blank" });
  });

  it("rejects identical heatmap axes", () => {
    expect(
      buildAnalysisJobRequest(modelInput, {
        analysis: "heatmap",
        sweep: {
          sweepParamX: "tau",
          parameterValueX: 10,
          sweepParamY: "tau",
          parameterValueY: 2,
        },
      }),
    ).toEqual({ ok: false, error: "analysis_request_heatmap_axes_identical" });
    expect(
      buildAnalysisJobRequest(modelInput, {
        analysis: "heatmap",
        sweep: {
          sweepParamX: "  tau  ",
          parameterValueX: 10,
          sweepParamY: "tau",
          parameterValueY: 2,
        },
      }),
    ).toEqual({ ok: false, error: "analysis_request_heatmap_axes_identical" });
  });

  it("returns only analysis and payload keys on success", () => {
    const built = buildAnalysisJobRequest(modelInput, { analysis: "fi_curve" });
    expect(built.ok).toBe(true);
    if (!built.ok) {
      return;
    }
    expect(Object.keys(built.value).sort()).toEqual(["analysis", "payload"]);
  });
});
