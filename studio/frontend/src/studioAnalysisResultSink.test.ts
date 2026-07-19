// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — studioAnalysisResultSink pure tests
import { describe, expect, it } from "vitest";

import type {
  AnalysisJobKind,
  AnalysisJobResult,
  AnalysisResultMetadata,
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
} from "./studioAnalysisState";
import {
  studioAnalysisResultSink,
  studioAnalysisResultViewTab,
} from "./studioAnalysisResultSink";

function meta(analysisType: string): AnalysisResultMetadata {
  return {
    analysis_type: analysisType,
    evidence_classification: "analysis",
    input_sha256: "a".repeat(64),
    output_keys: ["x"],
    result_sha256: "b".repeat(64),
    schema_version: "studio.analysis-result.v1",
    source: "model",
    status: "completed",
  };
}

const fi: FICurveResponse = {
  analysis_metadata: meta("fi_curve"),
  currents: [0, 1],
  rates: [0, 5],
};

const bif: BifurcationResponse = {
  analysis_metadata: meta("bifurcation"),
  param_name: "tau",
  param_values: [1, 2],
  attractors: [[1], [2]],
};

const heat: HeatmapResponse = {
  analysis_metadata: meta("heatmap"),
  param_x: "tau",
  param_y: "C",
  x_values: [1, 2],
  y_values: [3, 4],
  rates: [
    [0, 1],
    [2, 3],
  ],
  rate_min: 0,
  rate_max: 3,
};

const sens: SensitivityResponse = {
  analysis_metadata: meta("sensitivity"),
  base_rate: 10,
  sensitivities: [
    { param: "tau", sensitivity: 0.5, rate_minus: 9, rate_plus: 11 },
  ],
};

describe("studioAnalysisResultSink success paths", () => {
  it("matches legacy result patches for all four kinds and sets view tabs", () => {
    const cases: Array<{
      kind: AnalysisJobKind;
      result: AnalysisJobResult;
      expected:
        | ReturnType<typeof studioFICurveResultState>
        | ReturnType<typeof studioBifurcationResultState>
        | ReturnType<typeof studioHeatmapResultState>
        | ReturnType<typeof studioSensitivityResultState>;
      tab: ReturnType<typeof studioAnalysisResultViewTab>;
    }> = [
      {
        kind: "fi_curve",
        result: fi,
        expected: studioFICurveResultState(fi),
        tab: "fi-curve",
      },
      {
        kind: "bifurcation",
        result: bif,
        expected: studioBifurcationResultState(bif),
        tab: "bifurcation",
      },
      {
        kind: "heatmap",
        result: heat,
        expected: studioHeatmapResultState(heat),
        tab: "heatmap",
      },
      {
        kind: "sensitivity",
        result: sens,
        expected: studioSensitivityResultState(sens),
        tab: "sensitivity",
      },
    ];

    for (const entry of cases) {
      expect(studioAnalysisResultViewTab(entry.kind)).toBe(entry.tab);
      const sunk = studioAnalysisResultSink(entry.kind, entry.result);
      expect(sunk.ok).toBe(true);
      if (!sunk.ok) {
        continue;
      }
      expect(sunk.patch).toMatchObject({
        ...entry.expected,
        activeTab: entry.tab,
        error: null,
      });
      expect(sunk.patch.isSimulating).toBe(false);
    }
  });

  it("preserves result object identity via legacy patch helpers", () => {
    const sunk = studioAnalysisResultSink("fi_curve", fi);
    expect(sunk.ok).toBe(true);
    if (!sunk.ok) {
      return;
    }
    expect(sunk.patch.activeTab).toBe("fi-curve");
    if (sunk.patch.activeTab !== "fi-curve") {
      return;
    }
    expect(sunk.patch.fiResult).toBe(fi);
  });
});

describe("studioAnalysisResultSink fail-closed", () => {
  it("rejects missing metadata analysis_type", () => {
    const bad: FICurveResponse = {
      ...fi,
      analysis_metadata: {
        ...fi.analysis_metadata,
        analysis_type: "",
      },
    };
    expect(studioAnalysisResultSink("fi_curve", bad)).toEqual({
      ok: false,
      error: "analysis_result_sink_metadata_missing",
    });
  });

  it("rejects kind/metadata mismatch without writing patches", () => {
    expect(studioAnalysisResultSink("heatmap", fi)).toEqual({
      ok: false,
      error: "analysis_result_sink_kind_mismatch:heatmap:fi_curve",
    });
  });

  it("rejects metadata-aligned but shape-invalid payloads", () => {
    const badFi = {
      analysis_metadata: meta("fi_curve"),
      currents: "nope",
      rates: [0],
    } as unknown as FICurveResponse;
    expect(studioAnalysisResultSink("fi_curve", badFi)).toEqual({
      ok: false,
      error: "analysis_result_sink_fi_curve_shape_invalid",
    });

    const badBif = {
      analysis_metadata: meta("bifurcation"),
      param_name: "tau",
      param_values: "nope",
      attractors: [],
    } as unknown as BifurcationResponse;
    expect(studioAnalysisResultSink("bifurcation", badBif)).toEqual({
      ok: false,
      error: "analysis_result_sink_bifurcation_shape_invalid",
    });

    const badHeat = {
      analysis_metadata: meta("heatmap"),
      param_x: "tau",
      param_y: "C",
      x_values: [1],
      y_values: [2],
      rates: "nope",
      rate_min: 0,
      rate_max: 1,
    } as unknown as HeatmapResponse;
    expect(studioAnalysisResultSink("heatmap", badHeat)).toEqual({
      ok: false,
      error: "analysis_result_sink_heatmap_shape_invalid",
    });

    const badSens = {
      analysis_metadata: meta("sensitivity"),
      base_rate: 1,
      sensitivities: "nope",
    } as unknown as SensitivityResponse;
    expect(studioAnalysisResultSink("sensitivity", badSens)).toEqual({
      ok: false,
      error: "analysis_result_sink_sensitivity_shape_invalid",
    });
  });

  it("rejects sensitivity entries that are not records", () => {
    const badSens: SensitivityResponse = {
      analysis_metadata: meta("sensitivity"),
      base_rate: 1,
      sensitivities: [null as unknown as SensitivityResponse["sensitivities"][number]],
    };
    expect(studioAnalysisResultSink("sensitivity", badSens)).toEqual({
      ok: false,
      error: "analysis_result_sink_sensitivity_shape_invalid",
    });
  });

  it("rejects blank bifurcation param_name", () => {
    const bad: BifurcationResponse = {
      ...bif,
      param_name: "",
    };
    expect(studioAnalysisResultSink("bifurcation", bad)).toEqual({
      ok: false,
      error: "analysis_result_sink_bifurcation_shape_invalid",
    });
  });
});
