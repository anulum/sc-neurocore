// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio simulation export builder tests

import { describe, expect, it } from "vitest";

import type { SimulateResponse } from "./api/client";
import {
  simulationExportArtefact,
  simulationExportPlan,
  simulationCsvExport,
  simulationCsvFilename,
  simulationCsvText,
  simulationJsonExport,
  simulationJsonFilename,
  simulationSvgExport,
  simulationSvgFilename,
  simulationSvgText,
} from "./simulationExports";

const result: SimulateResponse = {
  time: [0, 0.1, 0.2],
  states: {
    "v<bad>&": [-65, -64.5, -64],
    u: [0.1, 0.2, 0.3],
  },
  current_trace: [10, 10.5, 11],
  spikes: [1],
  spike_count: 1,
  stats: {} as SimulateResponse["stats"],
  dt: 0.1,
  n_steps: 3,
  model_name: "lif/model <x>",
  run_metadata: {} as SimulateResponse["run_metadata"],
};

describe("Studio simulation export builders", () => {
  it("sanitises simulation export filenames", () => {
    expect(simulationJsonFilename(result)).toBe("simulation_lif_model_x.json");
    expect(simulationCsvFilename(result)).toBe("simulation_lif_model_x.csv");
    expect(simulationSvgFilename(result)).toBe("sc_neurocore_lif_model_x.svg");
  });

  it("builds browser download artefacts with canonical filenames and MIME types", () => {
    const jsonExport = simulationJsonExport(result);
    const csvExport = simulationCsvExport(result);
    const svgExport = simulationSvgExport(result);

    expect(jsonExport.filename).toBe("simulation_lif_model_x.json");
    expect(jsonExport.blob.type).toBe("application/json");
    expect(csvExport.filename).toBe("simulation_lif_model_x.csv");
    expect(csvExport.blob.type).toBe("text/csv");
    expect(svgExport.filename).toBe("sc_neurocore_lif_model_x.svg");
    expect(svgExport.blob.type).toBe("image/svg+xml");
  });

  it("builds deterministic CSV output from result traces", () => {
    expect(simulationCsvText(result)).toBe([
      "time,v<bad>&,u,current",
      "0.0000,-65.000000,0.100000,10.0000",
      "0.1000,-64.500000,0.200000,10.5000",
      "0.2000,-64.000000,0.300000,11.0000",
    ].join("\n"));
  });

  it("escapes SVG text labels sourced from model and state names", () => {
    const svg = simulationSvgText(result);

    expect(svg).toContain("v&lt;bad&gt;&amp;");
    expect(svg).toContain("lif/model &lt;x&gt;");
    expect(svg).not.toContain(">v<bad>&<");
    expect(svg).not.toContain(">lif/model <x><");
  });

  it("renders an SVG even when the result has no state traces", () => {
    const emptyResult: SimulateResponse = {
      ...result,
      states: {},
      time: [],
      current_trace: [],
      spikes: [],
      model_name: undefined,
    };

    expect(simulationSvgText(emptyResult)).toContain("<svg");
  });

  it("selects simulation export artefacts by kind", () => {
    expect(simulationExportArtefact("json", result).filename).toBe("simulation_lif_model_x.json");
    expect(simulationExportArtefact("csv", result).filename).toBe("simulation_lif_model_x.csv");
    expect(simulationExportArtefact("svg", result).filename).toBe("sc_neurocore_lif_model_x.svg");
  });

  it("plans available simulation export downloads with injectable writers", () => {
    const plan = simulationExportPlan("csv", result);

    expect(plan.available).toBe(true);
    if (!plan.available) {
      throw new Error("expected available simulation export plan");
    }

    const downloads: Array<{ filename: string; payload: Blob }> = [];
    plan.writeArtefact((payload, filename) => {
      downloads.push({ filename, payload });
    });

    expect(plan.artefact.filename).toBe("simulation_lif_model_x.csv");
    expect(plan.artefact.blob.type).toBe("text/csv");
    expect(downloads).toEqual([{
      filename: "simulation_lif_model_x.csv",
      payload: plan.artefact.blob,
    }]);
  });

  it("plans no-op unavailable JSON and CSV exports", () => {
    const jsonPlan = simulationExportPlan("json", null);
    const csvPlan = simulationExportPlan("csv", null);

    expect(jsonPlan.available).toBe(false);
    expect(csvPlan.available).toBe(false);
    if (jsonPlan.available || csvPlan.available) {
      throw new Error("expected unavailable simulation export plans");
    }

    expect(jsonPlan.runFallback()).toBe(false);
    expect(csvPlan.runFallback()).toBe(false);
  });

  it("plans SVG canvas fallback when no simulation result exists", () => {
    const plan = simulationExportPlan("svg", null);

    expect(plan.available).toBe(false);
    if (plan.available) {
      throw new Error("expected unavailable SVG export plan");
    }

    let fallbackCount = 0;
    expect(plan.runFallback(() => {
      fallbackCount += 1;
      return true;
    })).toBe(true);
    expect(fallbackCount).toBe(1);
  });
});
